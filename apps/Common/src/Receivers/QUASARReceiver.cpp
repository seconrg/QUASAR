#include <Receivers/QUASARReceiver.h>

using namespace quasar;

QUASARReceiver::QUASARReceiver(QuadSet& quadSet, 
                               DepthPeelingRenderer& remoteRendererDP,
                               DeferredRenderer& remoteRenderer,
                               Scene& remoteScene,
                               uint maxLayers, 
                               const std::string& videoURL, 
                               const std::string& proxiesURL)
    : quadSet(quadSet)
    , maxLayers(maxLayers)
    , remoteRendererDP(remoteRendererDP)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , videoURL(videoURL)
    , proxiesURL(proxiesURL)
    , remoteCamera(quadSet.getSize())
    , remoteCameraWideFOV(quadSet.getSize())
    , frameGenerator(quadSet)
    , videoAtlasTexture({
        // .width = 2 * quadSet.getSize().x,
        // .height = 3 * quadSet.getSize().y,
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, videoURL)
    , alphaAtlasTexture({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_R8,
        .format = GL_RED,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , alphaCodec(alphaAtlasTexture.width, alphaAtlasTexture.height)
    , residualFrameMesh(quadSet, videoAtlasTexture, alphaAtlasTexture)
    , bufferPool(quadSet.getSize(), maxLayers)
    , DataReceiverTCP(proxiesURL)
{
    meshes.reserve(maxLayers);
    referenceFrames.resize(maxLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    // Untile texture atlas
    glm::vec4 textureExtent(0.0f, 0.0f, 0.5f, 1.0f / 3.0f);
    for (int layer = 0; layer < maxLayers; layer++) {
        meshes.emplace_back(
            quadSet, videoAtlasTexture, alphaAtlasTexture, textureExtent);

        textureExtent.x += 0.5f;
        if (textureExtent.x >= 1.0f) {
            textureExtent.x = 0.0f;
            textureExtent.y += 1.0f / 3.0f;
        }
        textureExtent.z = textureExtent.x + 0.5f;
        textureExtent.w = textureExtent.y + 1.0f / 3.0f;
    }
    residualFrameMesh.setTextureExtent(textureExtent);

    frameInUse = std::make_shared<Frame>(bufferPool);
    framePending = std::make_shared<Frame>(bufferPool);

    frameFree = framePending;

    uint numHidLayers = maxLayers - 1;
    frameRTsHidLayer_noTone.reserve(numHidLayers);

    // Setup hidden layers and wide fov RTs
    RenderTargetCreateParams rtParams = {
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    };
    for (int layer = 0; layer < numHidLayers; layer++) {
        frameRTsHidLayer_noTone.emplace_back(rtParams);
    }

    cv.notify_one();

    threadPool = std::make_unique<BS::thread_pool<>>(4);

    if (!proxiesURL.empty()) {
        spdlog::info("Created QUASARReceiver that recvs from URL: tcp://{}", proxiesURL);
    }
}

QUASARReceiver::QUASARReceiver(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        uint maxLayers, float remoteFOV, float remoteFOVWide,
        const std::string& videoURL, const std::string& proxiesURL)
        : QUASARReceiver(quadSet, remoteRendererDP, remoteRenderer, remoteScene, maxLayers, videoURL, proxiesURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());

    remoteCameraWideFOV.setFovyDegrees(remoteFOVWide);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
}

void QUASARReceiver::copyPoseToCamera(PerspectiveCamera& camera) {
    camera.setViewMatrix(remoteCamera.getViewMatrix());
    camera.setProjectionMatrix(remoteCamera.getProjectionMatrix());
}

void QUASARReceiver::setDrawState(QuadMesh::DrawState drawState) {
    for (auto& mesh : meshes) {
        mesh.setDrawState(drawState);
    }
    residualFrameMesh.setDrawState(drawState);
}

void QUASARReceiver::onDataReceived(const std::vector<char>& data) {
    loadFromMemory(data);
}

QuadFrame::FrameType QUASARReceiver::recvData() {
    QuadFrame::FrameType frameType = QuadFrame::FrameType::NONE;

    if (proxiesURL.empty()) {
        return frameType;
    }

    if (!videoAtlasTexture.containsFrames()) {
        return frameType;
    }

    // Wait for a frame that has been written to
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            return frameType;
        }

        if (videoAtlasTexture.getLatestPoseID() < framePending->poseID) { // Video is behind, wait until video catches up
            return frameType;
        }

        frame = framePending;
        framePending.reset();
        frameInUse = frame;
    }

    // If video is ahead, search for a previous frame
    if (!videoAtlasTexture.containsFrameWithPoseID(frame->poseID)) {
        // This means we dropped a video frame. We have to wait for the next reference frame to resync
        waitUntilReferenceFrame = true;
    }
    else if (!waitUntilReferenceFrame || (waitUntilReferenceFrame && frame->frameType == QuadFrame::FrameType::REFERENCE)) {
        // Update color texture
        videoAtlasTexture.bind();
        videoAtlasTexture.draw(frame->poseID);

        // Update alpha texture
        alphaAtlasTexture.bind();
        alphaAtlasTexture.loadFromData(frame->bufferPool.alphaData.data());

        // For debug: save a copy of alpha texture
        Path debugPath = Path(std::string("debug_alpha_{}.png", frame->poseID));
        alphaAtlasTexture.writeToPNG(debugPath);

        // Reconstruct meshes from frame
        frameType = reconstructFrame(frame);

        // Video and proxies are synced now, no need to wait for reference frame anymore
        waitUntilReferenceFrame = false;
    }

    // Reset frame
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return frameType;
}

RenderStats QUASARReceiver::generateFrame(bool createResidualFrame, bool showNormals, bool showDepth) {
    stats = { 0 };

    spdlog::info("Generating frame with {} hidden layers", maxLayers - 2);

    // Render local scenes
    // Where should we get the remote Scene from?

    auto quadsGenerator = frameGenerator.getQuadsGenerator();
    
    double renderStartTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera, false);
    stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - renderStartTime);
    
    int numLayers = maxLayers;
    for (int layer = 1; layer < numLayers-1; layer++) {

        int hiddenLayerIndex = layer - 1;
        auto& cameraToUse = (layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV;
        auto& frameToUse_noTone = frameRTsHidLayer_noTone[hiddenLayerIndex];
        auto& meshToUse = meshes[layer];

        double startTime = timeutils::getTimeMicros();
        remoteRendererDP.peelingLayers[layer].blit(frameToUse_noTone);

        stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        // Generate reference frame 
        auto oldParams = quadsGenerator->params;
        quadsGenerator->params.depthThreshold = 1e-3f;
        quadsGenerator->params.maxIterForceMerge = 4;
        quadsGenerator->params.expandEdges = false;

        frameGenerator.createReferenceFrame(
            frameToUse_noTone, 
            cameraToUse,
            meshToUse,
            referenceFrames[layer],
            layer
        );

        stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;
        stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;
        spdlog::info("    Layer {}: Created reference proxies in {} ms", 
                     layer, frameGenerator.stats.createQuadsTimeMs);
        spdlog::info("    Layer {}: Created reference frame mesh in {} ms", 
                     layer, frameGenerator.stats.createMeshTimeMs);
        
        tonemapper.setUniforms(frameToUse_noTone);
        tonemapper.drawToRenderTarget(remoteRenderer, frameToUse_noTone, false);
        quadsGenerator->params = oldParams;

    }
    spdlog::info("Total render time ms: {}", stats.totalRenderTimeMs);
    spdlog::info("Total create mesh time ms: {}", stats.totalCreateMeshTimeMs);
    return renderStats;
}

QuadFrame::FrameType QUASARReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    videoAtlasTexture.loadFromFile(colorFileName, true, true);

    // Read alpha data
    Path alphaFileName = dataPath / "alpha.png";
    alphaAtlasTexture.loadFromFile(alphaFileName, true, false);

    // Read previous camera data
    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileNamePrev);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);
    frameInUse->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    // Read metadata (viewSphereDiameter and wide FOV)
    const auto& metadata = FileIO::loadFromBinaryFile(dataPath / "metadata.bin");
    Params params;
    std::memcpy(&params, metadata.data(), metadata.size());

    if (params.numLayers != maxLayers) {
        spdlog::warn("Loaded number of layers {} does not match initialized number of layers {}", params.numLayers, maxLayers);
        maxLayers = params.numLayers;
    }
    spdlog::debug("Loaded wide FOV: {}", params.wideFOV);
    spdlog::debug("Loaded view sphere diameter: {}", params.viewSphereDiameter);

    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
    setViewSphereDiameter(params.viewSphereDiameter);

    // Read reference frames
    for (int layer = 0; layer < maxLayers ; layer++) {
        referenceFrames[layer].loadFromFiles(dataPath, layer);
    }
    stats.loadTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Decompress reference frame
    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::REFERENCE;
    size_t refSize = frameInUse->decompressReferenceHiddenLayersWideFOV(threadPool, referenceFrames);
    stats.decompressTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update reference GPU buffers
    if (refSize > 0) reconstructFrame(frameInUse);

    startTime = timeutils::getTimeMicros();

    // Read camera data
    Path cameraFileName = dataPath / "camera.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileName);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);
    frameInUse->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    // Read residual frame
    residualFrame.loadFromFiles(dataPath);
    stats.loadTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Decompress residual frame
    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::RESIDUAL;
    size_t resSize = frameInUse->decompressResidualFrame(threadPool, residualFrame);
    frameInUse->decompressHiddenLayersWideFOV(threadPool, referenceFrames);
    stats.decompressTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update residual GPU buffers
    if (resSize > 0) reconstructFrame(frameInUse);

    return frameInUse->frameType;
}

QuadFrame::FrameType QUASARReceiver::loadFromMemory(const std::vector<char>& inputData) {
    double startTime = timeutils::getTimeMicros();

    spdlog::debug("Loading inputData of size {}", inputData.size());

    const char* ptr = inputData.data();

    // Read header
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    size_t expectedSize = header.getSize();
    if (inputData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                 std::to_string(inputData.size()) +
                                 " is smaller than expected from header " +
                                 std::to_string(expectedSize));
    }

    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return frameFree != nullptr; });
        frame = frameFree;
        frameFree.reset();
    }

    frame->poseID = header.poseID;
    frame->frameType = header.frameType;

    // Read parameter data
    maxLayers = header.params.numLayers;
    setViewSphereDiameter(header.params.viewSphereDiameter);
    remoteCameraWideFOV.setFovyDegrees(header.params.wideFOV);

    spdlog::debug("Loading camera size: {}", header.cameraSize);
    spdlog::debug("Loading alpha size: {}", header.alphaSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

    // Read camera data
    frame->cameraPose.loadFromMemory(ptr, header.cameraSize);
    ptr += header.cameraSize;

    // Read alpha data
    alphaCodec.decompress(ptr, frame->bufferPool.alphaData, header.alphaSize);
    ptr += header.alphaSize;

    // Read geometry data
    const char* layerPtr = ptr;
    uint32_t layerSize;

    // Load all frame data in parallel
    std::vector<std::future<size_t>> futures;
    // Visible layer
    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        std::memcpy(&layerSize, layerPtr, sizeof(uint32_t));
        const char* dataPtr = layerPtr + sizeof(uint32_t);

        futures.emplace_back(threadPool->submit_task([&, dataPtr, layerSize]() {
            return referenceFrames[0].loadFromMemory(dataPtr, layerSize);
        }));

        layerPtr += sizeof(uint32_t) + layerSize;
    }
    else {
        std::memcpy(&layerSize, layerPtr, sizeof(uint32_t));
        const char* dataPtr = layerPtr + sizeof(uint32_t);

        futures.emplace_back(threadPool->submit_task([&, dataPtr, layerSize]() {
            return residualFrame.loadFromMemory(dataPtr, layerSize);
        }));

        layerPtr += sizeof(uint32_t) + layerSize;
    }
    // copy hidden layers from 
    // Hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; layer++) {
        std::memcpy(&layerSize, layerPtr, sizeof(uint32_t));
        const char* dataPtr = layerPtr + sizeof(uint32_t);

        futures.emplace_back(threadPool->submit_task([&, layer, dataPtr, layerSize]() {
            return referenceFrames[layer].loadFromMemory(dataPtr, layerSize);
        }));

        layerPtr += sizeof(uint32_t) + layerSize;
    }

    for (auto& f : futures) f.get();

    stats.loadTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Decompress (asynchronous)
    startTime = timeutils::getTimeMicros();
    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        frame->decompressReferenceHiddenLayersWideFOV(threadPool, referenceFrames);
    }
    else {
        frame->decompressResidualHiddenLayersWideFOV(threadPool, referenceFrames, residualFrame);
    }
    stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;
}

QuadFrame::FrameType QUASARReceiver::reconstructFrame(std::shared_ptr<Frame> frame) {
    if (frame->frameType == QuadFrame::FrameType::NONE) {
        return QuadFrame::FrameType::NONE;
    }

    spdlog::debug("Reconstructing {} Frame...", frame->frameType == QuadFrame::FrameType::REFERENCE ? "Reference" : "Residual");
    frame->cameraPose.copyPoseToCamera(remoteCamera);
    frame->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    const glm::vec2& gBufferSize = quadSet.getSize();
    double startTime = timeutils::getTimeMicros();
    // Reconstruct visible layer
    if (frame->frameType == QuadFrame::FrameType::REFERENCE) {
        // Transfer proxies to GPU for reconstruction
        auto sizes = quadSet.loadFromMemory(bufferPool.uncompressedQuads[0], bufferPool.uncompressedOffsets[0]);
        referenceFrames[0].numQuads = sizes.numQuads;
        referenceFrames[0].numDepthOffsets = sizes.numDepthOffsets;
        stats.transferTimeMs = quadSet.stats.transferTimeMs;

        // Using GPU buffers, reconstruct mesh using proxies
        const auto& cameraToUse = getCameraToUse(0);
        startTime = timeutils::getTimeMicros();
        meshes[0].appendQuads(quadSet, gBufferSize);
        meshes[0].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[0].getBufferSizes();
        stats.totalTriangles = meshBufferSizes.numIndices / 3;
        stats.sizes = sizes;

        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }
    else {
        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.loadFromMemory(bufferPool.uncompressedQuads[0], bufferPool.uncompressedOffsets[0]);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.transferTimeMs = quadSet.stats.transferTimeMs;

        // Using GPU buffers, update reference frame mesh using proxies
        startTime = timeutils::getTimeMicros();
        meshes[0].appendQuads(quadSet, gBufferSize, false /* not a reference frame */);
        meshes[0].createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto refMeshBufferSizes = meshes[0].getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;

        // Transfer revealed proxies to GPU for reconstruction
        auto sizesRevealed = quadSet.loadFromMemory(bufferPool.uncompressedQuadsRevealed, bufferPool.uncompressedOffsetsRevealed);
        residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
        residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
        stats.transferTimeMs += quadSet.stats.transferTimeMs;

        // Using GPU buffers, reconstruct revealed mesh using proxies
        startTime = timeutils::getTimeMicros();
        residualFrameMesh.appendQuads(quadSet, gBufferSize);
        residualFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.createMeshTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto resMeshBufferSizes = residualFrameMesh.getBufferSizes();
        stats.totalTriangles += resMeshBufferSizes.numIndices / 3;
        stats.sizes = sizesUpdated + sizesRevealed;
    }

    // Reconstruct hidden layers and wide FOV
    // Temporarily disable for performance testing
    
    for (int layer = 1; layer < maxLayers; layer++) {
        auto sizes = quadSet.loadFromMemory(bufferPool.uncompressedQuads[layer], bufferPool.uncompressedOffsets[layer]);
        referenceFrames[layer].numQuads = sizes.numQuads;
        referenceFrames[layer].numDepthOffsets = sizes.numDepthOffsets;
        stats.transferTimeMs += quadSet.stats.transferTimeMs;

        const auto& cameraToUse = getCameraToUse(layer);
        startTime = timeutils::getTimeMicros();
        meshes[layer].appendQuads(quadSet, gBufferSize);
        meshes[layer].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.createMeshTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[layer].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }

    return frame->frameType;
}
