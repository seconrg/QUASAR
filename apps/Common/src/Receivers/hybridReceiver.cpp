#include <Receivers/HybridReceiver.h>
#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

HybridReceiver::HybridReceiver(
        const glm::uvec2& remoteGBufferSize, 
        uint depthFactor,
        uint vertexGroupSize,
        QuadSet& quadSet, uint hiddenLayers, 
        const std::string& videoAtlasURL, 
        const std::string& proxiesURL,
        const std::string& videoVisibleURL,
        const std::string& depthVisibleURL,
        const std::string& videoVisibleWideFovURL,
        const std::string& depthVisibleWideFovURL)
    : vertexGroupSize(vertexGroupSize)
    , depthFactor(depthFactor)
    , adjustedSize(remoteGBufferSize / vertexGroupSize)
    , hiddenLayers(hiddenLayers)
    , videoVisibleURL(videoVisibleURL)
    , depthVisibleURL(depthVisibleURL)
    , videoVisibleWideFovURL(videoVisibleWideFovURL)
    , depthVisibleWideFovURL(depthVisibleWideFovURL)
    , visibleTexture({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoVisibleURL)
    , depthTexture({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthVisibleURL)
    , visibleTextureWideFOV({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoVisibleWideFovURL)
    , depthTextureWideFOV({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthVisibleWideFovURL)
    , meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_BC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_BC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , meshWarpReconstructShader({
        .computeCodeData = SHADER_COMMON_MESHWARP_RECONSTRUCT_COMP,
        .computeCodeSize = SHADER_COMMON_MESHWARP_RECONSTRUCT_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , visibleMeshMaterial({ .baseColorTexture = &visibleTexture })
    , visibleMesh({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &visibleMeshMaterial,
        .usage = GL_DYNAMIC_DRAW
    })
    , visibleMeshWideFOVMaterial({ .baseColorTexture = &visibleTextureWideFOV})
    , visibleMeshWideFOV({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &visibleMeshWideFOVMaterial,
        .usage = GL_DYNAMIC_DRAW
    })
    ,QUASARReceiver(quadSet, hiddenLayers, videoAtlasURL, proxiesURL)
{
    // Initialize visible layers for meshwarp generation
    meshFromBC4Shader.bind();
    meshFromBC4Shader.setBool("unlinearizeDepth", true);
    meshFromBC4Shader.setVec2("depthMapSize", glm::vec2(depthTexture.width, depthTexture.height));
    meshFromBC4Shader.setUint("vertexGroupSize", vertexGroupSize);

    meshWarpReconstructShader.bind();
    meshWarpReconstructShader.setBool("unlinearizeDepth", true);
    meshWarpReconstructShader.setVec2("depthMapSize", glm::vec2(depthTexture.width, depthTexture.height));
    meshWarpReconstructShader.setUint("vertexGroupSize", vertexGroupSize);

    // Initialize CSV file for stats
    statsCSVFileName = "hybrid_receiver_stats.csv";
    statsCSVFile.open(statsCSVFileName);
    statsCSVFile << "frame_id,memory_transfer_time_ms,visible,wide_fov";
    for (int layer = 0; layer < hiddenLayers; layer++) {
        statsCSVFile << ",layer_" << layer << "_decompress";
        statsCSVFile << ",layer_" << layer << "_depth_peeling";
    }
    statsCSVFile << ",total_time_ms" << std::endl;
    statsCSVFile.close();

}

void HybridReceiver::updateMesh(bool isWideFOV) {
    // Set shader uniforms

    PerspectiveCamera& cameraInUse = isWideFOV ? remoteCameraWideFOV : remoteCamera;
    Mesh& meshInUse = isWideFOV ? visibleMeshWideFOV : visibleMesh;
    BC4DepthVideoTexture& depthTextureInUse = isWideFOV ? depthTextureWideFOV : depthTexture;
    if (isWideFOV) {
        visibleTextureWideFOV.bind();
        poseIdColor = visibleTextureWideFOV.draw();
        depthTextureWideFOV.bind();
        poseIdDepth = depthTextureWideFOV.draw();

    } else {
        visibleTexture.bind();
        poseIdColor = visibleTexture.draw();
        depthTexture.bind();
        poseIdDepth = depthTexture.draw();
    }

    meshFromBC4Shader.bind();
    {
        meshFromBC4Shader.setMat4("projection", cameraInUse.getProjectionMatrix());
        meshFromBC4Shader.setMat4("projectionInverse", cameraInUse.getProjectionMatrixInverse());
        meshFromBC4Shader.setMat4("viewColor", colorFramePose.mono.view);
        meshFromBC4Shader.setMat4("viewInverseDepth", glm::inverse(depthFramePose.mono.view));
        meshFromBC4Shader.setFloat("near", cameraInUse.getNear());
        meshFromBC4Shader.setFloat("far", cameraInUse.getFar());
    }
    {
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshInUse.vertexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, meshInUse.indexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthTextureInUse.bc4CompressedBuffer);
    }

    // Generate vertices and indices for mesh from depth map
    meshFromBC4Shader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                               ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    
    
    meshWarpReconstructShader.bind();
    {
        meshWarpReconstructShader.setMat4("projection", cameraInUse.getProjectionMatrix());
        meshWarpReconstructShader.setMat4("view", cameraInUse.getViewMatrix());
        meshWarpReconstructShader.setFloat("near", cameraInUse.getNear());
        meshWarpReconstructShader.setFloat("far", cameraInUse.getFar());
    }
    {
        meshWarpReconstructShader.setFloat("depthThreshold", 0.05f);
    }
    {
        meshWarpReconstructShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshInUse.vertexBuffer);
        meshWarpReconstructShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, meshInUse.indexBuffer);
    }

    meshWarpReconstructShader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                        ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshWarpReconstructShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    glFinish();
}


void HybridReceiver::recvData(
        const PoseStreamer& poseStreamer,
        double& elapsedTimeColor, 
        double& elapsedTimeDepth) {


    struct TimeStats timeStats;
    timeStats.decompressTimeMsByLayer.resize(hiddenLayers);
    timeStats.depthPeelingTimeMsByLayer.resize(hiddenLayers);
    timeStats.meshwarpReconstructVisibleTimeMs = 0.0;
    timeStats.meshwarpReconstructWideFovTimeMs = 0.0;
    timeStats.totalTimeMs = 0.0;


    double startTime = timeutils::getTimeMicros();

    // Get poses for the frames
    poseStreamer.getPose(poseIdColor, &colorFramePose, &elapsedTimeColor);
    poseStreamer.getPose(poseIdDepth, &depthFramePose, &elapsedTimeDepth);

    // Update both visible and wide FOV meshes
    updateMesh(true);
    
    timeStats.meshwarpReconstructWideFovTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    startTime = timeutils::getTimeMicros();
    
    updateMesh(false);
    
    timeStats.meshwarpReconstructVisibleTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    // Wait for a frame that has been written to
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            spdlog::info("No frame pending, waiting for frame");
            return;
        }

        if (videoAtlasTexture.getLatestPoseID() < framePending->poseID) { // Video is behind, wait until video catches up
            spdlog::info("Video is behind, waiting for video to catch up");
            return;
        }

        frame = framePending;
        framePending.reset();
        frameInUse = frame;
    }

    // Update color texture
    videoAtlasTexture.bind();
    videoAtlasTexture.draw(frame->poseID);

    // Update alpha texture
    alphaAtlasTexture.bind();
    alphaAtlasTexture.loadFromData(frame->bufferPool.alphaData.data());

    glFinish();

    // Reconstruct meshes from frame
    TimeStats timeStatsDp = reconstructHiddenLayers(frame);

    // Update time stats
    timeStats.totalTimeMs += timeStats.meshwarpReconstructVisibleTimeMs;
    timeStats.totalTimeMs += timeStats.meshwarpReconstructWideFovTimeMs;
    timeStats.decompressTimeMsByLayer = timeStatsDp.decompressTimeMsByLayer;
    timeStats.depthPeelingTimeMsByLayer = timeStatsDp.depthPeelingTimeMsByLayer;
    timeStats.totalTimeMs += timeStatsDp.totalTimeMs;

    // Write stats to CSV file
    statsCSVFile.open(statsCSVFileName, std::ios::app);
    statsCSVFile << frameID << ",";
    statsCSVFile << timeStats.memoryTransferTimeMs << ",";
    statsCSVFile << timeStats.meshwarpReconstructVisibleTimeMs << ",";
    statsCSVFile << timeStats.meshwarpReconstructWideFovTimeMs << ",";
    for (int layer = 0; layer < hiddenLayers; layer++) {
        statsCSVFile << timeStats.decompressTimeMsByLayer[layer] << ",";
        statsCSVFile << timeStats.depthPeelingTimeMsByLayer[layer] << ",";
    }
    statsCSVFile << timeStats.totalTimeMs << std::endl;
    statsCSVFile.close();

    // Reset frame
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return;
}


HybridReceiver::TimeStats HybridReceiver::reconstructHiddenLayers(std::shared_ptr<Frame> frame) {
    
    TimeStats timeStats;
    timeStats.decompressTimeMsByLayer.resize(hiddenLayers);
    timeStats.depthPeelingTimeMsByLayer.resize(hiddenLayers);
    timeStats.totalTimeMs = 0.0;

    frame->cameraPose.copyPoseToCamera(remoteCamera);

    spdlog::info("    Loading camera pose: {}, {}, {}", remoteCamera.getPosition().x, remoteCamera.getPosition().y, remoteCamera.getPosition().z);
    spdlog::info("    Loading camera rotation: {}, {}, {}", remoteCamera.getRotationEuler().x, remoteCamera.getRotationEuler().y, remoteCamera.getRotationEuler().z);
    spdlog::info("    Loading camera fovy: {}", remoteCamera.getFovyDegrees());
    const glm::vec2& gBufferSize = quadSet.getSize();

    for (int layer = 0; layer < hiddenLayers; layer++) {
        double startTime = timeutils::getTimeMicros();
        auto sizes = quadSet.loadFromMemory(bufferPool.uncompressedQuads[layer], bufferPool.uncompressedOffsets[layer]);
        referenceFrames[layer].numQuads = sizes.numQuads;
        referenceFrames[layer].numDepthOffsets = sizes.numDepthOffsets;
        stats.transferTimeMs += quadSet.stats.transferTimeMs;

        timeStats.decompressTimeMsByLayer[layer] = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        startTime = timeutils::getTimeMicros();

        meshes[layer].appendQuads(quadSet, gBufferSize);

        spdlog::info("    Appending quads for layer {} took {} ms", layer, timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));
        double tmpStartTime = timeutils::getTimeMicros();
        meshes[layer].createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        spdlog::info("    Creating mesh from proxies for layer {} took {} ms", layer, timeutils::microsToMillis(timeutils::getTimeMicros() - tmpStartTime));

        auto meshBufferSizes = meshes[layer].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;

        timeStats.depthPeelingTimeMsByLayer[layer] = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        
        timeStats.totalTimeMs += timeStats.decompressTimeMsByLayer[layer];
        timeStats.totalTimeMs += timeStats.depthPeelingTimeMsByLayer[layer];
    }

    return timeStats;
}


QuadFrame::FrameType HybridReceiver::loadFromMemory(const std::vector<char>& inputData) {
    

    spdlog::info("Loading inputData of size {}", inputData.size());
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

    spdlog::info("    Loading camera size: {}", header.cameraSize);
    spdlog::info("    Loading alpha size: {}", header.alphaSize);
    spdlog::info("    Loading geometry size: {}", header.geometrySize);
    
    // Read camera data
    frame->cameraPose.loadFromMemory(ptr, header.cameraSize);
    ptr += header.cameraSize;

    // Read alpha data
    alphaCodec.decompress(ptr, frame->bufferPool.alphaData, header.alphaSize);

    // dump alpha data to file
    ptr += header.alphaSize;

    const char* layerPtr = ptr;
    uint32_t layerSize;

    std::vector<std::future<size_t>> futures;

    double startTime = timeutils::getTimeMicros();
    int layersSize = 0;
    for (int layer = 0; layer < hiddenLayers; layer++) {
        std::memcpy(&layerSize, layerPtr, sizeof(uint32_t));
        const char* dataPtr = layerPtr + sizeof(uint32_t);

        // print out to screen the first 100 bytes of the data
        // spdlog::info("First 100 bytes of data for layer {} with size {}: {}", layer, layerSize, std::string(dataPtr, std::min((int)layerSize, 100)));

        futures.emplace_back(threadPool->submit_task([&, layer, dataPtr, layerSize]() {
            return referenceFrames[layer].loadFromMemory(dataPtr, layerSize);
        }));

        layerPtr += sizeof(uint32_t) + layerSize;

        if (layer < hiddenLayers - 1) {
            layersSize += layerSize;
        }
    }

    for (auto& f : futures) f.get();

    frame->decompressReferenceHiddenLayersWideFOV(threadPool, referenceFrames);

    spdlog::info("    Total hidden layers size: {}", layersSize);


    stats.loadTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;

}