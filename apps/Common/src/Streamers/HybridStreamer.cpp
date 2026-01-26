#include <Streamers/HybridStreamer.h>
// #include <shaders_common.h>
#include <nvtx3/nvToolsExt.h>
#include <Utils/FileIO.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

HybridStreamer::HybridStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer, 
        Scene& remoteScene,
        Scene& localScene,
        PerspectiveCamera& remoteCamera,
        const HybridStreamerCreateParams& params)
    : videoAtlasURL(params.videoAtlasURL)
    , proxiesURL(params.proxiesURL)
    , videoURL(params.videoURL)
    , depthURL(params.depthURL)
    , videoWideFovURL(params.videoWideFovURL)
    , depthWideFovURL(params.depthWideFovURL)
    , hiddenLayers(params.hiddenLayers)
    , remoteRendererDP(remoteRendererDP)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , localScene(localScene)
    , remoteCamera(remoteCamera)
    , adjustedSize(glm::uvec2(remoteRenderer.width, remoteRenderer.height) / params.vertexGroupSize)
    , depthMapSize(glm::uvec2(remoteRenderer.width, remoteRenderer.height) / params.depthFactor)
    , quadSet(quadSet)
    , frameGenerator(quadSet)
    , depthStreamerRT({
        .width = remoteRenderer.width / params.depthFactor,
        .height = remoteRenderer.height / params.depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.depthURL, params.maxFrameRate)
    , depthStreamerWideFOV({
        .width = remoteRenderer.width / params.depthFactor,
        .height = remoteRenderer.height / params.depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.depthWideFovURL, params.maxFrameRate)
    , depthEffect(remoteCamera)
    , videoAtlasStreamerRT({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.videoAtlasURL, params.maxFrameRate, params.targetBitRate)
    , alphaAtlasRT({
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
    , frameRTVisible({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , frameRTVisibleWideFov({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , visibleVideoStreamerRT({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, params.videoURL, params.maxFrameRate, params.targetBitRate)
    , visibleVideoStreamerWideFOV({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, params.videoWideFovURL, params.maxFrameRate, params.targetBitRate)
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
    , visibleMeshMaterial({ .baseColorTexture = &frameRTVisible.colorTexture })
    , visibleMeshWideFOVMaterial({ .baseColorTexture = &frameRTVisibleWideFov.colorTexture })
    , visibleMesh({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &visibleMeshMaterial,
        .usage = GL_DYNAMIC_DRAW,
    })
    , visibleMeshWideFOV({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &visibleMeshWideFOVMaterial,
        .usage = GL_DYNAMIC_DRAW,
    })
    , alphaCodec(alphaAtlasRT.width, alphaAtlasRT.height)
    , DataStreamerTCP(params.proxiesURL)
{
    // Initialize hidden layer resources
    referenceFrames.resize(hiddenLayers);
    frameRTsHidLayer.reserve(hiddenLayers);
    frameRTsHidLayer_noTone.reserve(hiddenLayers);

    meshesHidLayer.reserve(hiddenLayers);
    nodesHidLayer.reserve(hiddenLayers);
    wireframesHidLayer.reserve(hiddenLayers);
    
    
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
    for (int layer = 0; layer < hiddenLayers; layer++) {
        frameRTsHidLayer.emplace_back(rtParams);
        frameRTsHidLayer_noTone.emplace_back(rtParams);
    }

    for (int layer = 0; layer < hiddenLayers; layer++) {
        meshesHidLayer.emplace_back(
            quadSet, 
            frameRTsHidLayer_noTone[layer].colorTexture, 
            frameRTsHidLayer_noTone[layer].alphaTexture
        );

        nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
        nodesHidLayer[layer].frustumCulled = false;

        const glm::vec4& color = colors[(layer + 1) % colors.size()];

        wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
        wireframesHidLayer[layer].frustumCulled = false;
        wireframesHidLayer[layer].wireframe = true;
        wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

    }

    setViewSphereDiameter(params.viewSphereDiameter);

    // Initialize for Meshwarp-related shader
    meshFromBC4Shader.bind();
    meshFromBC4Shader.setBool("unlinearizeDepth", true);
    meshFromBC4Shader.setVec2("depthMapSize", depthMapSize);
    meshFromBC4Shader.setUint("vertexGroupSize", params.vertexGroupSize);

    /*
    ============================
    Information used for local meshwarp simulated rendering and debugging
    ============================
    */
    
    meshWarpReconstructShader.bind();
    meshWarpReconstructShader.setBool("unlinearizeDepth", true);
    meshWarpReconstructShader.setVec2("depthMapSize", depthMapSize);
    meshWarpReconstructShader.setUint("vertexGroupSize", params.vertexGroupSize);

    // setup wide fov camera
    remoteCameraWideFOV.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

    visibleMeshNode = Node(&visibleMesh);
    visibleMeshNode.frustumCulled = false;
    // visibleMeshNode.wireframe = true;
    // visibleMeshNode.overrideMaterial = new QuadMaterial({ .baseColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) });

    visibleMeshWideFOVNode = Node(&visibleMeshWideFOV);
    visibleMeshWideFOVNode.frustumCulled = false;

    // visibleMeshWideFOVNode.wireframe = true;
    // visibleMeshWideFOVNode.overrideMaterial = new QuadMaterial({ .baseColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) });

    // Add meshes to wide fov scene
    wideFovNode = Node(&visibleMesh);
    wideFovNode.frustumCulled = false;
    for (int layer = hiddenLayers - 1; layer >=0 ; layer--) {
        wideFovScene.addChildNode(&nodesHidLayer[layer]);
    }
    wideFovScene.addChildNode(&visibleMeshNode);
    
    proxyMetadatas.resize(hiddenLayers);
    uncompressedAlphaData.resize(alphaAtlasRT.width * alphaAtlasRT.height);

    spdlog::info("HybridStreamer initialized");

}

void HybridStreamer::addMeshesToScene(Scene& localScene) {
    
    localScene.addChildNode(&visibleMeshWideFOVNode);
    // add all hidden layers, from farthest to nearest
    for (int layer = hiddenLayers - 1; layer >=0 ; layer--) {
        localScene.addChildNode(&nodesHidLayer[layer]);
        // localScene.addChildNode(&wireframesHidLayer[layer]);
    }

    localScene.addChildNode(&visibleMeshNode);
    // wideFovScene = localScene; // copy for wide fov rendering
    
}

void HybridStreamer::setViewSphereDiameter(float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;
    remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);
}

typedef struct subFrameIndex {
    uint row;
    uint col;
} subFrameIndex;

// Find the next position in the atlas given the previous index
inline subFrameIndex getNextSubFrameIndex(
    uint currentRow, 
    uint currentCol, 
    uint subFrameWidth, 
    uint subFrameHeight, 
    uint atlasWidth, 
    uint atlasHeight) 
{
    subFrameIndex nextIndex;
    
    nextIndex.col = currentCol + subFrameWidth;
    nextIndex.row = currentRow;

    // jump to the next row if we exceed the width
    if (nextIndex.col >= atlasWidth) {
        nextIndex.col = 0;
        nextIndex.row += subFrameHeight;
        if (nextIndex.row >= atlasHeight) {
            nextIndex.row = 0; // Wrap around to the beginning
            nextIndex.col = 0;
        }
    }
    return nextIndex;
}

void HybridStreamer::reconstructMeshwarp(PerspectiveCamera &camera, Mesh &mesh, BC4DepthStreamer &depthStreamer)
{
    RenderStats renderStats;

    nvtxRangePushA("Vertex generation");
    
    meshFromBC4Shader.bind();
    {
        meshFromBC4Shader.setMat4("projection", camera.getProjectionMatrix());
        meshFromBC4Shader.setMat4("projectionInverse", camera.getProjectionMatrixInverse());
        meshFromBC4Shader.setMat4("viewColor", camera.getViewMatrix());
        meshFromBC4Shader.setMat4("viewInverseDepth", camera.getViewMatrixInverse());
        meshFromBC4Shader.setFloat("near", camera.getNear());
        meshFromBC4Shader.setFloat("far", camera.getFar());
    }
    {
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthStreamer.bc4CompressedBuffer);
    }
    // Dispatch compute shader to generate vertices and indices for mesh
    meshFromBC4Shader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                               ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    nvtxRangePop();

    nvtxRangePushA("Frame Generation");
    meshWarpReconstructShader.bind();
    {
        meshWarpReconstructShader.setMat4("projection", camera.getProjectionMatrix());
        meshWarpReconstructShader.setMat4("view", camera.getViewMatrix());
        meshWarpReconstructShader.setFloat("near", camera.getNear());
        meshWarpReconstructShader.setFloat("far", camera.getFar());
    }
    {
        meshWarpReconstructShader.setFloat("depthThreshold", 0.05f);
    }
    {
        meshWarpReconstructShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
        meshWarpReconstructShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
    }

    meshWarpReconstructShader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshWarpReconstructShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    nvtxRangePop();

}

RenderStats HybridStreamer::generateFrame() {
    // Reset stats
    stats = { 0 };
    RenderStats renderStats;

    /*
    ============================
    Hidden Layer depth Peeling
    ============================
    */
    // Render all the objects in the scene
    spdlog::info("Rendering all the objects in the scene");
    renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);

    for (int layer = 0; layer < hiddenLayers; layer++) {
        // Always use the remoteCamera
        auto& renderTargetToUse = frameRTsHidLayer[layer];
        auto& renderTargetToUse_noTone = frameRTsHidLayer_noTone[layer];
        auto& meshToUse = meshesHidLayer[layer];
        
        // blit the hidden layer from depth peeling renderer
        remoteRendererDP.peelingLayers[layer+1].blit(renderTargetToUse_noTone);
        // renderTargetToUse_noTone.writeColorAsPNG("hid_layer_no_tone_" + std::to_string(layer) + ".png");

        /*
        ============================
        Generate hidden layer reference frames
        ============================
        */
        frameGenerator.createReferenceFrame(
            renderTargetToUse_noTone, 
            remoteCamera, 
            meshToUse, 
            referenceFrames[layer]);
        
        tonemapper.setUniforms(renderTargetToUse_noTone);
        tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
        
    }

    // // Render all objects in scene
    double startTime = timeutils::getTimeMicros();
    renderStats = remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCamera);

    // Copy to intermediate render target
    tonemapper.enableTonemapping(false);
    tonemapper.drawToRenderTarget(remoteRenderer, frameRTVisible);
    // // writeback the frameRTVisible to file
    // frameRTVisible.writeColorAsPNG("frameRTVisible.png");

    // Copy color and depth to video frames
    tonemapper.enableTonemapping(true);
    tonemapper.drawToRenderTarget(remoteRenderer, visibleVideoStreamerRT);
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerRT);
    stats.totalRenderTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Compress depth map to BC4 format with ZSTD
    stats.compressedSize = depthStreamerRT.generateFrame();
    stats.totalCompressTimeMs = depthStreamerRT.stats.compressTimeMs;

    // Reconstruct visible mesh using meshwarp
    reconstructMeshwarp(remoteCamera, visibleMesh, depthStreamerRT);

    spdlog::info("Reconstructing visible mesh using meshwarp done");

    /*
    ============================
    Wide FOV visible layer rendering
    ============================
    */

    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);

    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
    // From the previous mesh, see what parts are visible in wide fov
    renderStats += remoteRenderer.drawObjectsNoLighting(wideFovScene, remoteCameraWideFOV);
    tonemapper.enableTonemapping(false);
    tonemapper.drawToRenderTarget(remoteRenderer, frameRTVisibleWideFov);
    tonemapper.enableTonemapping(true);
    
    // use the previous generated stencil buffer to avoid drawing where wide fov has drawn
    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
    
    // Draw the whole scene and composite with wide fov
    renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraWideFOV, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    remoteRenderer.pipeline.stencilState.restoreStencilState();
    // render with tonemapper for video streaming
    tonemapper.enableTonemapping(false);
    tonemapper.drawToRenderTarget(remoteRenderer, frameRTVisibleWideFov);
    tonemapper.enableTonemapping(true);

    // render into depthStreamerWideFOV
    tonemapper.drawToRenderTarget(remoteRenderer, visibleVideoStreamerWideFOV);
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerWideFOV);
    depthStreamerWideFOV.generateFrame();

    spdlog::info("Generating wide fov depth map done");

    // Reconstruct wide fov visible mesh using meshwarp
    reconstructMeshwarp(remoteCameraWideFOV, visibleMeshWideFOV, depthStreamerWideFOV);

    // depthStreamerWideFOV.writeColorAsPNG("debug_depth_widefov.png");

    subFrameIndex atlasIndex = {0, 0};
    uint subFrameWidth = depthStreamerRT.width;
    uint subFrameHeight = depthStreamerRT.height;

    spdlog::info("Blitting the default layer to the atlas");
    
    // blit the default layer, directly from depth peeling renderer
    frameRTVisible.blit(
        videoAtlasStreamerRT, 0, 0, 
        subFrameWidth, 
        subFrameHeight, 
        atlasIndex.col, 
        atlasIndex.row, 
        atlasIndex.col + subFrameWidth, 
        atlasIndex.row + subFrameHeight
    );

    frameRTVisible.blit(
        alphaAtlasRT, 0, 0, 
        subFrameWidth, 
        subFrameHeight, 
        atlasIndex.col, 
        atlasIndex.row, 
        atlasIndex.col + subFrameWidth, 
        atlasIndex.row + subFrameHeight
    );

    atlasIndex = getNextSubFrameIndex(
        atlasIndex.row, 
        atlasIndex.col, 
        subFrameWidth, 
        subFrameHeight, 
        videoAtlasStreamerRT.width, 
        videoAtlasStreamerRT.height);
    
    // blit the hidden layers
    for (int i=0; i< hiddenLayers; i++) {
        frameRTsHidLayer[i].blit(
            videoAtlasStreamerRT, 0, 0, 
            frameRTsHidLayer[i].width, 
            frameRTsHidLayer[i].height, 
            atlasIndex.col, 
            atlasIndex.row, 
            atlasIndex.col + subFrameWidth, 
            atlasIndex.row + subFrameHeight
        );

        frameRTsHidLayer_noTone[i].blit(
            alphaAtlasRT, 0, 0, 
            frameRTsHidLayer_noTone[i].width, 
            frameRTsHidLayer_noTone[i].height, 
            atlasIndex.col, 
            atlasIndex.row, 
            atlasIndex.col + subFrameWidth, 
            atlasIndex.row + subFrameHeight
        );

        atlasIndex = getNextSubFrameIndex(
            atlasIndex.row, 
            atlasIndex.col, 
            subFrameWidth, 
            subFrameHeight, 
            videoAtlasStreamerRT.width, 
            videoAtlasStreamerRT.height);
    }
    // DEBUGGING WRITING out atlas frames
    // videoAtlasStreamerRT.writeColorAsPNG("video_atlas.png");
    // alphaAtlasRT.writeAlphaAsPNG("alpha_atlas.png");
    spdlog::info("HybridStreamer generated frame");
    return renderStats;
}

void HybridStreamer::sendFrame(pose_id_t poseID) {

    // write alpha atlas and compressed depth offset to memory
    stats.frameSize = writeToMemory(poseID, compressedData);

    visibleVideoStreamerRT.sendFrame(poseID);
    visibleVideoStreamerWideFOV.sendFrame(poseID);

    depthStreamerRT.sendFrame(poseID);
    depthStreamerWideFOV.sendFrame(poseID);

    // send atlas hidden frame
    if (!videoURL.empty() && !proxiesURL.empty()) {
        videoAtlasStreamerRT.sendFrame(poseID);
        send(compressedData);
    }
}

size_t HybridStreamer::writeToMemory(pose_id_t poseID, std::vector<char>& outputData) {

    // Save camera data
    spdlog::info("Writing camera data to memory");
    Pose cameraPose;
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save alpha data
    alphaAtlasRT.writeAlphaToMemory(uncompressedAlphaData);
    alphaCodec.compress(uncompressedAlphaData.data(), alphaData, uncompressedAlphaData.size());

    // We only need to save hidden layers here, because visible layer
    // is streamed through another streamer

    for (int layer = 0; layer < hiddenLayers; layer++) {
        referenceFrames[layer].writeToMemory(proxyMetadatas[layer]);
    }
    uint32_t proxySize = 0;
    for (const auto& proxyMetadata : proxyMetadatas) {
        proxySize += sizeof(uint32_t) + static_cast<uint32_t>(proxyMetadata.size());
    }

    QUASARReceiver::Header header{
        .poseID = poseID,
        .frameType = QuadFrame::FrameType::REFERENCE,
        .params {
            .numLayers = static_cast<uint32_t>(proxyMetadatas.size()),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
        },
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .alphaSize = static_cast<uint32_t>(alphaData.size()),
        .geometrySize = proxySize,
    };

    outputData.resize(header.getSize());
    char* ptr = outputData.data();

    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();

    // Write alpha data
    std::memcpy(ptr, alphaData.data(), alphaData.size());
    ptr += alphaData.size();

    // Write geometry data
    for (const auto& layerData : proxyMetadatas) {
        uint32_t layerSize = static_cast<uint32_t>(layerData.size());

        // Write size of layer
        std::memcpy(ptr, &layerSize, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Write layer data
        std::memcpy(ptr, layerData.data(), layerSize);
        ptr += layerSize;
    }

    spdlog::info("Total data size: {:.3f}MB", static_cast<float>(outputData.size()) / BYTES_PER_MEGABYTE);

    return outputData.size();
}

size_t HybridStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = outputPath / "camera.bin";
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save depth
    Path depthFileName = outputPath / "depth.bc4.zstd";
    size_t totalBytes = depthStreamerRT.writeToFile(depthFileName);

    return totalBytes;
}


uint HybridStreamer::getNumTriangles() const {
    uint numTriangles = 0; // Each triangle has 3 indices
    for (const auto& mesh : meshesHidLayer) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}
