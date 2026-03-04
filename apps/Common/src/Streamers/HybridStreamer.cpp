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

    // setup prev camera
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    remoteCameraPrev.setFovyDegrees(remoteCamera.getFovyDegrees());
    remoteCameraPrev.setNear(remoteCamera.getNear());
    remoteCameraPrev.setFar(remoteCamera.getFar());

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

    // open the stats CSV file
    statsCSVFileName = "hybrid_streamer_time.csv";
    statsCSVFile.open(statsCSVFileName);

    // write the header to the CSV file 
    statsCSVFile << "frameID";
    statsCSVFile << ",visibles_render";
    statsCSVFile << ",visibles_compress";
    statsCSVFile << ",visibles_mesh";
    statsCSVFile << ",wide_fov_render";
    statsCSVFile << ",wide_fov_compress";
    statsCSVFile << ",wide_fov_mesh";
    statsCSVFile << ",dep_render";
    for (int layer = 0; layer < hiddenLayers; layer++) { 
        
        statsCSVFile << ",layer_" << layer << "_create";
        statsCSVFile << ",layer_" << layer << "_compress";
        statsCSVFile << ",layer_" << layer << "_create_mesh";
    }

    statsCSVFile << ",total_render";
    statsCSVFile << ",total_create";
    statsCSVFile << ",total_compress";
    statsCSVFile << std::endl;
    statsCSVFile.close();

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

    glFinish();

}

RenderStats HybridStreamer::generateFrame() {
    frameID++;
    
    timeStats = {
        .genFrameStatsByLayer = std::vector<genFrameStats>(hiddenLayers),
        .visibleMeshGenFrameStats = { 0 },
        .wideFovMeshGenFrameStats = { 0 },
        .totalRenderTimeMs = 0.0,
        .totalCreateTimeMs = 0.0,
        .totalCompressTimeMs = 0.0,
        .frameSize = 0.0,
    };

    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats;

    /*
    ============================
    Hidden Layer depth Peeling
    ============================
    */
    // Render all the objects in the scene
    renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);

    double depthPeelingRenderTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    timeStats.totalRenderTimeMs += depthPeelingRenderTimeMs;
    startTime = timeutils::getTimeMicros();

    for (int layer = 0; layer < hiddenLayers; layer++) {
        // Always use the remoteCamera
        auto& renderTargetToUse = frameRTsHidLayer[layer];
        auto& renderTargetToUse_noTone = frameRTsHidLayer_noTone[layer];
        auto& meshToUse = meshesHidLayer[layer];
        
        // blit the hidden layer from depth peeling renderer
        remoteRendererDP.peelingLayers[layer+1].blit(renderTargetToUse_noTone);
        // renderTargetToUse_noTone.writeColorAsPNG("hid_layer_no_tone_" + std::to_string(layer) + ".png");
        glFinish();
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
        
        
        timeStats.genFrameStatsByLayer[layer].renderTimeMs = depthPeelingRenderTimeMs;
        timeStats.genFrameStatsByLayer[layer].createTimeMs = frameGenerator.stats.createQuadsTimeMs;
        timeStats.genFrameStatsByLayer[layer].compressTimeMs = frameGenerator.stats.compressTimeMs;
        timeStats.genFrameStatsByLayer[layer].createMeshTimeMs = frameGenerator.stats.createMeshTimeMs;
        timeStats.totalCreateTimeMs += timeStats.genFrameStatsByLayer[layer].createTimeMs;
        timeStats.totalCompressTimeMs += timeStats.genFrameStatsByLayer[layer].compressTimeMs;

        tonemapper.setUniforms(renderTargetToUse_noTone);
        tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
    }
    
    /*
    ============================
    Generate visible mesh
    ============================
    */
    // Render all objects in scene
    startTime = timeutils::getTimeMicros();
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
    
    // We don't have createTime for meshwarp, so we just keep renderTime and compressTime 
    double visibleMeshRenderTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    timeStats.totalRenderTimeMs += visibleMeshRenderTimeMs;
    timeStats.visibleMeshGenFrameStats.renderTimeMs = visibleMeshRenderTimeMs;

    // Compress depth map to BC4 format with ZSTD
    size_t visibleMeshCompressedSize = depthStreamerRT.generateFrame();
    
    timeStats.visibleMeshGenFrameStats.compressTimeMs = depthStreamerRT.stats.compressTimeMs;
    timeStats.totalCompressTimeMs += timeStats.visibleMeshGenFrameStats.compressTimeMs;

    // Reconstruct visible mesh using meshwarp
    startTime = timeutils::getTimeMicros();
    reconstructMeshwarp(remoteCamera, visibleMesh, depthStreamerRT);
    double visibleMeshReconstructTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    timeStats.totalRenderTimeMs += visibleMeshReconstructTimeMs;
    timeStats.visibleMeshGenFrameStats.createTimeMs = visibleMeshReconstructTimeMs;

    spdlog::info("Reconstructing visible mesh using meshwarp done");

    /*
    ============================
    Wide FOV visible layer rendering
    ============================
    */

    // use prevRenderPose and currentRenderPose to generate an estimation of the wide fov camera pose

    glm::mat4 prevProjectionMatrix = remoteCameraPrev.getProjectionMatrix();
    glm::mat4 prevViewMatrix = remoteCameraPrev.getViewMatrix();

    glm::mat4 currentProjectionMatrix = remoteCamera.getProjectionMatrix();
    glm::mat4 currentViewMatrix = remoteCamera.getViewMatrix();
    // assume we have 4 corners of the previous frame (-1,1), (-1,1), (1,-1), (1,1)
    // we can use perspective projection to project these 4 corners to the current frame
    // based on the projected corners, we can estimate the uncovered area of the wide fov camera
    // we can the use this to do 1) Decide whether to draw the wide fov layer or not
    // 2) Decide the region where the wide fov layer should be drawn
    // Print out the previous and current camera poses

    spdlog::info("Current viewport size: ({}, {})", quadSet.getSize().x, quadSet.getSize().y);

    spdlog::info("  Previous Position: ({:.3f}, {:.3f}, {:.3f})", prevViewMatrix[3][0], prevViewMatrix[3][1], prevViewMatrix[3][2]);
    spdlog::info("  Current Position: ({:.3f}, {:.3f}, {:.3f})", currentViewMatrix[3][0], currentViewMatrix[3][1], currentViewMatrix[3][2]);
    
    // reproject the 4 corners to the world space
    // glm::vec3 corners[4] = {
    //     glm::vec3(-1.0, -1.0, 0),
    //     glm::vec3(-1.0, 1.0, 0),
    //     glm::vec3(1.0, -1.0, 0),
    //     glm::vec3(1.0, 1.0, 0),
    // };
    glm::vec3 corners[4] = {
        glm::vec3(0, 0, 1.0),
        glm::vec3(quadSet.getSize().x, 0, 1.0),
        glm::vec3(0, quadSet.getSize().y, 1.0),
        glm::vec3(quadSet.getSize().x, quadSet.getSize().y, 1.0),
    };
    glm::vec3 newCorners[4];
    for (int i = 0; i < 4; i++) {
        glm::vec3 worldCorner = glm::unProject(
            corners[i], 
            prevViewMatrix, 
            prevProjectionMatrix, 
            glm::vec4(0, 0, quadSet.getSize().x, quadSet.getSize().y));
        newCorners[i] = glm::project(
            worldCorner, 
            currentViewMatrix, 
            currentProjectionMatrix, 
            glm::vec4(0, 0, quadSet.getSize().x, quadSet.getSize().y));


        spdlog::info("  New corner {}: ({:.3f}, {:.3f}, {:.3f})", i, newCorners[i].x, newCorners[i].y, newCorners[i].z);
        newCorners[i].x = std::max(newCorners[i].x, 0.0f);
        newCorners[i].x = std::min(newCorners[i].x, float(quadSet.getSize().x));

        newCorners[i].y = std::max(newCorners[i].y, 0.0f);
        newCorners[i].y = std::min(newCorners[i].y, float(quadSet.getSize().y));
    }

    // check the total black areas by the new corners
    float totalBlackArea = 0.0f;
    // 
    // check the frameID
    for (int i = 0; i < 4; i++) {
        spdlog::info("     New corner {}: ({}, {})", i, newCorners[i].x, newCorners[i].y);
    }

    newCorners[1].x = (quadSet.getSize().x - newCorners[1].x);
    newCorners[2].y = (quadSet.getSize().y - newCorners[2].y);
    newCorners[3].x = (quadSet.getSize().x - newCorners[3].x);
    newCorners[3].y = (quadSet.getSize().y - newCorners[3].y);

    for (int i = 0; i < 4; i++) {
        totalBlackArea += newCorners[i].x * newCorners[i].y;
    }

    totalBlackArea += (newCorners[0].y + newCorners[1].y) * (quadSet.getSize().x - newCorners[0].x - newCorners[1].x) / 2;
    totalBlackArea += (newCorners[0].x + newCorners[2].x) * (quadSet.getSize().y - newCorners[0].y - newCorners[2].y) / 2;
    totalBlackArea += (newCorners[2].y + newCorners[3].y) * (quadSet.getSize().x - newCorners[2].x - newCorners[3].x) / 2;
    totalBlackArea += (newCorners[1].x + newCorners[3].x) * (quadSet.getSize().y - newCorners[1].y - newCorners[3].y) / 2;

    spdlog::info("Frame ID: {}", frameID);
    spdlog::info("Total black area: {}", totalBlackArea);
    blackComputedCSVFile.open(blackComputedCSVFileName, std::ios::app);
    blackComputedCSVFile << frameID << "," << totalBlackArea << std::endl;
    blackComputedCSVFile.close();


    // work reversely reproject the new pose into the old pose's wide fov space
    glm::mat4 projectionMatrixWideFOV = remoteCameraWideFOV.getProjectionMatrix();

    spdlog::info(" prevProjectionMatrix: ");
    for (int i = 0; i < 4; i++) {
        spdlog::info("({}, {}, {}, {})", prevProjectionMatrix[i][0], prevProjectionMatrix[i][1], prevProjectionMatrix[i][2], prevProjectionMatrix[i][3]);
        
    }

    for (int i = 0; i < 4; i++) {
        glm::vec3 worldCorner = glm::unProject(
            corners[i], 
            currentViewMatrix, 
            currentProjectionMatrix, 
            glm::vec4(0, 0, quadSet.getSize().x, quadSet.getSize().y));

        glm::vec3 cornerInSourceWideFoVImage = glm::project(
            worldCorner, 
            prevViewMatrix, 
            projectionMatrixWideFOV, 
            glm::vec4(0, 0, quadSet.getSize().x, quadSet.getSize().y));

        // we compute the area of the corner in the wide fov image
        spdlog::info("  Corner in source wide fov image: ({}, {}, {})", cornerInSourceWideFoVImage.x, cornerInSourceWideFoVImage.y, cornerInSourceWideFoVImage.z);
    }

    // we get the new corners in the wide fov space
    



    // if (totalBlackArea > 10000.0f) {
    if (true) {
        // We can skip the wide fov layer rendering and reconstruction
        // 

    startTime = timeutils::getTimeMicros();

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
    
    // Render finished, log the time
    double wideFovMeshRenderTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    timeStats.totalRenderTimeMs += wideFovMeshRenderTimeMs;
    timeStats.wideFovMeshGenFrameStats.renderTimeMs = wideFovMeshRenderTimeMs;
    
    size_t wideFovMeshCompressedSize = depthStreamerWideFOV.generateFrame();

    timeStats.wideFovMeshGenFrameStats.compressTimeMs = depthStreamerWideFOV.stats.compressTimeMs;
    timeStats.totalCompressTimeMs += timeStats.wideFovMeshGenFrameStats.compressTimeMs;

    // // Reconstruct wide fov visible mesh using meshwarp
    startTime = timeutils::getTimeMicros();
    reconstructMeshwarp(remoteCameraWideFOV, visibleMeshWideFOV, depthStreamerWideFOV);
    double wideFovMeshReconstructTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    timeStats.totalRenderTimeMs += wideFovMeshReconstructTimeMs;
    timeStats.wideFovMeshGenFrameStats.createTimeMs = wideFovMeshReconstructTimeMs;

    // depthStreamerWideFOV.writeColorAsPNG("debug_depth_widefov.png");
    }

    // Update the previous camera pose
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    /*
    ============================
    Generate atlas frames
    ============================
    */

    subFrameIndex atlasIndex = {0, 0};
    uint subFrameWidth = depthStreamerRT.width;
    uint subFrameHeight = depthStreamerRT.height;

    spdlog::info("Blitting the default layer to the atlas");
    
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

    // write out the time stats to CSV file
    statsCSVFile.open(statsCSVFileName, std::ios::app);
    statsCSVFile << frameID << ",";
    statsCSVFile << timeStats.visibleMeshGenFrameStats.renderTimeMs << ",";
    statsCSVFile << timeStats.visibleMeshGenFrameStats.compressTimeMs << ",";
    statsCSVFile << timeStats.visibleMeshGenFrameStats.createTimeMs << ",";
    statsCSVFile << timeStats.wideFovMeshGenFrameStats.renderTimeMs << ",";
    statsCSVFile << timeStats.wideFovMeshGenFrameStats.compressTimeMs << ",";

    statsCSVFile << timeStats.wideFovMeshGenFrameStats.createTimeMs << ",";

    statsCSVFile << timeStats.genFrameStatsByLayer[0].renderTimeMs << ",";

    for (int layer = 0; layer < hiddenLayers; layer++) {
        
        statsCSVFile << timeStats.genFrameStatsByLayer[layer].createTimeMs << ",";
        statsCSVFile << timeStats.genFrameStatsByLayer[layer].compressTimeMs << ",";
        statsCSVFile << timeStats.genFrameStatsByLayer[layer].createMeshTimeMs << ",";
    }
    statsCSVFile << timeStats.totalRenderTimeMs << ",";
    statsCSVFile << timeStats.totalCreateTimeMs << ",";
    statsCSVFile << timeStats.totalCompressTimeMs << std::endl;
    statsCSVFile.close();

    return renderStats;
}

void HybridStreamer::sendFrame(PoseReceiver::PoseInfo poseInfo) {

    // write alpha atlas and compressed depth offset to memory
    pose_id_t poseID = poseInfo.pose_id;
    timeStats.frameSize = writeToMemory(poseInfo, compressedData);

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

size_t HybridStreamer::writeToMemory(PoseReceiver::PoseInfo poseInfo, std::vector<char>& outputData) {

    // Save camera data
    spdlog::info("Writing camera data to memory");
    Pose cameraPose;
    pose_id_t poseID = poseInfo.pose_id;
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

    double timestamp = double(timeutils::getTimeMicros());
    spdlog::info("Timestamp: {}", timestamp);

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
        .pose_send_timestamp = poseInfo.send_timestamp,
        .pose_recv_timestamp = poseInfo.recv_timestamp,
        .frame_send_timestamp = timestamp,
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

    spdlog::info("Total data size: {}", outputData.size());

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
