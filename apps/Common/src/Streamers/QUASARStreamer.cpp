#include "RenderTargets/FrameRenderTarget.h"
#include <Streamers/QUASARStreamer.h>

#include <cmath>

namespace {

/// Shoelace area for quad vertex order (0,1), (1,3), (3,2), (2,0) — matches quad_mask.vert triangle strip.
double narrowReprojectionQuadAreaPx2(const glm::vec3 c[4]) {
    const glm::vec2 v[4] = {
        glm::vec2(c[0]), glm::vec2(c[1]), glm::vec2(c[3]), glm::vec2(c[2]),
    };
    double a = 0.0;
    for (int i = 0; i < 4; i++) {
        const int j = (i + 1) % 4;
        a += static_cast<double>(v[i].x) * static_cast<double>(v[j].y);
        a -= static_cast<double>(v[j].x) * static_cast<double>(v[i].y);
    }
    return std::abs(a) * 0.5;
}

} // namespace

using namespace quasar;

QUASARStreamer::QUASARStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QUASARStreamerCreateParams& params)
    : quadSet(quadSet)
    , videoURL(params.videoURL)
    , proxiesURL(params.proxiesURL)
    , wideFovImageDumpDir(params.wideFovImageDumpDir)
    , maxLayers(params.maxLayers)
    , wideFovPoseLagFrames(params.wideFovPoseLagFrames)
    , wideFovUpdatePeriodFrames(params.wideFovUpdatePeriodFrames != 0u ? params.wideFovUpdatePeriodFrames : 1u)
    , remoteRenderer(remoteRenderer)
    , remoteRendererDP(remoteRendererDP)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
    , frameGenerator(quadSet)
    , referenceFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , referenceFrameRT_noTone({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameMaskRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameRT_noTone({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
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
    }, params.videoURL, params.targetFramerate, params.targetBitRate)
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
    , quadMaskShader({
        .vertexCodeData = SHADER_COMMON_QUAD_MASK_VERT,
        .vertexCodeSize = SHADER_COMMON_QUAD_MASK_VERT_len,
        .fragmentCodeData = SHADER_COMMON_QUAD_MASK_FRAG,
        .fragmentCodeSize = SHADER_COMMON_QUAD_MASK_FRAG_len,
    })
    , debugMaskRT({
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
    , alphaCodec(alphaAtlasRT.width, alphaAtlasRT.height)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture, residualFrameRT_noTone.alphaTexture)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(params.proxiesURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    wideFovNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    referenceFrames.resize(maxLayers);
    geometryMetadatas.resize(maxLayers);

    uint numHidLayers = maxLayers - 1;
    frameRTsHidLayer.reserve(numHidLayers);
    frameRTsHidLayer_noTone.reserve(numHidLayers);
    meshesHidLayer.reserve(numHidLayers);
    depthMeshesHidLayer.reserve(numHidLayers);
    nodesHidLayer.reserve(numHidLayers);
    wireframesHidLayer.reserve(numHidLayers);
    depthNodesHidLayer.reserve(numHidLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    remoteCameraWideFOV.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
    wideFovReuseViewMatrix = remoteCamera.getViewMatrix();

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
        frameRTsHidLayer.emplace_back(rtParams);
        frameRTsHidLayer_noTone.emplace_back(rtParams);
    }

    // Setup visible layer for reference frame
    for (int i = 0; i < meshScenes.size(); i++) {
        referenceFrameMeshes.emplace_back(
            quadSet, referenceFrameRT_noTone.colorTexture, referenceFrameRT_noTone.alphaTexture);

        referenceFrameNodes.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&referenceFrameNodes[i]);

        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodesLocal[i].frustumCulled = false;

        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameWireframesLocal[i].frustumCulled = false;
        referenceFrameWireframesLocal[i].wireframe = true;
        referenceFrameWireframesLocal[i].visible = false;
        referenceFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
    }

    // Setup masks for residual frame
    residualFrameNode.addEntity(&residualFrameMesh);
    residualFrameNode.frustumCulled = false;
    residualFrameNodeLocal.addEntity(&residualFrameMesh);
    residualFrameNodeLocal.frustumCulled = false;

    residualFrameWireframeLocal.addEntity(&residualFrameMesh);
    residualFrameWireframeLocal.frustumCulled = false;
    residualFrameWireframeLocal.wireframe = true;
    residualFrameWireframeLocal.visible = false;
    residualFrameWireframeLocal.overrideMaterial = &maskWireframeMaterial;

    // Setup depth mesh
    depthNode.addEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    for (int layer = 0; layer < numHidLayers; layer++) {
        meshesHidLayer.emplace_back(
            quadSet, 
            frameRTsHidLayer_noTone[layer].colorTexture, 
            frameRTsHidLayer_noTone[layer].alphaTexture);
        if (layer == numHidLayers - 1) {
            // Increase expand amount by 3px for wide FOV
            // This makes it so that we can merge more and still cover holes
            meshesHidLayer[layer].setExpandQuadAmount(3.0f);
        }

        nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
        nodesHidLayer[layer].frustumCulled = false;

        const glm::vec4& color = colors[(layer + 1) % colors.size()];

        wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
        wireframesHidLayer[layer].frustumCulled = false;
        wireframesHidLayer[layer].wireframe = true;
        wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

        depthMeshesHidLayer.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        depthNodesHidLayer.emplace_back(&depthMeshesHidLayer[layer]);
        depthNodesHidLayer[layer].frustumCulled = false;
        depthNodesHidLayer[layer].visible = false;
        depthNodesHidLayer[layer].primitiveType = GL_POINTS;
    }

    if (numHidLayers > 0) {
        const uint wideLayerIdx = numHidLayers - 1;
        wideFovQuadTexelUsageMaterial = std::make_unique<QuadTexelUsageMaterial>(QuadMaterialCreateParams{
            .baseColorTexture = &frameRTsHidLayer_noTone[wideLayerIdx].colorTexture,
            .alphaTexture = &frameRTsHidLayer_noTone[wideLayerIdx].alphaTexture,
        });
        nodesHidLayer[wideLayerIdx].overrideMaterial = wideFovQuadTexelUsageMaterial.get();
    }

    // Setup scene to use as mask for wide fov camera
    for (int i = 0; i < meshScenes.size(); i++) {
        wideFovNodes.emplace_back(&referenceFrameMeshes[i]);
        wideFovNodes[i].frustumCulled = false;
        sceneWideFov.addChildNode(&wideFovNodes[i]);
    }
    for (int i = 0; i < numHidLayers - 1; i++) {
        sceneWideFov.addChildNode(&nodesHidLayer[i]);
    }
    sceneWideFov.addChildNode(&residualFrameNode);

    alphaImageData.resize(alphaAtlasRT.width * alphaAtlasRT.height);

    setViewSphereDiameter(params.viewSphereDiameter);

    if (!videoURL.empty() && !proxiesURL.empty()) {
        spdlog::info("Created QUASARStreamer that sends to URL: tcp://{}", proxiesURL);
    }

    // strip by the last / in the wideFovImageDumpDir
    std::string outputDir = wideFovImageDumpDir.substr(0, wideFovImageDumpDir.find_last_of('/'));

    quasarStatsCSVFileName = outputDir + "/quasar_stats.csv";
    quasarStatsCSVFile.open(quasarStatsCSVFileName);
    quasarStatsCSVFile << "frame_id";
    quasarStatsCSVFile << ",visible_render";
    for (int layer = 0; layer < maxLayers; layer++) {
        quasarStatsCSVFile << ",layer_" << layer << "_create_proxies";
        quasarStatsCSVFile << ",layer_" << layer << "_compress";
        quasarStatsCSVFile << ",layer_" << layer << "_create_mesh";
    }
    quasarStatsCSVFile << ",total_compress" << std::endl;
    quasarStatsCSVFile.close();

    bandwidthstats.proxy_size_by_layer.resize(maxLayers);
    bandwidthstats.depth_offset_size_by_layer.resize(maxLayers);
    bandwidthstats.alphaSize = 0;
    bandwidthstats.totalSize = 0;
    
    // add output dir to bandwidth stats

    bandwidthStatsCSVFileName = outputDir + "/quasar_streamer_bitrate.csv";
    bandwidthStatsCSVFile.open(bandwidthStatsCSVFileName);
    bandwidthStatsCSVFile << "frameID";
    bandwidthStatsCSVFile << ",atlas_bitrate";
    bandwidthStatsCSVFile << ",depth_peeling_bitrate";
    // Add for bandwidth reference of each layer, above information is enough
    for (int layer = 0; layer < maxLayers; layer++) {
        bandwidthStatsCSVFile << ",layer_" << layer << "_proxy_size";
        bandwidthStatsCSVFile << ",layer_" << layer << "_depth_offset_size";
    }
    bandwidthStatsCSVFile << std::endl;
    bandwidthStatsCSVFile.close();

    // Given the projection matrix of wideFov and the normal projection matrix, 
    // we can pre-compute the corners of the normal view space in the wide fov space
    glm::mat4 wideFovProjectionMatrix = remoteCameraWideFOV.getProjectionMatrix();
    glm::mat4 normalProjectionMatrix = remoteCamera.getProjectionMatrix();
    glm::vec3 corners[4] = {
        glm::vec3(0, 0, 1.0),
        glm::vec3(quadSet.getSize().x, 0, 1.0),
        glm::vec3(0, quadSet.getSize().y, 1.0),
        glm::vec3(quadSet.getSize().x, quadSet.getSize().y, 1.0),
    };
    for (int i = 0; i < 4; i++) {
        corners[i] = glm::vec3(corners[i].x, corners[i].y, corners[i].z);
        corners[i] = glm::unProject(
            corners[i], 
            normalProjectionMatrix, 
            glm::mat4(1.0f), 
            glm::vec4(0.0f, 0.0f, remoteRenderer.width, remoteRenderer.height));
        normalViewCornersInWideFoVImage[i] = glm::project(
            corners[i],
            wideFovProjectionMatrix, 
            glm::mat4(1.0f), 
            glm::vec4(0.0f, 0.0f, remoteRenderer.width, remoteRenderer.height));
    }
}

QUASARStreamer::~QUASARStreamer() {
    videoAtlasStreamerRT.stop();
}

uint QUASARStreamer::getNumTriangles() const {
    int currMeshIndex  = meshIndex % 2;
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    uint numTriangles = refMeshSizes.numIndices / 3; // Each triangle has 3 indices
    for (const auto& mesh : meshesHidLayer) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QUASARStreamer::setDrawState(QuadMesh::DrawState drawState) {
    for (auto& mesh : referenceFrameMeshes) {
        mesh.setDrawState(drawState);
    }
    residualFrameMesh.setDrawState(drawState);
    for (auto& mesh : meshesHidLayer) {
        mesh.setDrawState(drawState);
    }
}

void QUASARStreamer::addMeshesToScene(Scene& localScene) {
    // Add in reverse order to have correct layering
    for (int layer = nodesHidLayer.size() - 1; layer >= 0; layer--) {
        localScene.addChildNode(&nodesHidLayer[layer]);
        localScene.addChildNode(&wireframesHidLayer[layer]);
        localScene.addChildNode(&depthNodesHidLayer[layer]);
    }

    for (int i = 0; i < meshScenes.size(); i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNodeLocal);
    localScene.addChildNode(&residualFrameWireframeLocal);
    localScene.addChildNode(&depthNode);
}

void QUASARStreamer::setViewSphereDiameter(float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;
    remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);
}

RenderStats QUASARStreamer::generateFrame(
    bool createResidualFrame,
    bool showNormals,
    bool showDepth,
    const glm::mat4* wideFovGroundTruthView) {
    // Reset stats
    Stats prevStats = stats;
    stats = { 0 };
    stats.frameSize = prevStats.frameSize; // Keep previous frame size

    // Draw all meshes for proper masking
    setDrawState(QuadMesh::DrawState::BOTH);

    int currMeshIndex  = meshIndex % 2;
    int prevMeshIndex  = (meshIndex + 1) % 2;

    // Wide-FOV pose lag: ring buffer of remote views; pickIndex selects the lagged view when we refresh the snapshot.
    wideFovCameraViewHistory.push_back(remoteCamera.getViewMatrix());
    const size_t lag = static_cast<size_t>(wideFovPoseLagFrames);
    const size_t maxHistory = lag + 1u;
    while (wideFovCameraViewHistory.size() > maxHistory) {
        wideFovCameraViewHistory.pop_front();
    }
    const size_t pickIndex = wideFovCameraViewHistory.size() > lag ? wideFovCameraViewHistory.size() - 1u - lag : 0u;

    auto quadsGenerator = frameGenerator.getQuadsGenerator();

    /*
    ============================
    Render scene normally to create Reference Frame textures
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);
    stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    
    frameID++;
    spdlog::info("Frame ID: {}", frameID);
    // open file 
    quasarStatsCSVFile.open(quasarStatsCSVFileName, std::ios::app);
    quasarStatsCSVFile << frameID;
    quasarStatsCSVFile << "," << stats.totalRenderTimeMs;
    for (int layer = 0; layer < maxLayers; layer++) {
        int hiddenLayerIndex = layer - 1;
        const bool isWideFovLayer = (layer == maxLayers - 1);

        auto& remoteCameraToUse = (layer == 0 && createResidualFrame)
                                    ? remoteCameraPrev
                                    : ((layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV);

        auto& renderTargetToUse        = (layer == 0) ? referenceFrameRT        : frameRTsHidLayer[hiddenLayerIndex];
        auto& renderTargetToUse_noTone = (layer == 0) ? referenceFrameRT_noTone : frameRTsHidLayer_noTone[hiddenLayerIndex];

        auto& meshToUse      = (layer == 0) ? referenceFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenLayerIndex];
        auto& meshToUseDepth = (layer == 0) ? depthMesh                           : depthMeshesHidLayer[hiddenLayerIndex];

        startTime = timeutils::getTimeMicros();
        if (layer == 0) {
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
            remoteRenderer.copyToFrameRT(renderTargetToUse);
        }
        else if (layer < maxLayers - 1) {
            // Hidden layers need to use the noTone render targets to generate quads for some reason...
            remoteRendererDP.peelingLayers[hiddenLayerIndex+1].blit(renderTargetToUse_noTone);
            // renderTargetToUse_noTone.writeColorAsPNG("quasar_hid_layer_no_tone_" + std::to_string(layer) + ".png");
        }
        // Wide fov camera
        else {
            // Draw old center mesh at new remoteCamera layer, filling stencil buffer with 1
            bool trimWideFov = false;

            glm::mat4 prevProjectionMatrix = wideFovGroundTruthView != nullptr
                ? remoteCamera.getProjectionMatrix()
                : remoteCameraPrev.getProjectionMatrix();
            // glm::mat4 prevViewMatrix = remoteCameraPrev.getViewMatrix();
            glm::mat4 prevViewMatrix = wideFovGroundTruthView != nullptr
                ? remoteCamera.getViewMatrix()
                : remoteCameraPrev.getViewMatrix();

            glm::mat4 currentProjectionMatrix = remoteCamera.getProjectionMatrix();
            glm::mat4 currentViewMatrix = wideFovGroundTruthView != nullptr 
                ? *wideFovGroundTruthView : 
                remoteCamera.getViewMatrix();

            // work reversely reproject the new pose into the old pose's wide fov space
            glm::mat4 projectionMatrixWideFOV = remoteCameraWideFOV.getProjectionMatrix();
            
            spdlog::info("Current viewport size: ({}, {})", quadSet.getSize().x, quadSet.getSize().y);


            glm::mat4 prevViewMatrixInverse = glm::inverse(prevViewMatrix);
            glm::mat4 currentViewMatrixInverse = glm::inverse(currentViewMatrix);

            spdlog::info("  Previous Position: ({:.3f}, {:.3f}, {:.3f})", prevViewMatrixInverse[3][0], prevViewMatrixInverse[3][1], prevViewMatrixInverse[3][2]);
            spdlog::info("  Current Position: ({:.3f}, {:.3f}, {:.3f})", currentViewMatrixInverse[3][0], currentViewMatrixInverse[3][1], currentViewMatrixInverse[3][2]);
            
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
                
                newCorners[i].x = std::max(newCorners[i].x, 0.0f);
                newCorners[i].x = std::min(newCorners[i].x, float(quadSet.getSize().x));
        
                newCorners[i].y = std::max(newCorners[i].y, 0.0f);
                newCorners[i].y = std::min(newCorners[i].y, float(quadSet.getSize().y));
            }


            remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
        
            // if (totalBlackArea > 10000.0f) {
            if (trimWideFov) {
            const glm::vec2 viewportSize(remoteRenderer.width, remoteRenderer.height);
            // remoteRenderer.gBuffer.unbind();
        
            // Pass 1: Render the uncovered area in the wideFov Image
            // By doing the reprojection from the normal view to the wide fov image using projection matrix
            // The rectangular area is distorted, some pixels are overflowed into wide fov region, 
            // resulting in "uncovered" areas in the normal view
            
            // get the region of the union of reprojected corners and normal view corners
            glm::vec3 reprojectedCornersInWideFoVImage[4];
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
                reprojectedCornersInWideFoVImage[i] = cornerInSourceWideFoVImage;

            }
            
            glm::vec3 maskCorners[4];
            maskCorners[0] = glm::vec3(std::min(normalViewCornersInWideFoVImage[0].x, reprojectedCornersInWideFoVImage[0].x),
                                       std::min(normalViewCornersInWideFoVImage[0].y, reprojectedCornersInWideFoVImage[0].y), 1.0f);
            maskCorners[1] = glm::vec3(std::max(normalViewCornersInWideFoVImage[1].x, reprojectedCornersInWideFoVImage[1].x),
                                       std::min(normalViewCornersInWideFoVImage[1].y, reprojectedCornersInWideFoVImage[1].y), 1.0f);
            maskCorners[2] = glm::vec3(std::min(normalViewCornersInWideFoVImage[2].x, reprojectedCornersInWideFoVImage[2].x),
                                       std::max(normalViewCornersInWideFoVImage[2].y, reprojectedCornersInWideFoVImage[2].y), 1.0f);
            maskCorners[3] = glm::vec3(std::max(normalViewCornersInWideFoVImage[3].x, reprojectedCornersInWideFoVImage[3].x),
                                       std::max(normalViewCornersInWideFoVImage[3].y, reprojectedCornersInWideFoVImage[3].y), 1.0f);
        
            // TODO: Check whether we need any heuristic way to expand the mask region
            // double the distance of the mask corners
            // compute the vector from the mask corners to the normal view corners
            // we compute from mask corner to the normal view corner, and then double the distance
            glm::vec3 maskCornersExpanded[4];
            for (int i = 0; i < 4; i++) {
                glm::vec3 vector = maskCorners[i] - normalViewCornersInWideFoVImage[i];
                maskCornersExpanded[i] = maskCorners[i] + 5.0f * vector;
            }
        
            remoteRenderer.gBuffer.bind();
            // remoteRenderer.outputRT.bind();
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            // remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            // We use the existing rect region as the mask: Those are covered, no need to draw
            // remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.apply();
            glClearStencil(0);
            glClear(GL_STENCIL_BUFFER_BIT);
            quadMaskShader.bind();
            quadMaskShader.setVec4("uDebugColor", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
            quadMaskShader.setVec2("uViewport", viewportSize);
            quadMaskShader.setVec2("uCorners[0]", maskCornersExpanded[0]);
            quadMaskShader.setVec2("uCorners[1]", maskCornersExpanded[1]);
            quadMaskShader.setVec2("uCorners[2]", maskCornersExpanded[2]);
            quadMaskShader.setVec2("uCorners[3]", maskCornersExpanded[3]);
            quadMaskQuad.draw();
            // remoteRenderer.outputRT.unbind();
            remoteRenderer.gBuffer.unbind();
        
            // Pass 1: render the normal view scene range into the gbuffer
            // those are the parts that are visible in the normal view, so no need to draw them in the wideFov
            // use it as a stencil mask to avoid drawing them again
        
            remoteRenderer.gBuffer.bind();
            // remoteRenderer.outputRT.bind();
            remoteRenderer.pipeline.stencilState.stencilRef = 0;
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_ZERO, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 0);
            remoteRenderer.pipeline.stencilState.writeStencilMask = 0xFF;
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            remoteRenderer.pipeline.apply();
            
            quadMaskShader.bind();
            quadMaskShader.setVec4("uDebugColor", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
            quadMaskShader.setVec2("uViewport", viewportSize);
            quadMaskShader.setVec2("uCorners[0]", normalViewCornersInWideFoVImage[0]);
            quadMaskShader.setVec2("uCorners[1]", normalViewCornersInWideFoVImage[1]);
            quadMaskShader.setVec2("uCorners[2]", normalViewCornersInWideFoVImage[2]);
            quadMaskShader.setVec2("uCorners[3]", normalViewCornersInWideFoVImage[3]);
            quadMaskQuad.draw();
            // remoteRenderer.outputRT.unbind();
            // log out the mask image
            // remoteRenderer.outputRT.writeColorAsPNG("debug_mask_" + std::to_string(frameID) + ".png");
            remoteRenderer.gBuffer.unbind();
        
            // use the previous generated stencil buffer to draw only the uncovered area
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_EQUAL, 1);
            } else {
            
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                // remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                wideFovNodes[currMeshIndex].visible = true;
                wideFovNodes[prevMeshIndex].visible = false;
                renderStats += remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);
                // remoteRendereer.outputRT.writeColorAsPNG("quasar_wide_fov_no_tone.png");

                // Render remoteScene using stencil buffer as a mask
                // At values where stencil buffer is not 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            }
            
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();
            remoteRenderer.copyToFrameRT(renderTargetToUse);

            // log out the rendered wide fov image to the dump directory
            if (!wideFovImageDumpDir.empty()) {
                Path dumpDir(wideFovImageDumpDir);
                dumpDir.mkdirRecursive();
                Path pngPath = dumpDir / ("widefov_" + std::to_string(frameID) + ".png");
                renderTargetToUse.writeColorAsPNG(pngPath.str());
            }

        }
        stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        auto oldParams = quadsGenerator->params;
        // Wide FOV has very loose parameters to reduce data size
        if (layer == maxLayers - 1) {
            quadsGenerator->params.planeSimilarityThreshold *= 4.0f;
            quadsGenerator->params.expandEdges = true;
        }
        // Hidden layers have looser parameters to reduce data size
        else if (layer > 0) {
            quadsGenerator->params.planeSimilarityThreshold *= (layer * 2.0f);
            quadsGenerator->params.expandEdges = false;
        }
        ReferenceFrame dummyFrame;
        glFinish();
        frameGenerator.createReferenceFrame(
            (layer != 0 && layer != maxLayers - 1) ? renderTargetToUse_noTone : renderTargetToUse,
            remoteCameraToUse,
            meshToUse,
            (layer == 0 && createResidualFrame) ? dummyFrame : referenceFrames[layer] // Don't save output of this reference frame if we are making a residual frame
        );
        if (!showNormals) {
            if (layer == 0) {
                remoteRenderer.copyToFrameRT(referenceFrameRT_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, referenceFrameRT);
            }
            else if (layer < maxLayers - 1) {
                tonemapper.setUniforms(renderTargetToUse_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
            }
            else {
                remoteRenderer.copyToFrameRT(renderTargetToUse_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse);
            }
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, renderTargetToUse_noTone);
        }
        quadsGenerator->params = oldParams;

        stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
        stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
        stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

        stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
        stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
        stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;


        stats.createProxiesTimeMsByLayer.push_back(frameGenerator.stats.createQuadsTimeMs);
        stats.compressTimeMsByLayer.push_back(frameGenerator.stats.compressTimeMs);
        stats.createMeshTimeMsByLayer.push_back(frameGenerator.stats.createMeshTimeMs);

        if (!createResidualFrame || layer != 0) {
            stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (layer == 0) {
            if (createResidualFrame) {
                /*
                ============================
                Generate masked Residual Frame textures
                ============================
                */
                frameGenerator.updateResidualRenderTargets(
                    residualFrameMaskRT, residualFrameRT,
                    remoteRenderer, remoteScene,
                    meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                    remoteCamera, remoteCameraPrev
                );

                /*
                ============================
                Generate Residual Frame
                ============================
                */
                quadsGenerator->params.expandEdges = true;
                frameGenerator.createResidualFrame(
                    residualFrameMaskRT, residualFrameRT,
                    remoteCamera, remoteCameraPrev,
                    referenceFrameMeshes[prevMeshIndex], residualFrameMesh,
                    residualFrame
                );
                if (!showNormals) {
                    residualFrameRT.blit(residualFrameRT_noTone);
                    tonemapper.setUniforms(residualFrameRT_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, residualFrameRT, false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, residualFrameRT_noTone);
                }

                stats.totalRenderTimeMs += frameGenerator.stats.updateRTsTimeMs;

                stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
                stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
                stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
                stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

                stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
                stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
                stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

                stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
            }
            else {
                // Only update the previous camera pose if we are not generating a Residual Frame
                remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
                lastMeshIndex = meshIndex;
                meshIndex++;
            }

            residualFrameNode.visible = createResidualFrame;
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            meshToUseDepth.update((layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV, renderTargetToUse);
            stats.totalGenDepthTimeMs += meshToUseDepth.stats.genDepthTime;
        }

        if (!(createResidualFrame && layer == 0)) {
            stats.proxySizes.numQuads += referenceFrames[layer].getTotalNumQuads();
            stats.proxySizes.numDepthOffsets += referenceFrames[layer].getTotalNumDepthOffsets();
            stats.proxySizes.quadsSize += referenceFrames[layer].getTotalQuadsSize();
            stats.proxySizes.depthOffsetsSize += referenceFrames[layer].getTotalDepthOffsetsSize();
            spdlog::debug("Reference frame generated with {} quads ({:.3f}MB), {} depth offsets ({:.3f}MB)",
                          referenceFrames[layer].getTotalNumQuads(), referenceFrames[layer].getTotalQuadsSize() / BYTES_PER_MEGABYTE,
                          referenceFrames[layer].getTotalNumDepthOffsets(), referenceFrames[layer].getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
        }
        else {
            stats.proxySizes.numQuads += residualFrame.getTotalNumQuads();
            stats.proxySizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
            stats.proxySizes.quadsSize += residualFrame.getTotalQuadsSize();
            stats.proxySizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
            spdlog::debug("Residual frame generated with {} updated quads ({:.3f}MB) and {} revealed quads ({:.3f}MB), {} updated depth offsets ({:.3f}MB) and {} revealed depth offsets ({:.3f}MB)",
                          residualFrame.getTotalNumQuadsUpdated(), residualFrame.getTotalQuadsUpdatedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumQuadsRevealed(), residualFrame.getTotalQuadsRevealedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumDepthOffsetsUpdated(), residualFrame.getTotalDepthOffsetsUpdatedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumDepthOffsetsRevealed(), residualFrame.getTotalDepthOffsetsRevealedSize() / BYTES_PER_MEGABYTE);
        }
    }
     
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    for (int layer = 0; layer < maxLayers; layer++) {
        quasarStatsCSVFile << "," << stats.createProxiesTimeMsByLayer[layer];
        quasarStatsCSVFile << "," << stats.compressTimeMsByLayer[layer];
        quasarStatsCSVFile << "," << stats.createMeshTimeMsByLayer[layer];
    }
    quasarStatsCSVFile << "," << stats.totalCompressTimeMs << std::endl;
    quasarStatsCSVFile.close();

    // Update color and alpha atlases (tile frames side by side)
    uint row = 0, col = 0;
    uint dstWidth = referenceFrameRT.width, dstHeight = referenceFrameRT.height;
    for (int layer = 0; layer < maxLayers; layer++) {
        if (layer == 0) {
            referenceFrameRT.blit(videoAtlasStreamerRT,
                0, 0, referenceFrameRT.width, referenceFrameRT.height,
                col, row, dstWidth, dstHeight
            );
            referenceFrameRT.blit(alphaAtlasRT,
                0, 0, referenceFrameRT.width, referenceFrameRT.height,
                col, row, dstWidth, dstHeight
            );
        }
        else {
            int hiddenLayerIndex = layer - 1;
            frameRTsHidLayer[hiddenLayerIndex].blit(videoAtlasStreamerRT,
                0, 0, frameRTsHidLayer[hiddenLayerIndex].width, frameRTsHidLayer[hiddenLayerIndex].height,
                col, row, dstWidth, dstHeight
            );
            frameRTsHidLayer_noTone[hiddenLayerIndex].blit(alphaAtlasRT,
                0, 0, frameRTsHidLayer_noTone[hiddenLayerIndex].width, frameRTsHidLayer_noTone[hiddenLayerIndex].height,
                col, row, dstWidth, dstHeight
            );
        }
        col += referenceFrameRT.width;
        dstWidth += referenceFrameRT.width;
        if (col >= videoAtlasStreamerRT.width) {
            col = 0;
            dstWidth = referenceFrameRT.width;

            row += referenceFrameRT.height;
            dstHeight += referenceFrameRT.height;
            if (row >= videoAtlasStreamerRT.height) {
                row = 0;
                dstHeight = referenceFrameRT.height;
            }
        }
    }
    residualFrameRT.blit(videoAtlasStreamerRT,
        0, 0, residualFrameRT.width, residualFrameRT.height,
        col, row, dstWidth, dstHeight
    );
    residualFrameRT_noTone.blit(alphaAtlasRT,
        0, 0, residualFrameRT_noTone.width, residualFrameRT_noTone.height,
        col, row, dstWidth, dstHeight
    );

    // videoAtlasStreamerRT.writeColorAsPNG("debug_quasar_video_atlas.png");
    // alphaAtlasRT.writeAlphaAsPNG("debug_quasar_alpha_atlas.png");

    return renderStats;
}

void QUASARStreamer::sendFrame(PoseReceiver::PoseInfo poseInfo, bool createResidualFrame) {

    stats.frameSize = writeToMemory(poseInfo, createResidualFrame, compressedData);
    size_t compressedDataSize = compressedData.size();
    if (!videoURL.empty() && !proxiesURL.empty()) {
        // Send atlas frame
        videoAtlasStreamerRT.sendFrame(poseInfo.pose_id);
        // Send proxies
        send(compressedData);
    }

    if (prevSendTimeMs != 0.0) {
        
        double compressedDataSendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevSendTimeMs);
        double bitrateMbps = ((8.0 * compressedDataSize) / BYTES_PER_MEGABYTE) / timeutils::millisToSeconds(compressedDataSendTimeMs);

        bandwidthStatsCSVFile.open(bandwidthStatsCSVFileName, std::ios::app);
        bandwidthStatsCSVFile << frameID;
        bandwidthStatsCSVFile << "," << videoAtlasStreamerRT.stats.bitrateMbps;
        bandwidthStatsCSVFile << "," << bitrateMbps;
        for (int layer = 0; layer < maxLayers; layer++) {
            bandwidthStatsCSVFile << "," << bandwidthstats.proxy_size_by_layer[layer];
            bandwidthStatsCSVFile << "," << bandwidthstats.depth_offset_size_by_layer[layer];
        }
        bandwidthStatsCSVFile << std::endl;
        bandwidthStatsCSVFile.close();
    }
    prevSendTimeMs = timeutils::getTimeMicros();
}

void QUASARStreamer::writeTexturesToFiles(const Path& outputPath) {
    // Save color
    Path colorFileName = (outputPath / "color.jpg");
    videoAtlasStreamerRT.writeColorAsJPG(colorFileName);

    // Save alpha
    Path alphaFileName = (outputPath / "alpha.png");
    alphaAtlasRT.writeAlphaAsPNG(alphaFileName);
}

size_t QUASARStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = outputPath / "camera.bin";
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    Path cameraFileNamePrev = outputPath / "camera_prev.bin";
    cameraPose.setProjectionMatrix(remoteCameraPrev.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCameraPrev.getViewMatrix());
    cameraPose.writeToFile(cameraFileNamePrev);

    // Save metadata (viewSphereDiameter and wide FOV)
    QUASARReceiver::Params params = {
        .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
        .viewSphereDiameter = viewSphereDiameter,
        .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
    };
    FileIO::writeToBinaryFile(outputPath / "metadata.bin", &params, sizeof(params));

    writeTexturesToFiles(outputPath);

    // Save proxies
    size_t totalOutputSize = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        totalOutputSize += referenceFrames[layer].writeToFiles(outputPath, layer);
    }
    totalOutputSize += residualFrame.writeToFiles(outputPath);
    return totalOutputSize;
}

size_t QUASARStreamer::writeToMemory(PoseReceiver::PoseInfo poseInfo, bool writeResidualFrame, std::vector<char>& outputData) {
    // Save camera data
    Pose cameraPose;
    pose_id_t poseID = poseInfo.pose_id;
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save alpha data
    alphaAtlasRT.writeAlphaToMemory(alphaImageData);
    alphaCodec.compress(alphaImageData.data(), alphaData, alphaImageData.size());

    // Save geometry data
    // Save visible layer
    if (!writeResidualFrame) {
        referenceFrames[0].writeToMemory(geometryMetadatas[0]);
    }
    else {
        residualFrame.writeToMemory(geometryMetadatas[0]);
    }
    // Save hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; layer++) {
        referenceFrames[layer].writeToMemory(geometryMetadatas[layer]);
    }

    uint32_t geometrySize = 0;
    for (const auto& layerData : geometryMetadatas) {
        geometrySize += sizeof(uint32_t) + static_cast<uint32_t>(layerData.size());
    }

    double timestamp = double(timeutils::getTimeMicros());
    spdlog::info("Timestamp: {}", timestamp);

    QUASARReceiver::Header header{
        .poseID = poseID,
        .frameType = !writeResidualFrame ? QuadFrame::FrameType::REFERENCE : QuadFrame::FrameType::RESIDUAL,
        .params {
            .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
        },
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .alphaSize = static_cast<uint32_t>(alphaData.size()),
        .geometrySize = geometrySize,
        .pose_send_timestamp = poseInfo.send_timestamp,
        .pose_recv_timestamp = poseInfo.recv_timestamp,
        .frame_send_timestamp = timestamp,
    };

    spdlog::debug("Writing camera size: {:.3f}MB", static_cast<float>(header.cameraSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing alpha size: {:.3f}MB", static_cast<float>(header.alphaSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing geometry size: {:.3f}MB", static_cast<float>(header.geometrySize) / BYTES_PER_MEGABYTE);

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
    for (const auto& layerData : geometryMetadatas) {
        uint32_t layerSize = static_cast<uint32_t>(layerData.size());

        // Write size of layer
        std::memcpy(ptr, &layerSize, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Write layer data
        std::memcpy(ptr, layerData.data(), layerSize);
        ptr += layerSize;
    }

    spdlog::debug("Total data size: {:.3f}MB", static_cast<float>(outputData.size()) / BYTES_PER_MEGABYTE);

    for (int layer = 0; layer < maxLayers; layer++) {
        bandwidthstats.proxy_size_by_layer[layer] = referenceFrames[layer].getTotalQuadsSize();
        bandwidthstats.depth_offset_size_by_layer[layer] = referenceFrames[layer].getTotalDepthOffsetsSize();
    }
    bandwidthstats.alphaSize = alphaData.size();
    bandwidthstats.totalSize = outputData.size();
    return outputData.size();
}
