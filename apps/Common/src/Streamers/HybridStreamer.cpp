#include "Streamers/HybridStreamer.h"
#include <shaders_common.h>

#include <nvtx3/nvToolsExt.h>

using namespace quasar;

HybridStreamer::HybridStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer, 
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const HybridStreamerCreateParams& params)
    : videoURL(params.videoURL)
    , depthAndProxiesURL(params.depthAndProxiesURL)
    , hiddenLayers(params.hiddenLayers)
    , remoteRendererDP(remoteRendererDP)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
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
    }, depthAndProxiesURL, params.maxFrameRate)
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
    }, depthAndProxiesURL, params.maxFrameRate)
    , depthEffect(remoteCamera)
;

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

RenderStats HybridStreamer::generateFrame() {
    RenderStats renderStats;

    /*
    ============================
    Visible layer meshwarping
    ============================
    */

    // normal visible layer
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerRT);
    depthStreamerRT.generateFrame();

    // wide fov visible layer
    // based on the visible layer, draw wide fov
    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(
        GL_KEEP, GL_KEEP, GL_REPLACE);
    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
    renderStats += remoteRenderer.drawObjectsNoLighting(
        sceneWideFov, remoteCameraWideFOV);
    
    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
    remoteRenderer.pipeline.writeMaskState.enableColorWrites();

    // render into depthStreamerWideFOV
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerWideFOV);
    depthStreamerWideFOV.generateFrame();

    /* 
    ============================
    Hidden Layer depth Peeling
    ============================
    */
    // Render all the objects in the scene
    renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);

    for (int layer = 0; layer < hiddenLayers; layer++) {
        
        // Always use the remoteCamera
        auto& renderTargetToUse = frameRTsHidLayer[layer];
        auto& renderTargetToUse_noTone = frameRTsHidLayer_noTone[layer];
        auto& meshToUse = meshesHidLayer[layer];
        

        // blit the hidden layer from depth peeling renderer
        remoteRendererDP.peelingLayers[layer + 1].blit(renderTargetToUse_noTone);

        /*
        ============================
        Generate hidden layer reference frames
        ============================
        */

        frameGenerator.createReferenceFrames(
            renderTargetToUse_noTone, 
            remoteCamera, 
            meshToUse, 
            referenceFrames[layer]);
        
        tonemapper.setUniforms(renderTargetToUse_noTone);
        tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
    }

    subFrameIndex atlasIndex = {0, 0};
    uint subFrameWidth = depthStreamerRT.width;
    uint subFrameHeight = depthStreamerRT.height;

    // blit the default layer, directly from depth peeling renderer
    depthStreamerRT.blit(
        videoAtlasStreamerRT.frameRT, 0, 0, 
        subFrameWidth, 
        subFrameHeight, 
        atlasIndex.col, 
        atlasIndex.row, 
        atlasIndex.col + subFrameWidth, 
        atlasIndex.row + subFrameHeight
    );

    depthStreamerRT.blit(
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
            videoAtlasStreamerRT.frameRT, 0, 0, 
            frameRTsHidLayer[i].width, 
            frameRTsHidLayer[i].height, 
            atlasIndex.col, 
            atlasIndex.row, 
            atlasIndex.col + subFrameWidth, 
            atlasIndex.row + subFrameHeight
        );

        frameRTsHidLayer[i].blit(
            alphaAtlasRT, 0, 0, 
            frameRTsHidLayer[i].width, 
            frameRTsHidLayer[i].height, 
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

    // blit the wide-fov layer, from meshwarp streamer
    depthStreamerWideFOV.blit(
        videoAtlasStreamerRT.frameRT, 0, 0, 
        subFrameWidth, 
        subFrameHeight, 
        atlasIndex.col, 
        atlasIndex.row, 
        atlasIndex.col + subFrameWidth, 
        atlasIndex.row + subFrameHeight
    );

    depthStreamerWideFOV.blit(
        alphaAtlasRT, 0, 0, 
        subFrameWidth, 
        subFrameHeight, 
        atlasIndex.col, 
        atlasIndex.row, 
        atlasIndex.col + subFrameWidth, 
        atlasIndex.row + subFrameHeight
    );
}


void HybridStreamer::writeToMemory(pose_id_t poseID, std::vector<char>& outputData) {


    // combine depth information and proxies into one 
    // compressed data
    vector<char> depthData;
    vector<char> proxyData;
    vector<char> depthDataWideFOV;
    vector<char> combinedData;

    depthStreamerRT.writeToMemory(poseID, depthData);
    depthStreamerWideFOV.writeToMemory(poseID, depthDataWideFOV);
    
    // compress the hidden layers using the same as QUASARStreamer
    vector<char> cameraData;
    vector<char> alphaData;
    vector<vector<char>> geometryMetadatas;
    geometryMetadatas.resize(hiddenLayers);
    
    Pose cameraPose;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save alpha data
    alphaAtlasRT.writeAlphaToMemory(alphaImageData);
    alphaCodec.compress(alphaImageData.data(), alphaData, alphaImageData.size());

    // Save hidden layers and wide FOV
    for (int layer = 0; layer < hiddenLayers; layer++) {
        frameRTsHidLayer[layer].writeToMemory(geometryMetadatas[layer]);
    }

    uint32_t geometrySize = 0;
    for (const auto& geomData : geometryMetadatas) {
        geometrySize += sizeof(uint32_t) + static_cast<uint32_t>(geomData.size());
    }

    // Prepare header
    HybridReceiver::Header header{
        .poseID = poseID,
        .params = {
            .hiddenLayers = hiddenLayers,
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
        },
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .alphaSize = static_cast<uint32_t>(alphaData.size()),
        .geometrySize = geometrySize,
        .visibleLayerSize = static_cast<uint32_t>(videoAtlasStreamerRT.getFrameSize()),
        .visibleLayerWideFovSize = static_cast<uint32_t>(depthDataWideFOV.size()),
    };

    outputData.resize(header.getSize());
    memcpy(outputData.data(), &header, sizeof(header));
    size_t offset = sizeof(header);

    // Write camera data
    memcpy(outputData.data() + offset, cameraData.data(), cameraData.size());
    offset += cameraData.size();

    // Write alpha data
    memcpy(outputData.data() + offset, alphaData.data(), alphaData.size());
    offset += alphaData.size();

    // Write geometry data
    for (const auto& geomData : geometryMetadatas) {
        memcpy(outputData.data() + offset, geomData.data(), geomData.size());
        offset += geomData.size();
    }

    // Write depth data
    memcpy(outputData.data() + offset, depthData.data(), depthData.size());
    offset += depthData.size();

    // Write wide FOV depth data
    memcpy(outputData.data() + offset, depthDataWideFOV.data(), depthDataWideFOV.size());
    offset += depthDataWideFOV.size();
}

void HybridStreamer::sendFrame(pose_id_t poseID) {
    // send the videos
    videoAtlasStreamerRT.sendFrame(poseID);

    // send the proxies
    size_t outputSize = 0;
    std::vector<char> outputData;
    writeToMemory(poseID, outputData);
    outputSize = outputData.size();

    send(outputData);
}
