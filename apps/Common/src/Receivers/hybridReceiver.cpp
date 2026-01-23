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
    , remoteCamera(quadSet.getSize())
    , remoteCameraWideFOV(quadSet.getSize())
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

}

void HybridReceiver::updateMesh(bool isWideFOV) {
    // Set shader uniforms

    PerspectiveCamera& cameraInUse = isWideFOV ? remoteCameraWideFOV : remoteCamera;
    Mesh& meshInUse = isWideFOV ? visibleMeshWideFOV : visibleMesh;
    BC4DepthVideoTexture& depthTextureInUse = isWideFOV ? depthTextureWideFOV : depthTexture;

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
}


void HybridReceiver::recvData(
        const PoseStreamer& poseStreamer,
        double& elapsedTimeColor, 
        double& elapsedTimeDepth) {

    // Render depth video frame
    depthTexture.bind();
    if (sync) {
        poseIdDepth = depthTexture.draw(poseIdColor);
    }
    else {
        poseIdDepth = depthTexture.draw();
    }

    // Get poses for the frames
    poseStreamer.getPose(poseIdColor, &colorFramePose, &elapsedTimeColor);
    poseStreamer.getPose(poseIdDepth, &depthFramePose, &elapsedTimeDepth);

    // Update both visible and wide FOV meshes
    updateMesh(false);
    updateMesh(true);

    // Wait for a frame that has been written to
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            return;
        }

        if (videoAtlasTexture.getLatestPoseID() < framePending->poseID) { // Video is behind, wait until video catches up
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

    // Reconstruct meshes from frame
    reconstructFrame(frame);

    // Reset frame
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return;
}