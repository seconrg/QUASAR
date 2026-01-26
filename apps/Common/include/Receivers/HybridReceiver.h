#ifndef HYBRID_RECEIVER_H
#define HYBRID_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>

#include <CameraPose.h>
#include <Materials/UnlitMaterial.h>

#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>

#include <Receivers/QuadsReceiver.h>
#include <Receivers/QUASARReceiver.h>

#include <Primitives/Mesh.h>
#include <Receivers/VideoTexture.h>
#include <Receivers/BC4DepthVideoTexture.h>

#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Streamers/PoseStreamer.h>

namespace quasar {

class HybridReceiver : public QUASARReceiver {
public: 
    std::string videoVisibleURL;
    std::string videoVisibleWideFovURL;
    std::string depthVisibleURL;
    std::string depthVisibleWideFovURL;

    uint hiddenLayers;

    // visible layer for meshwarp
    VideoTexture visibleTexture;
    VideoTexture visibleTextureWideFOV;
    BC4DepthVideoTexture depthTexture;
    BC4DepthVideoTexture depthTextureWideFOV;

    bool sync = true;
    Pose colorFramePose, depthFramePose;
    pose_id_t poseIdColor = -1, poseIdDepth = -1;
    
    HybridReceiver(
        const glm::uvec2& remoteGBufferSize, 
        uint depthFactor,
        uint vertexGroupSize,
        QuadSet& quadSet, 
        uint hiddenLayers, 
        const std::string& videoAtlasURL, 
        const std::string& proxiesURL,
        const std::string& videoVisibleURL,
        const std::string& depthVisibleURL,
        const std::string& videoVisibleWideFovURL,
        const std::string& depthVisibleWideFovURL);
    ~HybridReceiver() = default;

    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }

    Mesh& getVisibleMesh() { return visibleMesh; }
    Mesh& getVisibleMeshWideFOV() { return visibleMeshWideFOV; }

    void recvData(const PoseStreamer& poseStreamer,
                  double& elapsedTimeColor, 
                  double& elapsedTimeDepth);

    void recvData();

    void updateMesh(bool isWideFOV);

private:
    uint depthFactor;
    uint vertexGroupSize;
    glm::uvec2 adjustedSize;

    // PerspectiveCamera remoteCamera;
    // PerspectiveCamera remoteCameraWideFOV;
    
    // Meshwarp shader for depth peeling
    Mesh visibleMesh;
    Mesh visibleMeshWideFOV;
    UnlitMaterial visibleMeshMaterial;
    UnlitMaterial visibleMeshWideFOVMaterial;

    ComputeShader meshFromBC4Shader;
    ComputeShader meshWarpReconstructShader;

};

} // namespace quasar

#endif // HYBRID_RECEIVER_H