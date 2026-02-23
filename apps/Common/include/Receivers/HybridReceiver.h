#ifndef HYBRID_RECEIVER_H
#define HYBRID_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <Path.h>
#include <thread>

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

    struct TimeStats {
        double memoryTransferTimeMs = 0.0;
        double meshwarpReconstructVisibleTimeMs = 0.0;
        double meshwarpReconstructWideFovTimeMs = 0.0;
        std::vector<double> decompressTimeMsByLayer;
        std::vector<double> depthPeelingTimeMsByLayer;
        double totalTimeMs = 0.0;
    };

    std::string videoVisibleURL;
    std::string videoVisibleWideFovURL;
    std::string depthVisibleURL;
    std::string depthVisibleWideFovURL;

    uint hiddenLayers;

    std::mutex useBackupMeshMutex;
    bool useBackupMesh = false;

    // CSV File for stats
    std::ofstream statsCSVFile;
    std::string statsCSVFileName;
    int frameID = 0;

    // TextureMeshCreateParams for the textures
    TextureDataCreateParams colorTextureCreateParams;
    TextureDataCreateParams depthTextureCreateParams;

    // visible layer for meshwarp
    VideoTexture visibleTexture;
    VideoTexture visibleTextureWideFOV;
    BC4DepthVideoTexture depthTexture;
    BC4DepthVideoTexture depthTextureWideFOV;

    Texture atlasTextureBackup;
    Texture alphaAtlasTextureBackup;

    const PoseStreamer& poseStreamer;
    double& elapsedTimeColor;
    double& elapsedTimeDepth;

    bool sync = true;
    Pose colorFramePose, depthFramePose;
    pose_id_t poseIdColor = -1, poseIdDepth = -1;
    
    HybridReceiver(
        const glm::uvec2& remoteGBufferSize, 
        uint depthFactor,
        uint vertexGroupSize,
        QuadSet& quadSet, 
        uint hiddenLayers, 
        PoseStreamer& poseStreamer,
        double& elapsedTimeColor, 
        double& elapsedTimeDepth,
        const std::string& videoAtlasURL, 
        const std::string& proxiesURL,
        const std::string& videoVisibleURL,
        const std::string& depthVisibleURL,
        const std::string& videoVisibleWideFovURL,
        const std::string& depthVisibleWideFovURL,
        GLFWwindow* window);
    ~HybridReceiver();

    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }

    Mesh& getVisibleMesh() { return visibleMesh; }
    Mesh& getVisibleMeshWideFOV() { return visibleMeshWideFOV; }
    Mesh& getVisibleMeshBackground() { return visibleMeshBackground; }
    Mesh& getVisibleMeshWideFOVBackground() { return visibleMeshWideFOVBackground; }
    QuadMesh& getMeshBackground(int layer) { return meshesBackground[layer]; }

    void recvData(const PoseStreamer& poseStreamer,
                  double& elapsedTimeColor, 
                  double& elapsedTimeDepth);

    void recvData(GLFWwindow* window);

    void updateMesh(bool isWideFOV, bool useBackgroundMesh);

    QuadFrame::FrameType loadFromMemory(const std::vector<char>& inputData) override;
    struct TimeStats reconstructHiddenLayers(std::shared_ptr<Frame> frame, bool useBackgroundMesh);


private:
    static constexpr size_t kMaxPendingFrames = 3;

    std::thread worker;
    std::condition_variable queueCv;
    // if we find out that the video is behind, we cannot proceed with the frame, we simply return and check again later
    // 
    std::atomic<bool> stopWorker{false};
    std::atomic<bool> workerRunning{false};

    uint depthFactor;
    uint vertexGroupSize;
    glm::uvec2 adjustedSize;
    
    // Meshwarp shader for depth peeling
    Mesh visibleMesh;
    Mesh visibleMeshWideFOV;
    // Mesh for background processing
    Mesh visibleMeshBackground;
    Mesh visibleMeshWideFOVBackground;
    std::vector<QuadMesh> meshesBackground;
    
    Texture visibleFrameTextureBackground;
    Texture visibleFrameTextureWideFOVBackground;
    BC4DepthVideoTexture depthTextureBackground;
    BC4DepthVideoTexture depthTextureWideFOVBackground;

    UnlitMaterial visibleMeshMaterial;
    UnlitMaterial visibleMeshWideFOVMaterial;

    UnlitMaterial visibleMeshBackgroundMaterial;
    UnlitMaterial visibleMeshWideFOVBackgroundMaterial;

    ComputeShader meshFromBC4Shader;
    ComputeShader meshWarpReconstructShader;

};

} // namespace quasar

#endif // HYBRID_RECEIVER_H