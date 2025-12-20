#ifndef HYBRID_STREAMER_H
#define HYBRID_STREAMER_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/QUASARReceiver.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <Codecs/AlphaCodec.h>


namespace quasar {

struct HybridStreamerCreateParams {
    // params for meshwarp
    uint depthFactor = 1;
    uint vertexGroupSize = 1;

    // params for depth peeling hidden layers
    uint hiddenLayers = 3;
    float viewSphereDiameter = 1.0f;
    float wideFOV = 140.0f;

    // Common params
    uint maxFrameRate = 30;
    uint targetBitRate = 12;
    std::string videoURL = "";
    std::string depthAndProxiesURL = "";

};

class HybridStreamer : public DataStreamerTCP {

public:
    uint hiddenLayers;
    float viewSphereDiameter;

    VideoStreamer videoAtlasStreamerRT;
    FrameRenderTarget alphaAtlasRT;

    // Hidden Layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // URLs to stream to and from
    std::string videoURL;
    std::string depthAndProxiesURL; // for this URL we are not only streaming proxy, but also depth
    
    struct stats {
        vector<double> renderTimeMsByLayer;
        vector<double> createProxiesTimeMsByLayer;
        vector<double> createMeshTimeMsByLayer;
        double totalCompressTimeMsByLayer;
        double frameSize = 0.0;
    }

    HybridStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer, 
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const HybridStreamerCreateParams& params = {});


    ~HybridStreamer() = default;

    RenderStats generateFrame();
    void sendFrame(pose_id_t poseID);
    
    size_t writeToFiles(const Path& outputPath);
    size_t writeToMemory(pose_id_t poseID, std::vector<char>& outputData);

private:
    DepthPeelingRenderer& remoteRendererDP;
    
    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;
    PerspectiveCamera& remoteCameraWideFOV;

    // Wide fov
    std::vector<Node> wideFovNodes;
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    std::vector<FrameRenderTarget> frameRTsHidLayer_noTone;

    Tonemapper tonemapper;
    ShowDepthEffect depthEffect;
    ShowNormalsEffect showNormalsEffect;

    ComputeShader meshFromBC4Shader;
    ComputeShader meshWarpReconstructShader;

    BC4DepthStreamer depthStreamerRT;
    BC4DepthStreamer depthStreamerWideFOV;

};

} // namespace quasar

#endif