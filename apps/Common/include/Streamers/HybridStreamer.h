#ifndef HYBRID_STREAMER_H
#define HYBRID_STREAMER_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/HybridReceiver.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Networking/DataStreamerTCP.h>

#include <Streamers/VideoStreamer.h>
#include <Streamers/BC4DepthStreamer.h>

#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <Codecs/AlphaCodec.h>


namespace quasar {

struct HybridStreamerCreateParams {

    // params for depth peeling hidden layers
    uint hiddenLayers = 3;
    float viewSphereDiameter = 1.0f;
    float wideFOV = 140.0f;

    // params for meshwarp
    uint depthFactor = 1;
    uint vertexGroupSize = 1;
    float depthRejectionThreshold = 0.05f;

    // Common params
    uint maxFrameRate = 30;
    uint targetBitRate = 28;
    std::string videoURL = "";
    std::string depthAndProxiesURL = "";

};

class HybridStreamer : public DataStreamerTCP {

public:
    // params for meshwarp
    float depthRejectionThreshold = 0.05f;
    
    // params for depth peeling hidden layers
    uint hiddenLayers = 3;
    float viewSphereDiameter = 1.0f;
    float wideFOV = 140.0f;

    VideoStreamer videoAtlasStreamerRT;
    FrameRenderTarget alphaAtlasRT;
    FrameRenderTarget frameRTVisible;
    FrameRenderTarget frameRTVisibleWideFov;

    struct Stats {
        double totalRenderTimeMs = 0.0;
        double totalGenMeshTime = 0.0;
        double totalCompressTimeMs = 0.0;
        size_t compressedSize = 0;
    } stats;

    // Hidden Layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<FrameRenderTarget> frameRTsHidLayer_noTone;

    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // URLs to stream to and from
    std::string videoURL;
    std::string depthAndProxiesURL; // for this URL we are not only streaming proxy, but also depth
    
    struct stats {
        std::vector<double> renderTimeMsByLayer;
        std::vector<double> createProxiesTimeMsByLayer;
        std::vector<double> createMeshTimeMsByLayer;
        double totalCompressTimeMsByLayer;
        double frameSize = 0.0;
    };

    HybridStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer, 
        Scene& remoteScene,
        Scene& localScene,
        PerspectiveCamera& remoteCamera,
        const HybridStreamerCreateParams& params = {});


    ~HybridStreamer() = default;

    RenderStats generateFrame();
    void sendFrame(pose_id_t poseID);
    
    size_t writeToFiles(const Path& outputPath);
    size_t writeToMemory(pose_id_t poseID, std::vector<char>& outputData);

    void setViewSphereDiameter(float viewSphereDiameter);

    void addMeshesToScene(Scene& localScene);

    void setDrawState(QuadMesh::DrawState drawState);

    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    Mesh& getVisibleMesh() { return visibleMesh; }
    Mesh& getVisibleMeshWideFOV() { return visibleMeshWideFOV; }

private:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary layer color is yellow
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
    };

    glm::uvec2 adjustedSize;
    glm::uvec2 depthMapSize;

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DepthPeelingRenderer& remoteRendererDP;
    DeferredRenderer& remoteRenderer;

    Scene& remoteScene;
    Scene& localScene;
    
    PerspectiveCamera& remoteCamera;
    PerspectiveCamera remoteCameraWideFOV;

    // Wide fov
    Node wideFovNode;
    Scene wideFovScene;

    AlphaCodec alphaCodec;

    std::vector<ReferenceFrame> referenceFrames;

    Tonemapper tonemapper;
    ShowDepthEffect depthEffect;
    ShowNormalsEffect showNormalsEffect;

    BC4DepthStreamer depthStreamerRT;
    BC4DepthStreamer depthStreamerWideFOV;

    /* Information used for local debugging and simulation*/
    ComputeShader meshFromBC4Shader;
    ComputeShader meshWarpReconstructShader;

    // RenderTarget renderTarget;
    // RenderTarget renderTargetWideFOV;

    Mesh visibleMesh;
    Mesh visibleMeshWideFOV;
    UnlitMaterial visibleMeshMaterial;
    UnlitMaterial visibleMeshWideFOVMaterial;

    // visible and wide fov nodes
    Node visibleMeshNode;
    Node visibleMeshWideFOVNode;

    void reconstructMeshwarp(PerspectiveCamera &camera, Mesh &mesh, BC4DepthStreamer &depthStreamer);

};

} // namespace quasar

#endif