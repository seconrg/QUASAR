#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/QUASARReceiver.h>
#include <Receivers/PoseReceiver.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <Codecs/AlphaCodec.h>
#include <Primitives/FullScreenQuad.h>
#include <Primitives/Node.h>
#include <Quads/QuadTexelUsageMaterial.h>
#include <Scene.h>
#include <Shaders/Shader.h>

namespace quasar {

enum class WideFovMaskMethod {
    None,
    Stencil,
    RecordedTexelUsage,
};

inline const char* wideFovMaskMethodToString(WideFovMaskMethod method) {
    switch (method) {
        case WideFovMaskMethod::None:
            return "none";
        case WideFovMaskMethod::Stencil:
            return "stencil";
        case WideFovMaskMethod::RecordedTexelUsage:
            return "recorded-texel-usage";
    }
    return "none";
}

inline bool tryParseWideFovMaskMethod(std::string_view value, WideFovMaskMethod& outMethod) {
    if (value == "none" || value == "off" || value == "disabled") {
        outMethod = WideFovMaskMethod::None;
        return true;
    }
    if (value == "stencil" || value == "trim" || value == "trim-stencil") {
        outMethod = WideFovMaskMethod::Stencil;
        return true;
    }
    if (value == "recorded-texel-usage" || value == "recorded_texel_usage"
        || value == "texel-usage" || value == "mask-gt")
    {
        outMethod = WideFovMaskMethod::RecordedTexelUsage;
        return true;
    }
    return false;
}

struct QUASARStreamerCreateParams {
    uint maxLayers = 5;
    float viewSphereDiameter = 1.0f;
    float wideFOV = 140.0f;
    /// Wide-FOV layer (layer == maxLayers - 1) uses the camera view from this many frames ago (0 = latest).
    uint wideFovPoseLagFrames = 0;
    /// Wide-FOV layer: full render, tonemap, and quad generation only when frameID % wideFovUpdatePeriodFrames == 0 (and once while the wide layer has no quads yet). Other frames reuse the last wide-FOV textures and proxies. Use 1 for every frame. If this is greater than 1, \p trimWideFov is forced off in the streamer ctor.
    uint wideFovUpdatePeriodFrames = 1;
    uint targetFramerate = 5;
    uint targetBitRate = 28;
    std::string videoURL = "";
    std::string proxiesURL = "";
    /// If non-empty, write wide-FOV tonemapped color PNGs to this folder each frame (`widefov_<frameID>.png`).
    std::string wideFovImageDumpDir;
    /// Selects how the wide-FOV layer should be masked: none, stencil trim, or recorded texel-usage masks.
    WideFovMaskMethod wideFovMaskMethod = WideFovMaskMethod::None;
    /// Legacy compatibility knob for stencil trim. Ignored when \p wideFovMaskMethod is explicitly set by the caller.
    bool trimWideFov = false;
    /// If true, \p generateFrame only honors \p wideFovGroundTruthView when it is non-null (client/true pose).
    bool useWideFovGroundTruth = false;
    /// If non-empty, write per-frame corner-depth datasets here.
    std::string datasetOutputDir;
};

class QUASARStreamer : public DataStreamerTCP {
public:
    uint maxLayers;
    float viewSphereDiameter;
    uint wideFovPoseLagFrames;
    uint wideFovUpdatePeriodFrames;
    WideFovMaskMethod wideFovMaskMethod;
    bool trimWideFov;
    bool useWideFovGroundTruth;

    // Reference frame
    FrameRenderTarget referenceFrameRT;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodes;
    int meshIndex = 0, lastMeshIndex = 1;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    ResidualFrame residualFrame;
    QuadMesh residualFrameMesh;
    Node residualFrameNode;

    VideoStreamer videoAtlasStreamerRT;
    FrameRenderTarget alphaAtlasRT;

    // Local objects
    std::vector<Node> referenceFrameWireframesLocal;
    std::vector<Node> referenceFrameNodesLocal;
    Node residualFrameNodeLocal;
    Node residualFrameWireframeLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // Depth point cloud for debugging
    DepthMesh depthMesh;
    std::vector<DepthMesh> depthMeshesHidLayer;
    Node depthNode;
    std::vector<Node> depthNodesHidLayer;

    std::string videoURL;
    std::string proxiesURL;
    std::string wideFovImageDumpDir;
    std::string datasetOutputDir;

    struct Stats {
        double totalRenderTimeMs = 0.0;
        double totalCreateProxiesTimeMs = 0.0;
        double totalGenQuadMapTimeMs = 0.0;
        double totalSimplifyTimeMs = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTimeMs = 0.0;
        double totalAppendQuadsTimeMs = 0.0;
        double totalCreateVertIndTimeMs = 0.0;
        double totalGenDepthTimeMs = 0.0;
        double totalCompressTimeMs = 0.0;
        double frameSize = 0.0;

        std::vector<double> createProxiesTimeMsByLayer;
        std::vector<double> compressTimeMsByLayer;
        std::vector<double> createMeshTimeMsByLayer;
        QuadSet::Sizes proxySizes;
        /// Shoelace area of the reprojected narrow-view quad in wide-FOV pixels^2 (last wide-FOV frame; debug).
        double wideFovNarrowReprojectionQuadAreaPx = 0.0;
    } stats;

    struct bandwidthStats {
        std::vector<size_t> proxy_size_by_layer;
        std::vector<size_t> depth_offset_size_by_layer;
        size_t alphaSize = 0;
        size_t totalSize = 0;
    };
    struct bandwidthStats bandwidthstats;

    // log out stats to CSV file
    std::ofstream quasarStatsCSVFile;
    std::string quasarStatsCSVFileName;

    std::ofstream bandwidthStatsCSVFile;
    std::string bandwidthStatsCSVFileName;
    std::ofstream cornerDepthDatasetCSVFile;
    std::string cornerDepthDatasetCSVFileName;
    double prevSendTimeMs = 0.0;

    int frameID = 0;

    QUASARStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QUASARStreamerCreateParams& params = {});
    ~QUASARStreamer();

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);
    void setViewSphereDiameter(float viewSphereDiameter);

    /// \param wideFovGroundTruthView When \p useWideFovGroundTruth is true and this is non-null, wide-FOV reprojection uses it as the current view (e.g. client pose) instead of \p remoteCamera.
    RenderStats generateFrame(
        bool createResidualFrame = false,
        bool showNormals = false,
        bool showDepth = false,
        const glm::mat4* wideFovGroundTruthView = nullptr);
    void sendFrame(PoseReceiver::PoseInfo poseInfo, bool createResidualFrame);

    void setDrawState(QuadMesh::DrawState drawState);

    void writeTexturesToFiles(const Path& outputPath);
    size_t writeToFiles(const Path& outputPath);
    size_t writeToMemory(PoseReceiver::PoseInfo poseInfo, bool writeResidualFrame, std::vector<char>& outputData);

    /// Non-null when there is a wide-FOV hidden layer; material used for local quad proxies + texel-usage tracking.
    QuadTexelUsageMaterial* getWideFovQuadTexelUsageMaterial() { return wideFovQuadTexelUsageMaterial.get(); }
    const QuadTexelUsageMaterial* getWideFovQuadTexelUsageMaterial() const { return wideFovQuadTexelUsageMaterial.get(); }
    const FrameRenderTarget* getWideFovHiddenLayerNoToneRT() const {
        return frameRTsHidLayer_noTone.empty() ? nullptr : &frameRTsHidLayer_noTone.back();
    }
    const glm::vec3* getNormalViewCornersInWideFovImage() const { return normalViewCornersInWideFoVImage; }

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

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DepthPeelingRenderer& remoteRendererDP;
    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;
    PerspectiveCamera remoteCameraPrev;
    PerspectiveCamera remoteCameraWideFOV;

    std::deque<glm::mat4> wideFovCameraViewHistory;
    glm::mat4 wideFovReuseViewMatrix{1.0f};

    // Wide fov
    std::vector<Node> wideFovNodes;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    FrameRenderTarget referenceFrameRT_noTone;
    FrameRenderTarget residualFrameRT_noTone;
    std::vector<FrameRenderTarget> frameRTsHidLayer_noTone;

    std::vector<unsigned char> alphaImageData;

    std::vector<char> cameraData;
    std::vector<char> alphaData;
    std::vector<std::vector<char>> geometryMetadatas;
    std::vector<char> compressedData;

    AlphaCodec alphaCodec;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    Tonemapper tonemapper;
    ShowNormalsEffect showNormalsEffect;

    Shader quadMaskShader;
    Shader atwDebugShader;
    FullScreenQuad quadMaskQuad;
    FrameRenderTarget debugMaskRT;

    glm::vec3 normalViewCornersInWideFoVImage[4];

    /// Wide-FOV hidden-layer quad proxies: QuadTexelUsageMaterial (quad layout; same usage image semantics as TexelUsageMaterial).
    std::unique_ptr<QuadTexelUsageMaterial> wideFovQuadTexelUsageMaterial;

    void computeReprojectedNarrowCornersInWideFov(glm::vec3 outCorners[4]) const;
    void drawNarrowFovReprojectionOverlayOnWideRt(FrameRenderTarget& rt, const glm::vec3 cornersInWidePx[4]) const;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
