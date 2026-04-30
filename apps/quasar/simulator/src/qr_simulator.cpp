#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>
#include <Renderers/DeferredRenderer.h>
#include <Renderers/DepthPeelingRenderer.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>
#include <UI/RecordWindow.h>
#include <UI/TexturePreviewWindow.h>
#include <UI/SceneWindow.h>

#include <Recorder.h>
#include <CameraAnimator.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <Streamers/QUASARStreamer.h>
#include <HoleFiller.h>

#include <Path.h>
#include <PoseSendRecvSimulator.h>
#include <Utils/TimeUtils.h>

#include <glad/glad.h>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace quasar;

namespace {

constexpr const char* kWideFovGroundTruthPoseRecordsPath =
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/profileCode/imageQualityCorrelation/robot_lab/"
    "quasarTrimWithRealRender_prev_curr_pose_records.json";
constexpr const char* kWideFovGroundTruthPoseRecordsFilename =
    "prev_curr_pose_records.json";
constexpr const char* kLegacyWideFovGroundTruthPoseRecordsFilename =
    "quasarTrimWithRealRender_prev_curr_pose_records.json";

std::string resolveWideFovGroundTruthPoseRecordsPath(const std::string& wideFovGroundTruthDir) {
    if (wideFovGroundTruthDir.empty()) {
        return std::string(kWideFovGroundTruthPoseRecordsPath);
    }

    const Path groundTruthDir(wideFovGroundTruthDir);
    const std::vector<Path> candidatePaths = {
        groundTruthDir / kWideFovGroundTruthPoseRecordsFilename,
        groundTruthDir / "loggedInfo" / kWideFovGroundTruthPoseRecordsFilename,
        groundTruthDir / kLegacyWideFovGroundTruthPoseRecordsFilename,
        groundTruthDir / "loggedInfo" / kLegacyWideFovGroundTruthPoseRecordsFilename,
    };

    for (const Path& candidatePath : candidatePaths) {
        if (candidatePath.exists()) {
            return candidatePath.str();
        }
    }

    return candidatePaths.front().str();
}

struct WideFovGroundTruthPoseRecord {
    int recordedFrameID = -1;
    glm::vec3 position{0.0f};
    glm::vec3 eulerRotationDegrees{0.0f};
};

using WideFovGroundTruthPoseRecordMap = std::unordered_map<int, WideFovGroundTruthPoseRecord>;

std::optional<WideFovGroundTruthPoseRecordMap> loadWideFovGroundTruthPoseRecords(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        spdlog::warn("Failed to open wide-FOV ground-truth pose record file: {}", jsonPath);
        return std::nullopt;
    }

    nlohmann::json root;
    try {
        file >> root;
    }
    catch (const std::exception& e) {
        spdlog::warn("Failed to parse wide-FOV ground-truth pose record file {}: {}", jsonPath, e.what());
        return std::nullopt;
    }

    const auto remoteRenderFramesIt = root.find("remote_render_frames");
    if (remoteRenderFramesIt == root.end() || !remoteRenderFramesIt->is_object()) {
        spdlog::warn("Wide-FOV ground-truth pose record file {} is missing remote_render_frames", jsonPath);
        return std::nullopt;
    }

    WideFovGroundTruthPoseRecordMap recordsByRemoteFrameID;
    for (const auto& [remoteFrameIDText, remoteFrameEntry] : remoteRenderFramesIt->items()) {
        if (!remoteFrameEntry.is_object()) {
            continue;
        }

        const auto recordedFramesIt = remoteFrameEntry.find("Recorded Frame");
        if (recordedFramesIt == remoteFrameEntry.end() || !recordedFramesIt->is_object() || recordedFramesIt->empty()) {
            continue;
        }

        int bestRecordedFrameID = -1;
        const nlohmann::json* bestRecordedFramePose = nullptr;
        for (const auto& [recordedFrameIDText, recordedFramePose] : recordedFramesIt->items()) {
            try {
                const int recordedFrameID = std::stoi(recordedFrameIDText);
                if (recordedFrameID > bestRecordedFrameID) {
                    bestRecordedFrameID = recordedFrameID;
                    bestRecordedFramePose = &recordedFramePose;
                }
            }
            catch (const std::exception&) {
                continue;
            }
        }

        if (bestRecordedFramePose == nullptr || bestRecordedFrameID < 0) {
            continue;
        }

        try {
            const auto& position = bestRecordedFramePose->at("position");
            const auto& rotation = bestRecordedFramePose->at("euler_rotation_degrees");
            WideFovGroundTruthPoseRecord record;
            record.recordedFrameID = bestRecordedFrameID;
            record.position = glm::vec3(
                position.at("x").get<float>(),
                position.at("y").get<float>(),
                position.at("z").get<float>());
            record.eulerRotationDegrees = glm::vec3(
                rotation.at("x").get<float>(),
                rotation.at("y").get<float>(),
                rotation.at("z").get<float>());
            recordsByRemoteFrameID.emplace(std::stoi(remoteFrameIDText), record);
        }
        catch (const std::exception& e) {
            spdlog::warn(
                "Skipping malformed wide-FOV ground-truth pose record for remote frame {} in {}: {}",
                remoteFrameIDText,
                jsonPath,
                e.what());
        }
    }

    spdlog::info(
        "Loaded {} remote-frame -> recorded-frame wide-FOV ground-truth pose mappings from {}",
        recordsByRemoteFrameID.size(),
        jsonPath);
    return recordsByRemoteFrameID;
}

glm::mat4 buildViewMatrixFromRecordedPose(
    const glm::vec3& position,
    const glm::vec3& eulerRotationDegrees,
    const glm::mat4& projectionMatrix)
{
    PerspectiveCamera poseCamera(projectionMatrix);
    poseCamera.setPosition(position);
    poseCamera.setRotationEuler(eulerRotationDegrees);
    poseCamera.updateViewMatrix();
    return poseCamera.getViewMatrix();
}

struct PoseCsvInfo {
    glm::vec3 position{0.0f};
    glm::quat rotationQuat{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 eulerRotationDegrees{0.0f};
};

PoseCsvInfo extractPoseCsvInfoFromViewMatrix(const glm::mat4& viewMatrix) {
    PoseCsvInfo info;
    glm::vec3 scale;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(viewMatrix), scale, info.rotationQuat, info.position, skew, perspective);
    info.rotationQuat = glm::normalize(info.rotationQuat);
    info.eulerRotationDegrees = glm::degrees(glm::eulerAngles(info.rotationQuat));
    return info;
}

float maxAbsMatrixElementDiff(const glm::mat4& a, const glm::mat4& b) {
    float maxDiff = 0.0f;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            maxDiff = std::max(maxDiff, std::abs(a[col][row] - b[col][row]));
        }
    }
    return maxDiff;
}

double normalizeWideFovUncertaintyConfidence(double confidence) {
    if (confidence > 1.0 && confidence <= 100.0) {
        confidence /= 100.0;
    }

    if (std::isfinite(confidence) && confidence > 0.0 && confidence < 1.0) {
        return confidence;
    }

    spdlog::warn(
        "Invalid wide-FOV reprojection uncertainty confidence {}; using default confidence 0.68",
        confidence);
    return 0.68;
}

} // namespace

int main(int argc, char** argv) {
    Config config{};
    config.title = "QUASAR Simulator";
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of local renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'N', "num-poses"}, -1);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::Flag posePredictionIn(parser, "pose-prediction", "Enable pose prediction", {'P', "pose-prediction"}, false);
    args::Flag poseSmoothingIn(parser, "pose-smoothing", "Enable pose smoothing", {'T', "pose-smoothing"}, false);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 140.0f);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
    args::ValueFlag<int> wideFovPoseLagFramesIn(parser, "wide-fov-pose-lag", "Wide-FOV layer renders with camera view from this many frames ago", {'L', "wide-fov-pose-lag"}, 0);
    args::ValueFlag<int> wideFovUpdatePeriodIn(parser, "wide-fov-update-period", "Regenerate wide-FOV layer only when frameID mod N == 0 (1 = every frame)", {'K', "wide-fov-update-period"}, 1);
    args::ValueFlag<std::string> wideFovDumpDirIn(parser, "path", "Dump wide-FOV tonemapped PNG per frame to this folder (empty = off)", {"wide-fov-dump-dir"}, "");
    args::Flag noWideFovDumpIn(
        parser,
        "no-wide-fov-dump",
        "Disable the default output-path/widefov_dump PNG dump when saving images",
        {"no-wide-fov-dump"});
    args::ValueFlag<std::string> wideFovClientColorTextureDumpDirIn(
        parser,
        "path",
        "After each local drawObjects, dump wide-FOV layer colorTexture as PNG here (empty = off)",
        {"dump-wide-fov-color-texture-after-draw"},
        "");
    args::ValueFlag<std::string> wideFovClientTexelUsageDumpDirIn(
        parser,
        "path",
        "Dump wide-FOV texel-usage PNG for recorded client frames; "
        "filename widefov_texel_usage_<frameID>.png matches frame_%06d.png (empty = off)",
        {"dump-wide-fov-texel-usage-after-draw"},
        "");
    args::Flag dumpWideFovClientTexelUsageIn(
        parser,
        "dump-wide-fov-texel-usage",
        "Enable dumping wide-FOV texel-usage PNGs to the default output directory",
        {"dump-wide-fov-texel-usage"});
    args::Flag showWideFovNarrowOverlayIn(
        parser,
        "show-wide-fov-narrow-overlay",
        "Tint red the narrow-FOV footprint reprojected into the wide-FOV target (after tonemap)",
        {"show-wide-fov-narrow-overlay"});
    args::Flag trimWideFovIn(
        parser,
        "trim-wide-fov",
        "Legacy shortcut for --wide-fov-mask-method stencil",
        {"trim-wide-fov"});
    args::ValueFlag<std::string> wideFovMaskMethodIn(
        parser,
        "method",
        "Wide-FOV mask method: none, stencil, stencilwithgtpose, or recorded-texel-usage",
        {"wide-fov-mask-method"},
        "");
    args::Flag useWideFovGroundTruthIn(
        parser,
        "use-wide-fov-ground-truth",
        "Wide-FOV reprojection uses local (client) view as current pose instead of predicted remote pose",
        {"use-wide-fov-ground-truth"});
    args::ValueFlag<std::string> wideFovGroundTruthDirIn(
        parser,
        "dir",
        "Directory containing wide-FOV ground-truth metadata files",
        {"wide-fov-ground-truth-dir"},
        "");
    args::ValueFlag<std::string> wideFovGroundTruthPoseRecordsIn(
        parser,
        "json",
        "Path to wide-FOV ground-truth pose records JSON; overrides --wide-fov-ground-truth-dir",
        {"wide-fov-ground-truth-pose-records"},
        "");
    args::ValueFlag<std::string> wideFovTexelUsageMaskDirIn(
        parser,
        "dir",
        "Directory containing recorded wide-FOV texel-usage PNG masks",
        {"wide-fov-texel-usage-mask-dir"},
        "");
    args::Flag wideFovReprojectionUncertaintyIn(
        parser,
        "wide-fov-reprojection-uncertainty",
        "Use analytic J Sigma J^T reprojection uncertainty for the wide-FOV stencil region",
        {"wide-fov-reprojection-uncertainty"});
    args::ValueFlag<int> wideFovCovarianceSamplesIn(
        parser,
        "count",
        "Legacy/debug: request covariance handling; pose sampling only runs with --wide-fov-covariance-debug-sampling",
        {"wide-fov-covariance-samples"},
        0);
    args::Flag wideFovCovarianceDebugSamplingIn(
        parser,
        "wide-fov-covariance-debug-sampling",
        "Also sample poses from predictPose covariance and union them into the wide-FOV stencil for debug/validation",
        {"wide-fov-covariance-debug-sampling"});
    args::ValueFlag<double> wideFovCovarianceConfidenceIn(
        parser,
        "confidence",
        "Confidence level for wide-FOV reprojection uncertainty, as fraction or percentage",
        {"wide-fov-covariance-confidence"},
        0.68);
    args::ValueFlag<float> wideFovUncertaintyMinRadiusIn(
        parser,
        "pixels",
        "Minimum analytic reprojection uncertainty radius in pixels",
        {"wide-fov-uncertainty-min-radius"},
        0.0f);
    args::ValueFlag<float> wideFovUncertaintyMaxRadiusIn(
        parser,
        "pixels",
        "Maximum analytic reprojection uncertainty radius in pixels",
        {"wide-fov-uncertainty-max-radius"},
        512.0f);
    // args::ValueFlag<std::string> EIn(parser, "E", "Path to E's size for each depth peeling call", {'E', "E-path"}, "");
    
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    if (verbose) spdlog::set_level(spdlog::level::debug);

    // Parse size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    // Parse remote size
    std::string rsizeStr = args::get(resIn);
    pos = rsizeStr.find('x');
    glm::uvec2 remoteWindowSize = glm::uvec2(std::stoi(rsizeStr.substr(0, pos)), std::stoi(rsizeStr.substr(pos + 1)));

    config.enableVSync = !args::get(novsync) && !saveImages;
    config.showWindow = !args::get(saveImages);

    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);
    
    // // Read E path
    // Path EPathFile = args::get(EIn);
    // std::ifstream efile(EPathFile.c_str());
    // if (!efile.is_open()) {
    //     spdlog::error("Failed to open E path file: {}", EPathFile.str());
    //     return -1;
    // }

    // std::queue<float> Es;
    // std::string line;
    // int lineNumber = 0;
    // while(std::getline(efile, line)) {
    //     if (line.empty() || line[0] == '#' || lineNumber == 0) {
    //         lineNumber++;
    //         continue; // Skip empty lines and comments
    //     }
    //     // the delimiter is , so we can have float numbers like 0.5,1.0,1.5
    //     std::stringstream ss(line);
    //     std::string token;
    //     // We only need the second element of each line
    //     int counter = 0;
    //     while (std::getline(ss, token, ',')) {
    //         if (counter == 1) {
    //             float E = std::stof(token);
    //             Es.push(E);
    //             break;
    //         }
    //         counter++;
    //     }
    //     lineNumber++;
    // }
    // efile.close();

    // spdlog::info("Loaded {} E values from {}", Es.size(), EPathFile.str());


    int numPoses = args::get(numPosesIn);
    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path loggedInfoPath = outputPath / "loggedInfo"; loggedInfoPath.mkdirRecursive();

    uint maxHidLayers = args::get(maxHiddenLayersIn);
    uint maxLayers = maxHidLayers + 2;

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DepthPeelingRenderer remoteRendererDP(config, maxLayers - 1, true); // DP layers doesn't include wide fov
    DeferredRenderer remoteRenderer(config);

    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteRendererDP.width, remoteRendererDP.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    // "Local" scene
    Scene localScene;
    // localScene.skybox = remoteScene.skybox;
    localScene.skybox = nullptr;
    PerspectiveCamera camera(windowSize);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    QuadSet quadSet(remoteWindowSize);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewSphereDiameter = args::get(viewSphereDiameterIn);
    int wideFovPoseLagArg = args::get(wideFovPoseLagFramesIn);
    uint wideFovPoseLagFrames = static_cast<uint>(wideFovPoseLagArg < 0 ? 0 : wideFovPoseLagArg);


    spdlog::info("Remote FOV Wide: {}", remoteFOVWide);

    // hardcode wide fov pose lag frames to 1
    wideFovPoseLagFrames = 0;

    int wideFovUpdatePeriodArg = args::get(wideFovUpdatePeriodIn);
    uint wideFovUpdatePeriodFrames = static_cast<uint>(wideFovUpdatePeriodArg < 1 ? 1 : wideFovUpdatePeriodArg);
    WideFovMaskMethod wideFovMaskMethod = WideFovMaskMethod::None;
    const std::string wideFovMaskMethodValue = args::get(wideFovMaskMethodIn);
    if (!wideFovMaskMethodValue.empty()) {
        if (!tryParseWideFovMaskMethod(wideFovMaskMethodValue, wideFovMaskMethod)) {
            spdlog::error(
                "Invalid wide-FOV mask method '{}'. Expected one of: none, stencil, stencilwithgtpose, recorded-texel-usage",
                wideFovMaskMethodValue);
            return 1;
        }
    }
    else if (args::get(trimWideFovIn)) {
        wideFovMaskMethod = WideFovMaskMethod::Stencil;
    }
    bool trimWideFov = wideFovMaskMethod == WideFovMaskMethod::Stencil;
    const bool wideFovMaskMethodRequestsGroundTruth =
        wideFovMaskMethodValue == "stencilwithgtpose"
        || wideFovMaskMethodValue == "stencil-with-gt-pose"
        || wideFovMaskMethodValue == "stencil_gt_pose";
    bool useWideFovGroundTruth =
        args::get(useWideFovGroundTruthIn) || wideFovMaskMethodRequestsGroundTruth;
    const std::string wideFovGroundTruthDir = args::get(wideFovGroundTruthDirIn);
    std::string wideFovGroundTruthPoseRecordsPath = args::get(wideFovGroundTruthPoseRecordsIn);
    if (wideFovGroundTruthPoseRecordsPath.empty()) {
        wideFovGroundTruthPoseRecordsPath =
            resolveWideFovGroundTruthPoseRecordsPath(wideFovGroundTruthDir);
    }
    const std::string wideFovTexelUsageMaskDir = args::get(wideFovTexelUsageMaskDirIn);
    const bool requestedWideFovReprojectionUncertainty =
        args::get(wideFovReprojectionUncertaintyIn) || args::get(wideFovCovarianceSamplesIn) > 0;
    const bool requestedWideFovCovarianceDebugSampling = args::get(wideFovCovarianceDebugSamplingIn);
    const bool suppressWideFovCovarianceForGroundTruthStencil = wideFovMaskMethodRequestsGroundTruth;
    const bool wideFovReprojectionUncertaintyRequested =
        requestedWideFovReprojectionUncertainty && !suppressWideFovCovarianceForGroundTruthStencil;
    const bool wideFovCovarianceDebugSampling =
        requestedWideFovCovarianceDebugSampling && !suppressWideFovCovarianceForGroundTruthStencil;
    const int wideFovCovarianceSampleCountArg = args::get(wideFovCovarianceSamplesIn);
    const size_t wideFovCovarianceSampleCount =
        static_cast<size_t>(std::max(wideFovCovarianceSampleCountArg, 0));
    const double wideFovCovarianceConfidence =
        normalizeWideFovUncertaintyConfidence(args::get(wideFovCovarianceConfidenceIn));
    const float wideFovUncertaintyMinRadiusPx = std::max(0.0f, args::get(wideFovUncertaintyMinRadiusIn));
    const float wideFovUncertaintyMaxRadiusPx =
        std::max(wideFovUncertaintyMinRadiusPx, args::get(wideFovUncertaintyMaxRadiusIn));
    const auto wideFovGroundTruthPoseRecords =
        loadWideFovGroundTruthPoseRecords(wideFovGroundTruthPoseRecordsPath);

    spdlog::info("Wide FOV Update Period Frames: {}", wideFovUpdatePeriodFrames);
    spdlog::info("Trim Wide FOV: {}", trimWideFov ? "Enabled" : "Disabled");
    spdlog::info("Wide FOV Mask Method: {}", wideFovMaskMethodToString(wideFovMaskMethod));
    spdlog::info("Use Wide FOV Ground Truth: {}", useWideFovGroundTruth ? "Enabled" : "Disabled");
    if (useWideFovGroundTruth || !wideFovGroundTruthDir.empty()) {
        spdlog::info("Wide FOV Ground Truth Pose Records: {}", wideFovGroundTruthPoseRecordsPath);
    }
    if (!wideFovGroundTruthDir.empty()) {
        spdlog::info("Wide FOV Ground Truth Metadata Dir: {}", wideFovGroundTruthDir);
    }
    if (!wideFovTexelUsageMaskDir.empty()) {
        spdlog::info("Wide FOV Texel Usage Mask Dir: {}", wideFovTexelUsageMaskDir);
    }
    if (suppressWideFovCovarianceForGroundTruthStencil
        && (requestedWideFovReprojectionUncertainty || requestedWideFovCovarianceDebugSampling))
    {
        spdlog::info(
            "Wide-FOV covariance uncertainty/sampling is disabled for stencilwithgtpose "
            "so GT-pose stencil bitrate measurements are not affected");
    }
    if (wideFovReprojectionUncertaintyRequested) {
        spdlog::info(
            "Wide-FOV analytic reprojection uncertainty: confidence={:.1f}%, radius clamp=[{:.3f}, {:.3f}] px",
            wideFovCovarianceConfidence * 100.0,
            wideFovUncertaintyMinRadiusPx,
            wideFovUncertaintyMaxRadiusPx);
        if (!trimWideFov) {
            spdlog::warn("Wide-FOV reprojection uncertainty is only applied by the stencil trim path");
        }
        if (wideFovCovarianceSampleCount > 0 && !wideFovCovarianceDebugSampling) {
            spdlog::info(
                "--wide-fov-covariance-samples now enables analytic uncertainty only; "
                "add --wide-fov-covariance-debug-sampling to run Monte Carlo pose sampling");
        }
    }
    if (wideFovCovarianceDebugSampling && wideFovCovarianceSampleCount == 0) {
        spdlog::warn("--wide-fov-covariance-debug-sampling was set but --wide-fov-covariance-samples is 0");
    }

    //
    std::string wideFovDumpDir = args::get(wideFovDumpDirIn);
    if (args::get(noWideFovDumpIn)) {
        wideFovDumpDir.clear();
    } else if (wideFovDumpDir.empty()) {
        wideFovDumpDir = outputPath.str() + "/widefov_dump";
    }

    QUASARStreamer quasar(
        quadSet,
        remoteRendererDP, remoteRenderer, remoteScene, remoteCamera,
        {
            .maxLayers = static_cast<uint>(maxLayers),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteFOVWide,
            .wideFovPoseLagFrames = wideFovPoseLagFrames,
            .wideFovUpdatePeriodFrames = wideFovUpdatePeriodFrames,
            .wideFovImageDumpDir = wideFovDumpDir,
            .wideFovMaskMethod = wideFovMaskMethod,
            .trimWideFov = trimWideFov,
            .useWideFovGroundTruth = useWideFovGroundTruth,
            .wideFovGroundTruthDir = wideFovGroundTruthDir,
            .wideFovTexelUsageMaskDir = wideFovTexelUsageMaskDir,
            .datasetOutputDir = loggedInfoPath.str(),
        });

    quasar.addMeshesToScene(localScene);

    std::ofstream recvPoseToRenderFile((loggedInfoPath / "recv_pose_to_render.csv").str());
    recvPoseToRenderFile
        << "frame_id,recorded_frame_id,pose_timestamp_us,"
        << "tx,ty,tz,qx,qy,qz,qw,rx_deg,ry_deg,rz_deg" << std::endl;
    std::ofstream predictionPoseTimestampsFile((loggedInfoPath / "pose_prediction_timestamps.csv").str());
    predictionPoseTimestampsFile
        << "frame_id,recorded_frame_id,prediction_used,"
        << "latest_pose_timestamp_us,prev_pose_timestamp_us,prev_but_two_pose_timestamp_us,predicted_pose_timestamp_us"
        << std::endl;
    std::ofstream recordedFramePoseFile((loggedInfoPath / "recorded_frame_render_poses.csv").str());
    recordedFramePoseFile
        << "recorded_frame_id,pose_timestamp_us,render_now_us,"
        << "tx,ty,tz,qx,qy,qz,qw,rx_deg,ry_deg,rz_deg" << std::endl;
    std::ofstream covarianceSampledPosesFile((loggedInfoPath / "widefov_covariance_sampled_poses.csv").str());
    covarianceSampledPosesFile
        << "name,frame_id,recorded_frame_id,sample_id,confidence,pose_timestamp_us,"
        << "tx,ty,tz,qx,qy,qz,qw,rx_deg,ry_deg,rz_deg" << std::endl;
    

    // Post processing
    HoleFiller holeFiller;
    Tonemapper tonemapper;

    Recorder recorder({
        .width = windowSize.x,
        .height = windowSize.y,
        .internalFormat = GL_RGBA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, renderer, holeFiller, outputPath, config.targetFramerate);
    CameraAnimator cameraAnimator(cameraPathFile, numPoses);

    if (saveImages) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        recorder.start();
    }

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCamera);
    }

    bool showDepth = false;
    bool showNormals = false;
    bool showWireframe = false;
    bool hideReferenceFrame = false, hideResidualFrame = false;
    bool showResidualFrame = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;
    bool restrictMovementToViewSphere = !cameraPathFileIn;

    bool sendReferenceFrame = true;
    bool sendResidualFrame = false;
    int refFrameInterval = 5;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 60};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "60 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 30 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = posePredictionIn;
    spdlog::info("Pose Prediction: {}", posePrediction ? "Enabled" : "Disabled");
    spdlog::info("Pose Smoothing: {}", poseSmoothingIn ? "Enabled" : "Disabled");
    bool poseSmoothing = poseSmoothingIn;
    PoseSendRecvSimulator poseSendRecvSimulator({
        .networkLatencyMs = networkLatency,
        .networkJitterMs = networkJitter,
        .renderTimeMs = rerenderIntervalMs,
        .posePrediction = posePrediction,
        .poseSmoothing = poseSmoothing,
    });

    spdlog::info("Network Latency: {} ms", networkLatency);
    spdlog::info("Network Jitter: {} ms", networkJitter);
    spdlog::info("View Sphere Diameter: {} m", viewSphereDiameter);

    bool* showLayers = new bool[maxLayers];
    for (int i = 0; i < maxLayers; i++) {
        showLayers[i] = true;
    }

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", quasar.videoAtlasStreamerRT.colorTexture, ImVec2(860, 860));
    TexturePreviewWindow alphaPreviewWindow("Alpha Texture", quasar.alphaAtlasRT.alphaTexture, ImVec2(860, 860));
    TexturePreviewWindow refFramePreviewWindow("Reference Frame", quasar.referenceFrameRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameChangedPreviewWindow("Residual Frame (changed geometry)", quasar.residualFrameMaskRT.colorTexture, ImVec2(430, 270));
    TexturePreviewWindow resFrameFullPreviewWindow("Residual Frame (revealed geometry)", quasar.residualFrameRT.colorTexture, ImVec2(430, 270));
    std::unique_ptr<TexturePreviewWindow> wideFovTexelUsagePreview;
    if (quasar.getWideFovQuadTexelUsageMaterial() != nullptr) {
        wideFovTexelUsagePreview = std::make_unique<TexturePreviewWindow>(
            "Wide FOV texel usage",
            *quasar.getWideFovQuadTexelUsageMaterial()->getTexelUsageTexture(),
            ImVec2(430, 270));
        wideFovTexelUsagePreview->visible = true;
    }
    SceneWindow sceneWindowRemote(remoteScene, ImVec2(430, 800));
    SceneWindow sceneWindowLocal(localScene, ImVec2(430, 800));
    CameraHeader cameraHeader(camera);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImages;
        static bool showMeshCapture = false;
        static bool showFramePreviewWindows = false;
        static bool showLayerPreviews = false;
        static bool saveAsSeparate = true;

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "ESC")) {
                window->close();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("FPS", 0, &frameRateWindow.visible);
            ImGui::MenuItem("UI", 0, &showUI);
            ImGui::MenuItem("Frame Capture", 0, &frameCaptureWindow.visible);
            ImGui::MenuItem("Record", 0, &recordWindow.visible);
            ImGui::MenuItem("Mesh Capture", 0, &showMeshCapture);
            ImGui::MenuItem("Frame Previews", 0, &showFramePreviewWindows);
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::MenuItem("Video Preview", 0, &videoPreviewWindow.visible);
            ImGui::MenuItem("Alpha Preview", 0, &alphaPreviewWindow.visible);
            if (wideFovTexelUsagePreview) {
                ImGui::MenuItem("Wide FOV texel usage", 0, &wideFovTexelUsagePreview->visible);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Remote Scene", 0, &sceneWindowRemote.visible);
            ImGui::MenuItem("Local Scene", 0, &sceneWindowLocal.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindowRemote.draw(now, dt);
        sceneWindowLocal.draw(now, dt);
        videoPreviewWindow.draw(now, dt);
        alphaPreviewWindow.draw(now, dt);
        if (wideFovTexelUsagePreview) {
            wideFovTexelUsagePreview->draw(now, dt);
        }

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            size_t totalTriangles = quasar.getNumTriangles();
            if (totalTriangles < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", totalTriangles);
            else if (totalTriangles < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", totalTriangles);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", totalTriangles);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               quasar.stats.proxySizes.numQuads,
                               quasar.stats.proxySizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               quasar.stats.proxySizes.numDepthOffsets,
                               quasar.stats.proxySizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            if (ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
            }
            if (ImGui::Checkbox("Show Normals Instead of Color", &showNormals)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
            }
            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Hide Reference Frame", &hideReferenceFrame); ImGui::SameLine();
            ImGui::Checkbox("Hide Residual Frame", &hideResidualFrame);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = quasar.getQuadsGenerator();
                if (ImGui::Checkbox("Expand Proxies", &quadsGenerator->params.expandProxies)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f")) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragFloat("Plane Similarity Threshold", &quadsGenerator->params.planeSimilarityThreshold, 0.001f, 0.0f, 5.0f)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
                if (ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 1, 0, quadsGenerator->numQuadMaps)) {
                    preventCopyingLocalPose = true;
                    sendReferenceFrame = true;
                    runAnimations = false;
                }
            }

            ImGui::Separator();

            if (ImGui::DragFloat("Network Latency (ms)", &networkLatency, 0.5f, 0.0f, 500.0f)) {
                poseSendRecvSimulator.setNetworkLatency(networkLatency);
            }
            if (ImGui::DragFloat("Network Jitter (ms)", &networkJitter, 0.25f, 0.0f, 50.0f)) {
                poseSendRecvSimulator.setNetworkJitter(networkJitter);
            }

            ImGui::Checkbox("Pose Prediction Enabled", &poseSendRecvSimulator.posePrediction);

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
                runAnimations = true;
            }

            float windowWidth = ImGui::GetContentRegionAvail().x;
            float buttonWidth = (windowWidth - ImGui::GetStyle().ItemSpacing.x) / 2.0f;
            if (ImGui::Button("Send Reference Frame", ImVec2(buttonWidth, 0))) {
                sendReferenceFrame = true;
                runAnimations = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Send Residual Frame", ImVec2(buttonWidth, 0))) {
                sendResidualFrame = true;
                runAnimations = true;
            }
            ImGui::DragInt("Ref Frame Interval", &refFrameInterval, 0.1, 1, 5);

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.025f, 0.1f, 2.0f)) {
                preventCopyingLocalPose = true;
                sendReferenceFrame = true;
                runAnimations = false;
                quasar.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Checkbox("Restrict Movement to View Sphere", &restrictMovementToViewSphere);

            ImGui::Separator();

            const int columns = 3;
            for (int layer = 0; layer < maxLayers; layer++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(layer)).c_str(), &showLayers[layer]);
                if ((layer + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }

        if (showMeshCapture) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCapture);

            ImGui::Checkbox("Save as Separate Files", &saveAsSeparate);
            if (ImGui::Button("Save Proxies")) {
                if (!saveAsSeparate) {
                    std::vector<char> compressedData;
                    spdlog::info("Saved {} bytes to {}", quasar.writeToMemory(PoseReceiver::PoseInfo{0, 0, 0}, sendResidualFrame, compressedData), outputPath.absolutePathStr());
                    Path filename = (outputPath / "frame").appendToName(".bin");
                    FileIO::writeToBinaryFile(filename, compressedData.data(), compressedData.size());
                    quasar.writeTexturesToFiles(outputPath);
                }
                else {
                    spdlog::info("Saved {} bytes to {}", quasar.writeToFiles(outputPath), outputPath.absolutePathStr());
                }
            }

            ImGui::End();
        }

        if (showFramePreviewWindows) {
            refFramePreviewWindow.visible = true; refFramePreviewWindow.draw(now, dt);
            resFrameChangedPreviewWindow.visible = true; resFrameChangedPreviewWindow.draw(now, dt);
            resFrameFullPreviewWindow.visible = true; resFrameFullPreviewWindow.draw(now, dt);
        }

        if (showLayerPreviews) {
            for (int layer = 0; layer < maxLayers; layer++) {
                int layerIdx = maxLayers - layer - 1;
                if (showLayers[layerIdx]) {
                    ImGui::Begin(("Layer " + std::to_string(layerIdx)).c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                    if (layerIdx == 0) {
                        ImGui::Image((void*)(intptr_t)(quasar.referenceFrameRT.colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    else {
                        ImGui::Image((void*)(intptr_t)(quasar.frameRTsHidLayer[layerIdx-1].colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    }
                    ImGui::End();
                }
            }
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        remoteRendererDP.setWindowSize(width, height);
        renderer.setWindowSize(width, height);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lastRenderTime = -INFINITY;
    bool updateClient = !saveImages;
    int frameCounter = 0;
    PoseSendRecvSimulator::PredictionDebugInfo activePredictionDebugInfo;
    PoseSendRecvSimulator::PredictionCovariance activePredictionCovariance;
    std::optional<Pose> activeRemoteRenderPose;
    std::vector<Pose> activeWideFovCovarianceSamples;

    std::string wideFovClientColorTextureDumpDir = args::get(wideFovClientColorTextureDumpDirIn);
    std::string wideFovClientTexelUsageDumpDir;
    if (args::get(dumpWideFovClientTexelUsageIn) || wideFovClientTexelUsageDumpDirIn) {
        wideFovClientTexelUsageDumpDir = args::get(wideFovClientTexelUsageDumpDirIn);
    }
    if (args::get(dumpWideFovClientTexelUsageIn) && wideFovClientTexelUsageDumpDir.empty()) {
        wideFovClientTexelUsageDumpDir = outputPath.str() + "/widefov_texel_usage";
    }

    app.onRender([&](double now, double dt) {
        // Handle mouse input
        if (!(ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)) {
            auto mouseButtons = window->getMouseButtons();
            window->setMouseCursor(!mouseButtons.LEFT_PRESSED);
            static bool dragging = false;
            static bool prevMouseLeftPressed = false;
            static float lastX = windowSize.x / 2.0;
            static float lastY = windowSize.y / 2.0;
            if (!prevMouseLeftPressed && mouseButtons.LEFT_PRESSED) {
                dragging = true;
                prevMouseLeftPressed = true;

                auto cursorPos = window->getCursorPos();
                lastX = static_cast<float>(cursorPos.x);
                lastY = static_cast<float>(cursorPos.y);
            }
            if (prevMouseLeftPressed && !mouseButtons.LEFT_PRESSED) {
                dragging = false;
                prevMouseLeftPressed = false;
            }
            if (dragging) {
                auto cursorPos = window->getCursorPos();
                float xpos = static_cast<float>(cursorPos.x);
                float ypos = static_cast<float>(cursorPos.y);

                float xoffset = xpos - lastX;
                float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

                lastX = xpos;
                lastY = ypos;

                camera.processMouseMovement(xoffset, yoffset, true);
            }
        }
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }

        if (cameraAnimator.running) {
            updateClient = cameraAnimator.update(!cameraPathFileIn ? dt : 1.0 / MILLISECONDS_IN_SECOND);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (updateClient) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        }
        else {
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
            camera.processKeyboard(keys, dt);
        }
        totalDT += dt;

        // // Render generated meshes
        // quasar.setDrawState(QuadMesh::DrawState::OPAQUE); // draw opaque quads first
        // renderStats = renderer.drawObjects(localScene, camera);
        // quasar.setDrawState(QuadMesh::DrawState::TRANSPARENT); // then draw transparent quads
        // renderStats += renderer.drawObjects(localScene, camera, 0);

        // tonemapper.drawToScreen(renderer);

        // auto quadsGenerator = quasar.getQuadsGenerator();
        // holeFiller.enableTonemapping(!showNormals);
        // holeFiller.setDepthThreshold(quadsGenerator->params.depthThreshold);
        // holeFiller.drawToScreen(renderer);
        

        if (rerenderIntervalMs > 0.0 && (now - lastRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendReferenceFrame = (frameCounter++) % refFrameInterval == 0; // insert Reference Frame every refFrameInterval frames
            sendResidualFrame = !sendReferenceFrame;
        }
        //  // write back onto disk
        // recorder.captureFrame(camera);
        // recorder.saveFrames(0);
        bool renderState = sendReferenceFrame || sendResidualFrame;
        if (sendReferenceFrame || sendResidualFrame) {
            spdlog::info("Do frame generation for frame: {}", frameCounter);
            // Update all animations
            if (runAnimations) {
                remoteScene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lastRenderTime = now;

            // "Send" pose to the server. this will wait until latency+/-jitter ms have passed
            spdlog::info("Send pose to server: {}, {}, {}", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
            poseSendRecvSimulator.sendPose(camera, now);
            bool receivedRemotePoseThisFrame = false;
            if (!preventCopyingLocalPose) {
                // "Receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                    activeRemoteRenderPose = clientPosePred;
                    activePredictionDebugInfo = poseSendRecvSimulator.getLastPredictionDebugInfo();
                    activePredictionCovariance = poseSendRecvSimulator.getLastPredictionCovariance();
                    receivedRemotePoseThisFrame = true;
                }
                spdlog::info("Remote Render with pose: {}, {}, {}", remoteCamera.getPosition().x, remoteCamera.getPosition().y, remoteCamera.getPosition().z);
                // If we do not have a new pose, just send a new frame with the old pose
            }

            // write remoteCamera to file
            glm::vec3 remotePos = remoteCamera.getPosition();
            glm::vec3 remoteRot = remoteCamera.getRotationEuler();
            spdlog::info(
                    "Remote Render with pose: pos=({}, {}, {}), rot=({}, {}, {})",
                    remotePos.x,
                    remotePos.y,
                    remotePos.z,
                    remoteRot.x,
                    remoteRot.y,
                    remoteRot.z);

            // pop E from Es on the front
            // if (!Es.empty()) {
            //     float E = Es.front();
            //     Es.pop();
            //     quasar.setViewSphereDiameter(E*2);
            //     spdlog::info("Set View Sphere Diameter to {}", E);
            // }

            const glm::mat4* wideFovGroundTruthView = nullptr;
            glm::mat4 nextPoseViewMatrix(1.0f);
            WideFovReprojectionUncertainty wideFovReprojectionUncertainty;
            std::vector<glm::mat4> debugWideFovCovarianceMaskTargetViews;
            activeWideFovCovarianceSamples.clear();
            const int nextQuasarRemoteFrameID = quasar.frameID + 1;
            const glm::mat4 remoteRenderViewMatrix = remoteCamera.getViewMatrix();
            const PoseCsvInfo remoteRenderPoseInfo = extractPoseCsvInfoFromViewMatrix(remoteRenderViewMatrix);
            int mappedRecordedFrameID = -1;
            if (wideFovGroundTruthPoseRecords.has_value()) {
                const auto recordIt = wideFovGroundTruthPoseRecords->find(nextQuasarRemoteFrameID);
                if (recordIt != wideFovGroundTruthPoseRecords->end()) {
                    mappedRecordedFrameID = recordIt->second.recordedFrameID;
                }
            }
            if (wideFovReprojectionUncertaintyRequested
                && receivedRemotePoseThisFrame
                && activeRemoteRenderPose.has_value()
                && activePredictionCovariance.valid)
            {
                const PoseCsvInfo covarianceCenterPoseInfo =
                    extractPoseCsvInfoFromViewMatrix(activeRemoteRenderPose->mono.view);
                const float covarianceCenterViewMatrixMaxDiff =
                    maxAbsMatrixElementDiff(remoteRenderViewMatrix, activeRemoteRenderPose->mono.view);
                const glm::quat remoteRotationForDiff = glm::normalize(glm::quat_cast(
                    glm::mat3(glm::inverse(remoteRenderViewMatrix))));
                glm::quat centerRotationForDiff = glm::normalize(glm::quat_cast(
                    glm::mat3(glm::inverse(activeRemoteRenderPose->mono.view))));
                if (glm::dot(remoteRotationForDiff, centerRotationForDiff) < 0.0f) {
                    centerRotationForDiff = -centerRotationForDiff;
                }
                const float covarianceCenterPositionDiffM = glm::distance(
                    remoteRenderPoseInfo.position,
                    covarianceCenterPoseInfo.position);
                const float covarianceCenterRotationDiffDeg = glm::degrees(
                    2.0f * std::acos(glm::clamp(
                        glm::dot(remoteRotationForDiff, centerRotationForDiff),
                        -1.0f,
                        1.0f)));
                spdlog::info(
                    "Covariance center check frame {}: remote-render pos=({}, {}, {}), rot=({}, {}, {}); "
                    "uncertainty-center pos=({}, {}, {}), rot=({}, {}, {}); diff=({:.9f}m, {:.9f}deg, {:.9e} matrix)",
                    nextQuasarRemoteFrameID,
                    remoteRenderPoseInfo.position.x,
                    remoteRenderPoseInfo.position.y,
                    remoteRenderPoseInfo.position.z,
                    remoteRenderPoseInfo.eulerRotationDegrees.x,
                    remoteRenderPoseInfo.eulerRotationDegrees.y,
                    remoteRenderPoseInfo.eulerRotationDegrees.z,
                    covarianceCenterPoseInfo.position.x,
                    covarianceCenterPoseInfo.position.y,
                    covarianceCenterPoseInfo.position.z,
                    covarianceCenterPoseInfo.eulerRotationDegrees.x,
                    covarianceCenterPoseInfo.eulerRotationDegrees.y,
                    covarianceCenterPoseInfo.eulerRotationDegrees.z,
                    covarianceCenterPositionDiffM,
                    covarianceCenterRotationDiffDeg,
                    covarianceCenterViewMatrixMaxDiff);
                if (covarianceCenterPositionDiffM > 1e-5f || covarianceCenterViewMatrixMaxDiff > 1e-5f) {
                    spdlog::warn(
                        "Covariance uncertainty center does not match remote render pose on frame {}",
                        nextQuasarRemoteFrameID);
                }

                wideFovReprojectionUncertainty.enabled = true;
                wideFovReprojectionUncertainty.valid = true;
                wideFovReprojectionUncertainty.meanViewMatrix = activeRemoteRenderPose->mono.view;
                wideFovReprojectionUncertainty.poseCovariance = activePredictionCovariance.pose6x6;
                wideFovReprojectionUncertainty.confidence = wideFovCovarianceConfidence;
                wideFovReprojectionUncertainty.minRadiusPx = wideFovUncertaintyMinRadiusPx;
                wideFovReprojectionUncertainty.maxRadiusPx = wideFovUncertaintyMaxRadiusPx;

                if (wideFovCovarianceDebugSampling && wideFovCovarianceSampleCount > 0) {
                    debugWideFovCovarianceMaskTargetViews.push_back(activeRemoteRenderPose->mono.view);
                    activeWideFovCovarianceSamples = poseSendRecvSimulator.samplePredictionCovariance(
                        *activeRemoteRenderPose,
                        activePredictionCovariance,
                        wideFovCovarianceSampleCount,
                        wideFovCovarianceConfidence);
                    for (const Pose& sampledPose : activeWideFovCovarianceSamples) {
                        debugWideFovCovarianceMaskTargetViews.push_back(sampledPose.mono.view);
                    }
                }
                spdlog::info(
                    "Prepared analytic wide-FOV reprojection uncertainty for frame {}{}",
                    nextQuasarRemoteFrameID,
                    activeWideFovCovarianceSamples.empty()
                        ? ""
                        : " with debug Monte Carlo covariance samples");
            }

            predictionPoseTimestampsFile
                << nextQuasarRemoteFrameID << ","
                << mappedRecordedFrameID << ","
                << (activePredictionDebugInfo.usedPrediction ? 1 : 0) << ","
                << activePredictionDebugInfo.latestTimestampUs << ","
                << activePredictionDebugInfo.previousTimestampUs << ","
                << activePredictionDebugInfo.secondPreviousTimestampUs << ","
                << activePredictionDebugInfo.predictedTimestampUs
                << std::endl;

            recvPoseToRenderFile
                << nextQuasarRemoteFrameID << ","
                << mappedRecordedFrameID << ","
                << (activeRemoteRenderPose.has_value() ? static_cast<int64_t>(activeRemoteRenderPose->send_timestamp) : -1) << ","
                << remoteRenderPoseInfo.position.x << "," << remoteRenderPoseInfo.position.y << "," << remoteRenderPoseInfo.position.z << ","
                << remoteRenderPoseInfo.rotationQuat.x << "," << remoteRenderPoseInfo.rotationQuat.y << ","
                << remoteRenderPoseInfo.rotationQuat.z << "," << remoteRenderPoseInfo.rotationQuat.w << ","
                << remoteRenderPoseInfo.eulerRotationDegrees.x << "," << remoteRenderPoseInfo.eulerRotationDegrees.y << ","
                << remoteRenderPoseInfo.eulerRotationDegrees.z
                << std::endl;

            for (size_t sampleIndex = 0; sampleIndex < activeWideFovCovarianceSamples.size(); ++sampleIndex) {
                const Pose& sampledPose = activeWideFovCovarianceSamples[sampleIndex];
                const PoseCsvInfo sampledPoseInfo = extractPoseCsvInfoFromViewMatrix(sampledPose.mono.view);
                covarianceSampledPosesFile
                    << "frame_id_" << nextQuasarRemoteFrameID << "_sample_" << sampleIndex << ","
                    << nextQuasarRemoteFrameID << ","
                    << mappedRecordedFrameID << ","
                    << sampleIndex << ","
                    << wideFovCovarianceConfidence << ","
                    << static_cast<int64_t>(sampledPose.send_timestamp) << ","
                    << sampledPoseInfo.position.x << "," << sampledPoseInfo.position.y << "," << sampledPoseInfo.position.z << ","
                    << sampledPoseInfo.rotationQuat.x << "," << sampledPoseInfo.rotationQuat.y << ","
                    << sampledPoseInfo.rotationQuat.z << "," << sampledPoseInfo.rotationQuat.w << ","
                    << sampledPoseInfo.eulerRotationDegrees.x << "," << sampledPoseInfo.eulerRotationDegrees.y << ","
                    << sampledPoseInfo.eulerRotationDegrees.z
                    << std::endl;
            }

            if (useWideFovGroundTruth) {
                bool foundRecordedPoseForRemoteFrame = false;

                if (wideFovGroundTruthPoseRecords.has_value()) {
                    const auto recordIt = wideFovGroundTruthPoseRecords->find(nextQuasarRemoteFrameID);
                    if (recordIt != wideFovGroundTruthPoseRecords->end()) {
                        nextPoseViewMatrix = buildViewMatrixFromRecordedPose(
                            recordIt->second.position,
                            recordIt->second.eulerRotationDegrees,
                            camera.getProjectionMatrix());
                        wideFovGroundTruthView = &nextPoseViewMatrix;
                        foundRecordedPoseForRemoteFrame = true;
                        spdlog::info(
                            "Using wide-FOV GT recorded pose from recorded frame {} for QUASAR remote frame {}: "
                            "pos=({}, {}, {}), rot_deg=({}, {}, {})",
                            recordIt->second.recordedFrameID,
                            nextQuasarRemoteFrameID,
                            recordIt->second.position.x,
                            recordIt->second.position.y,
                            recordIt->second.position.z,
                            recordIt->second.eulerRotationDegrees.x,
                            recordIt->second.eulerRotationDegrees.y,
                            recordIt->second.eulerRotationDegrees.z);
                    }
                }

                if (!foundRecordedPoseForRemoteFrame) {
                    wideFovGroundTruthView = &camera.getViewMatrix();

                    if (const auto nextPose = cameraAnimator.getNextPose()) {
                        PerspectiveCamera nextPoseCamera(camera.getProjectionMatrix());
                        nextPoseCamera.setPosition(nextPose->position);
                        nextPoseCamera.setRotationQuat(nextPose->rotation);
                        nextPoseCamera.updateViewMatrix();

                        nextPoseViewMatrix = nextPoseCamera.getViewMatrix();
                        wideFovGroundTruthView = &nextPoseViewMatrix;
                    }

                    spdlog::info(
                        "Falling back to animator/client wide-FOV GT pose for QUASAR remote frame {} because no recorded pose mapping was found",
                        nextQuasarRemoteFrameID);
                }
            }

            quasar.generateFrame(
                sendResidualFrame,
                showNormals,
                showDepth,
                wideFovGroundTruthView,
                debugWideFovCovarianceMaskTargetViews.empty() ? nullptr : &debugWideFovCovarianceMaskTargetViews,
                wideFovReprojectionUncertainty.valid ? &wideFovReprojectionUncertainty : nullptr);
            quasar.sendFrame(PoseReceiver::PoseInfo{0, 0, 0}, sendResidualFrame);

            std::string frameType = sendReferenceFrame ? "Reference Frame" : "Residual Frame";
            spdlog::info("======================================================");
            spdlog::info("Rendering Time ({}): {:.3f}ms", frameType, quasar.stats.totalRenderTimeMs);
            spdlog::info("Create Proxies Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateProxiesTimeMs);
            spdlog::info("  Gen Quad Map Time ({}): {:.3f}ms", frameType, quasar.stats.totalGenQuadMapTimeMs);
            spdlog::info("  Simplify Time ({}): {:.3f}ms", frameType, quasar.stats.totalSimplifyTimeMs);
            spdlog::info("  Gather Quads Time ({}): {:.3f}ms", frameType, quasar.stats.totalGatherQuadsTime);
            spdlog::info("Create Mesh Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateMeshTimeMs);
            spdlog::info("  Append Quads Time ({}): {:.3f}ms", frameType, quasar.stats.totalAppendQuadsTimeMs);
            spdlog::info("  Create Vert/Ind Time ({}): {:.3f}ms", frameType, quasar.stats.totalCreateVertIndTimeMs);
            spdlog::info("Compress Time ({}): {:.3f}ms", frameType, quasar.stats.totalCompressTimeMs);
            if (showDepth) spdlog::info("Gen Depth Time ({}): {:.3f}ms", frameType, quasar.stats.totalGenDepthTimeMs);
            spdlog::info("Frame Size: {:.3f}MB", quasar.stats.frameSize / BYTES_PER_MEGABYTE);
            spdlog::info("Num Proxies: {}Proxies", quasar.stats.proxySizes.numQuads);

            showResidualFrame = sendResidualFrame;
            preventCopyingLocalPose = false;
            sendReferenceFrame = false;
            sendResidualFrame = false;
        }

        poseSendRecvSimulator.update(now);

        // Hide/show nodes based on user input
        int currentIndex  = quasar.lastMeshIndex % 2;
        int previousIndex = (quasar.lastMeshIndex + 1) % 2;
        for (int layer = 0; layer < maxLayers; layer++) {
            bool showLayer = showLayers[layer];
            if (layer == 0) {
                quasar.referenceFrameNodesLocal[currentIndex].visible = showLayer && !hideReferenceFrame;
                quasar.referenceFrameNodesLocal[previousIndex].visible = false;
                quasar.referenceFrameWireframesLocal[currentIndex].visible = showLayer && !hideReferenceFrame && showWireframe;
                quasar.referenceFrameWireframesLocal[previousIndex].visible = false;
                quasar.depthNode.visible = showLayer && !hideReferenceFrame && showDepth;
            }
            else {
                quasar.nodesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame;
                quasar.wireframesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame && showWireframe;
                quasar.depthNodesHidLayer[layer-1].visible = showLayer && !hideReferenceFrame && showDepth;
            }
        }
        // quasar.nodesHidLayer[maxLayers-1].visible = false;
        quasar.residualFrameNodeLocal.visible = showResidualFrame && !hideResidualFrame;
        quasar.residualFrameWireframeLocal.visible = quasar.residualFrameNodeLocal.visible && showWireframe;

        if (restrictMovementToViewSphere) {
            glm::vec3 remotePosition = remoteCamera.getPosition();
            glm::vec3 position = camera.getPosition();
            glm::vec3 direction = position - remotePosition;
            float distanceSquared = glm::dot(direction, direction);
            float radius = viewSphereDiameter / 2.0f;
            if (distanceSquared > radius * radius) {
                position = remotePosition + glm::normalize(direction) * radius;
            }
            camera.setPosition(position);
            camera.updateViewMatrix();
        }

        double startTime = window->getTime();

        // Render generated meshes
        // quasar.setDrawState(QuadMesh::DrawState::OPAQUE); // draw opaque quads first
        // if (QuadTexelUsageMaterial* wideFovTexelMat = quasar.getWideFovQuadTexelUsageMaterial()) {
        //     wideFovTexelMat->clearTexelUsageMap();
        // }
        renderStats = renderer.drawObjects(localScene, camera);
        // if (!wideFovClientColorTextureDumpDir.empty() && !quasar.frameRTsHidLayer.empty()) {
        //     // add outputPath to the dumpDir
        //     Path dumpDir(wideFovClientColorTextureDumpDir);
        //     dumpDir.mkdirRecursive();
        //     Path pngPath = dumpDir / ("widefov_colorTexture_" + std::to_string(quasar.frameID) + ".png");
        //     quasar.frameRTsHidLayer.back().colorTexture.writeToPNG(pngPath.str());
        // }
        // if (renderState) {
        //     spdlog::info("Do recording for frame: {}", frameCounter);
        //     if (cameraPathFileIn) {
        //         recorder.captureFrame(camera);

        //         if (!cameraAnimator.running) {
        //             poseSendRecvSimulator.printErrors();
        //             recorder.stop();
        //             window->close();
        //         }
        //     }
        //     else if (recordWindow.isRecording()) {
        //         recorder.captureFrame(camera);
        //     }
        // }
        // quasar.setDrawState(QuadMesh::DrawState::TRANSPARENT); // then draw transparent quads
        // renderStats += renderer.drawObjects(localScene, camera, 0);

        // Render to screen
        auto quadsGenerator = quasar.getQuadsGenerator();
        holeFiller.enableTonemapping(!showNormals);
        holeFiller.setDepthThreshold(quadsGenerator->params.depthThreshold);
        holeFiller.drawToScreen(renderer);

        auto dumpWideFovTexelUsageForCapture = [&](int captureFrameID) {
            if (wideFovClientTexelUsageDumpDir.empty()) {
                return;
            }

            QuadTexelUsageMaterial* wideFovTexelMat = quasar.getWideFovQuadTexelUsageMaterial();
            if (wideFovTexelMat == nullptr) {
                return;
            }

            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            glFinish();

            Path dumpDir(wideFovClientTexelUsageDumpDir);
            dumpDir.mkdirRecursive();
            Path pngPath = dumpDir / ("widefov_texel_usage_" + std::to_string(captureFrameID) + ".png");
            wideFovTexelMat->getTexelUsageTexture()->writeToPNG(pngPath.str());
        };

        if (!updateClient) {
            if (renderState) {
                spdlog::info("Not update client dispite rendering");
            }
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCamera);

        auto dumpPosePredictionAndRecordedPoseCsvRows = [&](int recordedFrameID) {
            const glm::quat recordedFrameRotationQuat = glm::normalize(camera.getRotationQuat());
            const int64_t recordedPoseTimestampUs = static_cast<int64_t>(timeutils::secondsToMicros(camera.getTimestamp()));
            const int64_t renderNowTimestampUs = static_cast<int64_t>(timeutils::secondsToMicros(now));
            const glm::vec3 recordedFramePosition = camera.getPosition();
            const glm::vec3 recordedFrameEuler = camera.getRotationEuler();
            recordedFramePoseFile
                << recordedFrameID << ","
                << recordedPoseTimestampUs << ","
                << renderNowTimestampUs << ","
                << recordedFramePosition.x << "," << recordedFramePosition.y << "," << recordedFramePosition.z << ","
                << recordedFrameRotationQuat.x << "," << recordedFrameRotationQuat.y << ","
                << recordedFrameRotationQuat.z << "," << recordedFrameRotationQuat.w << ","
                << recordedFrameEuler.x << "," << recordedFrameEuler.y << "," << recordedFrameEuler.z
                << std::endl;
        };

        if (cameraPathFileIn) {

            glm::vec3 cameraPos = camera.getPosition();
            glm::vec3 cameraRot = camera.getRotationEuler();
            spdlog::info(
                "Render with pose: pos=({}, {}, {}), rot=({}, {}, {})",
                cameraPos.x,
                cameraPos.y,
                cameraPos.z,
                cameraRot.x,
                cameraRot.y,
                cameraRot.z);
            const int recordedFrameID = recorder.getNextFrameID();
            dumpPosePredictionAndRecordedPoseCsvRows(recordedFrameID);
            dumpWideFovTexelUsageForCapture(recordedFrameID);
            recorder.captureFrame(camera);

            if (!cameraAnimator.running) {
                poseSendRecvSimulator.printErrors();
                recorder.stop();
                window->close();
            }
        }
        else if (recordWindow.isRecording()) {
            const int recordedFrameID = recorder.getNextFrameID();
            dumpPosePredictionAndRecordedPoseCsvRows(recordedFrameID);
            dumpWideFovTexelUsageForCapture(recordedFrameID);
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
