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

#include <Streamers/HybridStreamer.h>
#include <HoleFiller.h>

#include <PoseSendRecvSimulator.h>
#include <Utils/TimeUtils.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <glm/gtx/matrix_decompose.hpp>

using namespace quasar;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

namespace {

struct PoseCsvInfo {
    glm::vec3 position{0.0f};
    glm::quat rotationQuat{1.0f, 0.0f, 0.0f, 0.0f};
};

PoseCsvInfo extractPoseCsvInfoFromViewMatrix(const glm::mat4& viewMatrix) {
    PoseCsvInfo info;
    glm::vec3 scale;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(viewMatrix), scale, info.rotationQuat, info.position, skew, perspective);
    info.rotationQuat = glm::normalize(info.rotationQuat);
    return info;
}

glm::vec2 computeScreenMotionDirectionPx(
    const glm::mat4& previousViewMatrix,
    const glm::mat4& currentViewMatrix,
    const glm::mat4& projectionMatrix,
    const glm::uvec2& viewportSize,
    float probeDepthM)
{
    const glm::vec2 viewportCenter(
        static_cast<float>(viewportSize.x) * 0.5f,
        static_cast<float>(viewportSize.y) * 0.5f);
    const float depth = std::max(0.01f, probeDepthM);

    const glm::vec4 previousCenterWorld =
        glm::inverse(previousViewMatrix) * glm::vec4(0.0f, 0.0f, -depth, 1.0f);
    const glm::vec4 clip = projectionMatrix * currentViewMatrix * previousCenterWorld;
    if (std::abs(clip.w) > 1e-5f) {
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        const glm::vec2 projectedPx(
            (ndc.x + 1.0f) * 0.5f * static_cast<float>(viewportSize.x),
            (ndc.y + 1.0f) * 0.5f * static_cast<float>(viewportSize.y));
        const glm::vec2 directionPx = projectedPx - viewportCenter;
        if (std::isfinite(directionPx.x) && std::isfinite(directionPx.y)
            && glm::length(directionPx) > 1e-4f)
        {
            return glm::normalize(directionPx);
        }
    }

    const PoseCsvInfo previousPose = extractPoseCsvInfoFromViewMatrix(previousViewMatrix);
    const PoseCsvInfo currentPose = extractPoseCsvInfoFromViewMatrix(currentViewMatrix);
    const glm::vec3 cameraSpaceDelta =
        glm::mat3(currentViewMatrix) * (currentPose.position - previousPose.position);
    const glm::vec2 translationDirectionPx(cameraSpaceDelta.x, cameraSpaceDelta.y);
    if (std::isfinite(translationDirectionPx.x) && std::isfinite(translationDirectionPx.y)
        && glm::length(translationDirectionPx) > 1e-4f)
    {
        return glm::normalize(translationDirectionPx);
    }

    return glm::vec2(0.0f);
}

} // namespace

int main(int argc, char** argv) {
    Config config{};
    config.title = "Hybrid Simulator";
    config.sortTransparent = false;

    glfwSetErrorCallback(glfw_error_callback);

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::Flag noFrameDumpIn(
        parser,
        "no-frame-dump",
        "Run save-images/camera-path mode without writing per-frame PNG captures",
        {"no-frame-dump"},
        false);
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
    args::ValueFlag<int> numPosesIn(parser, "num-poses", "Number of poses to load from camera path", {'N', "num-poses"}, -1);
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<float> networkLatencyIn(parser, "network-latency", "Simulated network latency in ms", {'N', "network-latency"}, 25.0f);
    args::ValueFlag<float> networkJitterIn(parser, "network-jitter", "Simulated network jitter in ms", {'J', "network-jitter"}, 10.0f);
    args::Flag posePredictionIn(parser, "pose-prediction", "Enable pose prediction", {'P', "pose-prediction"}, false);
    args::Flag poseSmoothingIn(parser, "pose-smoothing", "Enable pose smoothing", {'T', "pose-smoothing"}, false);
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<uint> vertexGroupSizeIn(parser, "vertex", "Size of vertex grouping", {'g', "vertex-group-size"}, 1);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 140.0f);
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);
    args::Flag dynamicEdpEIn(
        parser,
        "dynamic-edp-e",
        "Drive depth-peeling E from pose prediction uncertainty instead of always using viewSphereDiameter / 2",
        {"dynamic-edp-e"});
    args::ValueFlag<float> dynamicEdpEMinIn(
        parser,
        "meters",
        "Minimum dynamic depth-peeling E radius",
        {"dynamic-edp-e-min"},
        0.005f);
    args::ValueFlag<float> dynamicEdpEMaxIn(
        parser,
        "meters",
        "Maximum dynamic depth-peeling E radius; negative means viewSphereDiameter / 2",
        {"dynamic-edp-e-max"},
        -1.0f);
    args::ValueFlag<float> dynamicEdpEPositionScaleIn(
        parser,
        "scale",
        "Scale applied to the position uncertainty term for dynamic depth-peeling E",
        {"dynamic-edp-e-position-scale"},
        1.0f);
    args::ValueFlag<float> dynamicEdpERotationLeverArmIn(
        parser,
        "meters",
        "World-space lever arm used to convert rotation uncertainty radians into dynamic depth-peeling E meters",
        {"dynamic-edp-e-rotation-lever-arm"},
        1.0f);
    args::Flag dynamicEdpEDirectionalIn(
        parser,
        "dynamic-edp-e-directional",
        "Use the full dynamic E along apparent screen-space motion and a smaller E perpendicular to it",
        {"dynamic-edp-e-directional"});
    args::ValueFlag<float> dynamicEdpENonMotionScaleIn(
        parser,
        "scale",
        "Perpendicular scale for directional dynamic E, where 1 keeps the circular EDP radius",
        {"dynamic-edp-e-non-motion-scale"},
        0.5f);
    args::ValueFlag<std::string> wideFovDumpDirIn(
        parser, "path", "Dump wide-FOV tonemapped PNG per frame (empty = off)", {"wide-fov-dump-dir"}, "");
    args::Flag noWideFovDumpIn(
        parser,
        "no-wide-fov-dump",
        "Disable wide-FOV PNG/debug dumps while keeping CSV and bitrate logs",
        {"no-wide-fov-dump"},
        false);
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

    const bool saveImagesEnabled = args::get(saveImages);
    const bool frameDumpEnabled = saveImagesEnabled && !args::get(noFrameDumpIn);

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

    config.enableVSync = !args::get(novsync) && !saveImagesEnabled;
    config.showWindow = !saveImagesEnabled;

    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path sceneFile = args::get(sceneFileIn);
    Path cameraPathFile = args::get(cameraPathFileIn);

        
    // Read E path
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

    int numPoses = args::get(numPosesIn);

    uint maxHidLayers = args::get(maxHiddenLayersIn);
    uint maxLayers = maxHidLayers + 2;

    uint depthFactor = args::get(depthFactorIn);
    uint vertexGroupSize = args::get(vertexGroupSizeIn);

    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    config.width = remoteWindowSize.x;
    config.height = remoteWindowSize.y;
    DepthPeelingRenderer remoteRendererDP(config, maxLayers-1, true);
    DeferredRenderer remoteRenderer(config);

    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteWindowSize.x, remoteWindowSize.y);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    // "Local" scene
    Scene scene;
    // scene.skybox = remoteScene.skybox;
    scene.skybox = nullptr;
    PerspectiveCamera camera(windowSize);
    camera.setViewMatrix(remoteCamera.getViewMatrix());

    QuadSet quadSet(remoteWindowSize);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewSphereDiameter = args::get(viewSphereDiameterIn);
    const float baseEdpERadius = viewSphereDiameter * 0.5f;
    const bool dynamicEdpE = args::get(dynamicEdpEIn);
    const float dynamicEdpEMin = std::max(0.0f, args::get(dynamicEdpEMinIn));
    const float dynamicEdpEMaxArg = args::get(dynamicEdpEMaxIn);
    const float dynamicEdpEMax = std::max(
        dynamicEdpEMin,
        dynamicEdpEMaxArg < 0.0f ? baseEdpERadius : dynamicEdpEMaxArg);
    const float dynamicEdpEPositionScale = std::max(0.0f, args::get(dynamicEdpEPositionScaleIn));
    const float dynamicEdpERotationLeverArm = std::max(0.0f, args::get(dynamicEdpERotationLeverArmIn));
    const bool dynamicEdpEDirectional = args::get(dynamicEdpEDirectionalIn);
    const float dynamicEdpENonMotionScale =
        glm::clamp(args::get(dynamicEdpENonMotionScaleIn), 0.0f, 1.0f);

    if (dynamicEdpE) {
        spdlog::info(
            "Dynamic EDP E: Enabled, base={:.6f}m, clamp=[{:.6f}, {:.6f}]m, position_scale={:.3f}, rotation_lever_arm={:.3f}m",
            baseEdpERadius,
            dynamicEdpEMin,
            dynamicEdpEMax,
            dynamicEdpEPositionScale,
            dynamicEdpERotationLeverArm);
        if (dynamicEdpEDirectional) {
            spdlog::info(
                "Directional dynamic EDP E: enabled, perpendicular_scale={:.3f}",
                dynamicEdpENonMotionScale);
        }
    }

    std::string wideFovDumpDir = args::get(wideFovDumpDirIn);
    if (args::get(noWideFovDumpIn)) {
        wideFovDumpDir.clear();
        spdlog::info("Wide-FOV image/debug dumping: disabled");
    }
    // if (wideFovDumpDir.empty()) {
    //     wideFovDumpDir = outputPath.str() + "/widefov_dump";
    // }

    HybridStreamer hybridStreamer(
        quadSet,
        remoteRendererDP, remoteRenderer, remoteScene, remoteCamera,
        {
            .hiddenLayers = maxHidLayers,
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteFOVWide,
            .wideFovImageDumpDir = wideFovDumpDir,
        });

    // Node node(&hybridStreamer.getVisibleMesh());
    // node.frustumCulled = false;
    // scene.addChildNode(&node);
    hybridStreamer.addMeshesToScene(scene);

    // remoteScene.skybox = nullptr;
    scene.skybox = nullptr;

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
    }, renderer, tonemapper, outputPath, config.targetFramerate);
    CameraAnimator cameraAnimator(cameraPathFile, numPoses);

    if (saveImagesEnabled) {
        recorder.setTargetFrameRate(-1 /* unlimited */);
        recorder.setFormat(Recorder::OutputFormat::PNG);
        if (frameDumpEnabled) {
            recorder.start();
        }
        else {
            spdlog::info("Frame image dumping disabled by --no-frame-dump; CSV and bitrate logs remain enabled.");
        }
    }

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
        cameraAnimator.copyPoseToCamera(remoteCamera);
    }

    bool showWireframe = false;
    bool showDepth = false;
    bool preventCopyingLocalPose = false;
    bool runAnimations = cameraPathFileIn;

    bool sendRemoteFrame = true;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 60};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "60 FPS"};
    int serverFPSIndex = !cameraPathFileIn ? 0 : 5; // default to 60 FPS
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
    float networkLatency = !cameraPathFileIn ? 0.0f : args::get(networkLatencyIn);
    float networkJitter = !cameraPathFileIn ? 0.0f : args::get(networkJitterIn);
    bool posePrediction = posePredictionIn;
    bool poseSmoothing = poseSmoothingIn;
    PoseSendRecvSimulator poseSendRecvSimulator({
        .networkLatencyMs = networkLatency,
        .networkJitterMs = networkJitter,
        .renderTimeMs = rerenderIntervalMs,
        .posePrediction = posePrediction,
        .poseSmoothing = poseSmoothing,
    });

    std::ofstream dynamicEdpLogFile((outputPath / "hybrid_dynamic_edp.csv").str());
    dynamicEdpLogFile
        << "frame_id,prediction_used,"
        << "latest_pose_timestamp_us,prev_pose_timestamp_us,prev_but_two_pose_timestamp_us,predicted_pose_timestamp_us,"
        << "position_uncertainty_m,rotation_uncertainty_rad,dynamic_edp_e_m,"
        << "dynamic_edp_direction_x,dynamic_edp_direction_y,dynamic_edp_perpendicular_scale"
        << std::endl;

    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, ImVec2(430, 270), outputPath);
    RecordWindow recordWindow(recorder, ImVec2(430, 270), outputPath);
    // TexturePreviewWindow videoPreviewWindow("Video Texture", hybridStreamer.renderTarget.colorTexture, ImVec2(430, 270));
    SceneWindow sceneWindow(scene, ImVec2(430, 800));
    CameraHeader cameraHeader(camera);
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = !saveImagesEnabled;
        static bool showMeshCapture = false;

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
            // ImGui::MenuItem("Frame Preview", 0, &videoPreviewWindow.visible);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene", 0, &sceneWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        recordWindow.draw(now, dt);
        sceneWindow.draw(now, dt);
        // videoPreviewWindow.draw(now, dt);

        if (showUI) {
            ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            if (renderStats.trianglesDrawn < 100000)
                ImGui::TextColored(ImVec4(0,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else if (renderStats.trianglesDrawn < 500000)
                ImGui::TextColored(ImVec4(1,1,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Triangles Drawn: %ld", renderStats.trianglesDrawn);

            if (renderStats.drawCalls < 200)
                ImGui::TextColored(ImVec4(0,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else if (renderStats.drawCalls < 500)
                ImGui::TextColored(ImVec4(1,1,0,1), "Draw Calls: %ld", renderStats.drawCalls);
            else
                ImGui::TextColored(ImVec4(1,0,0,1), "Draw Calls: %ld", renderStats.drawCalls);

            // ImGui::TextColored(ImVec4(0,1,1,1), "Data Size: %.3f MB", static_cast<float>(hybridStreamer.stats.compressedSize) / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::Checkbox("Show Wireframe", &showWireframe);
            ImGui::Checkbox("Show Depth Map as Point Cloud", &showDepth);

            ImGui::Separator();

            if (ImGui::DragFloat("Remote FOV", &remoteFOV, 0.1f, 80.0f, 180.0f)) {
                remoteCamera.setFovyDegrees(remoteFOV);

                preventCopyingLocalPose = true;
                sendRemoteFrame = true;
                runAnimations = false;
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

            if (ImGui::Button("Send Frame", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                sendRemoteFrame = true;
                runAnimations = true;
            }

            ImGui::End();
        }

        if (showMeshCapture) {
            ImGui::SetNextWindowSize(ImVec2(430, 270), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(windowSize.x * 0.4, 300), ImGuiCond_FirstUseEver);
            ImGui::Begin("Mesh Capture", &showMeshCapture);

            // if (ImGui::Button("Save Depth")) {
            //     spdlog::info("Saved {} bytes to {}", hybridStreamer.writeToFiles(outputPath), outputPath.absolutePathStr());
            // }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lasttotalRenderTime = -INFINITY;
    bool updateClient = !saveImagesEnabled;
    PoseSendRecvSimulator::PredictionDebugInfo activePredictionDebugInfo;
    float currentDynamicEdpE = baseEdpERadius;
    glm::vec2 currentDynamicEdpDirectionPx(0.0f);
    float currentDynamicEdpNonMotionScale = 1.0f;
    glm::mat4 previousDynamicEdpViewMatrix = remoteCamera.getViewMatrix();
    bool hasPreviousDynamicEdpViewMatrix = false;
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

        if (rerenderIntervalMs > 0.0 && (now - lasttotalRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendRemoteFrame = true;
        }
        if (sendRemoteFrame) {
            // Update all animations
            if (runAnimations) {
                remoteScene.updateAnimations(totalDT);
                totalDT = 0.0;
            }
            lasttotalRenderTime = now;

            // "Send" pose to the server. this will wait until latency+/-jitter ms have passed
            poseSendRecvSimulator.sendPose(camera, now);
            if (!preventCopyingLocalPose) {
                // "Receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                    activePredictionDebugInfo = poseSendRecvSimulator.getLastPredictionDebugInfo();
                }
                // If we do not have a new pose, just send a new frame with the old pose
            }

            // // pop E from Es on the front
            // if (!Es.empty()) {
            //     float E = Es.front();
            //     Es.pop();
            //     hybridStreamer.setViewSphereDiameter(E*2);
            //     spdlog::info("Set View Sphere Diameter to {}", E);
            // }

            if (dynamicEdpE) {
                currentDynamicEdpE = baseEdpERadius;
                currentDynamicEdpDirectionPx = glm::vec2(0.0f);
                currentDynamicEdpNonMotionScale = 1.0f;
                if (activePredictionDebugInfo.valid && activePredictionDebugInfo.usedPrediction) {
                    const float predictionUncertaintyM =
                        dynamicEdpEPositionScale * activePredictionDebugInfo.predictionPositionUncertaintyM
                        + dynamicEdpERotationLeverArm * activePredictionDebugInfo.predictionRotationUncertaintyRad;
                    currentDynamicEdpE = glm::clamp(predictionUncertaintyM, dynamicEdpEMin, dynamicEdpEMax);
                }
                remoteRendererDP.setEOverride(currentDynamicEdpE);
                if (dynamicEdpEDirectional && hasPreviousDynamicEdpViewMatrix) {
                    currentDynamicEdpDirectionPx = computeScreenMotionDirectionPx(
                        previousDynamicEdpViewMatrix,
                        remoteCamera.getViewMatrix(),
                        remoteCamera.getProjectionMatrix(),
                        remoteWindowSize,
                        std::max(0.01f, dynamicEdpERotationLeverArm));
                    if (glm::length(currentDynamicEdpDirectionPx) > 1e-4f) {
                        currentDynamicEdpNonMotionScale = dynamicEdpENonMotionScale;
                        remoteRendererDP.setEAnisotropy(currentDynamicEdpDirectionPx, currentDynamicEdpNonMotionScale);
                    }
                    else {
                        remoteRendererDP.clearEAnisotropy();
                    }
                }
                else {
                    remoteRendererDP.clearEAnisotropy();
                }
                spdlog::info(
                    "Dynamic EDP E for Hybrid frame {}: E={:.6f}m, base={:.6f}m, pos_uncertainty={:.6f}m, rot_uncertainty={:.6f}rad, direction=({:.3f}, {:.3f}), perpendicular_scale={:.3f}",
                    hybridStreamer.frameID + 1,
                    currentDynamicEdpE,
                    baseEdpERadius,
                    activePredictionDebugInfo.predictionPositionUncertaintyM,
                    activePredictionDebugInfo.predictionRotationUncertaintyRad,
                    currentDynamicEdpDirectionPx.x,
                    currentDynamicEdpDirectionPx.y,
                    currentDynamicEdpNonMotionScale);
            }
            else {
                currentDynamicEdpE = baseEdpERadius;
                currentDynamicEdpDirectionPx = glm::vec2(0.0f);
                currentDynamicEdpNonMotionScale = 1.0f;
                remoteRendererDP.clearEOverride();
                remoteRendererDP.clearEAnisotropy();
            }

            dynamicEdpLogFile
                << (hybridStreamer.frameID + 1) << ","
                << (activePredictionDebugInfo.usedPrediction ? 1 : 0) << ","
                << activePredictionDebugInfo.latestTimestampUs << ","
                << activePredictionDebugInfo.previousTimestampUs << ","
                << activePredictionDebugInfo.secondPreviousTimestampUs << ","
                << activePredictionDebugInfo.predictedTimestampUs << ","
                << activePredictionDebugInfo.predictionPositionUncertaintyM << ","
                << activePredictionDebugInfo.predictionRotationUncertaintyRad << ","
                << currentDynamicEdpE << ","
                << currentDynamicEdpDirectionPx.x << ","
                << currentDynamicEdpDirectionPx.y << ","
                << currentDynamicEdpNonMotionScale
                << std::endl;

            // Generate new frame
            hybridStreamer.generateFrame();
            previousDynamicEdpViewMatrix = remoteCamera.getViewMatrix();
            hasPreviousDynamicEdpViewMatrix = true;

            spdlog::info("======================================================");
            // spdlog::info("Rendering Time: {:.3f}ms", hybridStreamer.stats.totalRenderTimeMs);
            // spdlog::info("Create Mesh Time: {:.3f}ms", hybridStreamer.stats.totalGenMeshTime);
            // spdlog::info("Compress Time: {:.3f}ms", hybridStreamer.stats.totalCompressTimeMs);
            // spdlog::info("Frame Size: {:.3f}MB", static_cast<float>(hybridStreamer.stats.compressedSize) / BYTES_PER_MEGABYTE);

            preventCopyingLocalPose = false;
            sendRemoteFrame = false;
        }

        poseSendRecvSimulator.update(now);

        // nodeWireframe.visible = showWireframe;
        // nodePointCloud.visible = showDepth;

        double startTime = window->getTime();

        // Render generated meshes
        // This is using information from the new camera
        // spdlog::info("Drawing local scene with {} objects", scene.children.size());

        // for (int childid = 0; childid < scene.children.size(); childid++) {
        //     Node* node = scene.children[childid];
        //     Mesh *mesh = dynamic_cast<Mesh*>(node->entities[0]);
        //     // spdlog::info("   The number of textures is {}", mesh->getMaterial()->getTextureCount());
        //     // mesh->getMaterial()->writeTextureToFile(0, "mesh_" + std::to_string(childid) + ".png");
        //     spdlog::info("  Mesh {}: {} triangles", childid, mesh->indexBuffer.getSize()/3);
        // }
        // // camera.setFovyDegrees(140.0f); // Use wide fov for local rendering
        renderStats = renderer.drawObjects(scene, camera);
        tonemapper.drawToScreen(renderer);
        if (!updateClient) {
            return;
        }
        if (cameraAnimator.running) {
            spdlog::info("Client Render Time: {:.3f}ms", timeutils::secondsToMillis(window->getTime() - startTime));
        }

        poseSendRecvSimulator.accumulateError(camera, remoteCamera);

        if (cameraPathFileIn) {
            if (frameDumpEnabled) {
                recorder.captureFrame(camera);
            }

            if (!cameraAnimator.running) {
                poseSendRecvSimulator.printErrors();
                if (frameDumpEnabled) {
                    recorder.stop();
                }
                window->close();
            }
        }
        else if (recordWindow.isRecording()) {
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
