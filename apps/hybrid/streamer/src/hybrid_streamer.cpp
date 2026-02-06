#include <args/args.hxx>

#include <OpenGLApp.h>
#include <SceneLoader.h>
#include <Windowing/GLFWWindow.h>
#include <GUI/ImGuiManager.h>
#include <Renderers/ForwardRenderer.h>

#include <Path.h>
#include <Recorder.h>
#include <CameraAnimator.h>

#include <UI/CameraHeader.h>
#include <UI/FrameRateWindow.h>
#include <UI/FrameCaptureWindow.h>
#include <UI/RecordWindow.h>
#include <UI/SceneWindow.h>
#include <UI/TexturePreviewWindow.h>

#include <Streamers/HybridStreamer.h>
#include <Receivers/PoseReceiver.h>
#include <Streamers/PoseStreamer.h>

#include <HoleFiller.h>
#include <PostProcessing/Tonemapper.h>

using namespace quasar;

int main(int argc, char** argv) {
    Config config{};
    config.title = "Hybrid Streamer";
    config.targetFramerate = 30;
    config.sortTransparent = false;

    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});

    // Render Config
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> sceneFileIn(parser, "scene", "Path to scene file", {'S', "scene"}, "../assets/scenes/sponza.json");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<bool> displayIn(parser, "display", "Show window", {'d', "display"}, true);
    args::ValueFlag<uint> targetBitrateIn(parser, "target-bitrate", "Target bitrate (Mbps)", {'b', "target-bitrate"}, 12);
    // Render config for meshwarp
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote camera FOV in degrees", {'F', "remote-fov"}, 80.0f);
    args::ValueFlag<float> remoteFOVWideIn(parser, "remote-fov-wide", "Remote camera FOV in degrees for wide fov", {'W', "remote-fov-wide"}, 140.0f);
    // Render config for depth peeling 
    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    args::ValueFlag<float> viewSphereDiameterIn(parser, "view-sphere-diameter", "Size of view sphere in m", {'B', "view-size"}, 0.5f);

    // URL config
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<std::string> videoAtlasURLIn(parser, "videoAtlas", "URL to recv video", {'c', "video-atlas-url"}, "127.0.0.1:12345");
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to recv video", {"cv", "video-url"}, "127.0.0.1:12346");
    args::ValueFlag<std::string> videoWideFovURLIn(parser, "video-wide", "URL to recv wide fov video", {"cw", "video-wide-url"}, "127.0.0.1:12347");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "URL to recv depth", {"de", "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> depthWideFovURLIn(parser, "depth-wide", "URL to recv wide fov depth", {"we", "depth-widefov-url"}, "127.0.0.1:65433");
    args::ValueFlag<std::string> proxiesURLIn(parser, "proxies", "URL to recv quad proxy metadata", {"px", "proxies-url"}, "127.0.0.1:65434");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "URL to recv camera pose", {"po", "pose-url"}, "0.0.0.0:54321");
    
    // Parse Config
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

    // Parse Size
    std::string sizeStr = args::get(sizeIn);
    size_t pos = sizeStr.find('x');
    glm::uvec2 windowSize = glm::uvec2(std::stoi(sizeStr.substr(0, pos)), std::stoi(sizeStr.substr(pos + 1)));
    config.width = windowSize.x;
    config.height = windowSize.y;

    config.enableVSync = !args::get(novsync);
    config.showWindow = args::get(displayIn);

    Path sceneFile = args::get(sceneFileIn);
    std::string videoAtlasURL = args::get(videoAtlasURLIn);
    std::string videoURL = args::get(videoURLIn);
    std::string videoWideFovURL = args::get(videoWideFovURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string depthWideFovURL = args::get(depthWideFovURLIn);
    std::string poseURL = args::get(poseURLIn);
    std::string proxiesURL = args::get(proxiesURLIn);

    // Print out all the URLs with spdlog::info
    spdlog::info("Video Atlas URL: {}", videoAtlasURL);
    spdlog::info("Video URL: {}", videoURL);
    spdlog::info("Video Wide FOV URL: {}", videoWideFovURL);
    spdlog::info("Depth URL: {}", depthURL);
    spdlog::info("Depth Wide FOV URL: {}", depthWideFovURL);
    spdlog::info("Pose URL: {}", poseURL);
    spdlog::info("Proxies URL: {}", proxiesURL);
    spdlog::info("--------------------------------");

    uint maxHiddenLayers = args::get(maxHiddenLayersIn);
    uint targetBitrate = args::get(targetBitrateIn);
    uint depthFactor = args::get(depthFactorIn);
    
    // config window and GUI
    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;

    OpenGLApp app(config);
    ForwardRenderer renderer(config);
    DepthPeelingRenderer remoteRendererDP(config, maxHiddenLayers+1, true);
    DeferredRenderer remoteRenderer(config);
    
    // "Remote" scene
    Scene remoteScene;
    PerspectiveCamera remoteCamera(remoteRendererDP.width, remoteRendererDP.height);
    SceneLoader loader;
    loader.loadScene(sceneFile, remoteScene, remoteCamera);

    float remoteFOV = args::get(remoteFOVIn);
    remoteCamera.setFovyDegrees(remoteFOV);

    glm::vec3 initialPosition = remoteCamera.getPosition();
    
    QuadSet quadSet(windowSize);
    float remoteFOVWide = args::get(remoteFOVWideIn);
    float viewSphereDiameter = args::get(viewSphereDiameterIn);

    // "Local" scene for visualization
    Scene localScene;

    HybridStreamer hybridStreamer(
        quadSet, 
        remoteRendererDP, 
        remoteRenderer, 
        remoteScene,    
        remoteCamera, 
        {
            .hiddenLayers = maxHiddenLayers, 
            .viewSphereDiameter = viewSphereDiameter,
            .targetBitRate = targetBitrate,
            .videoAtlasURL = videoAtlasURL,
            .proxiesURL = proxiesURL,
            .videoURL = videoURL,
            .depthURL = depthURL,
            .videoWideFovURL = videoWideFovURL,
            .depthWideFovURL = depthWideFovURL,
        }
    );

    hybridStreamer.addMeshesToScene(localScene);

    PoseReceiver poseReceiver(&remoteCamera, poseURL);

    // Post processing
    HoleFiller holeFiller;

    bool sendFrame = true;

    const double serverFPSValues[] = {0, 1, 5, 10, 15, 30};
    const char* serverFPSLabels[] = {"0 FPS", "1 FPS", "5 FPS", "10 FPS", "15 FPS", "30 FPS"};
    int serverFPSIndex = 1;
    double rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];

    bool showVisibleLayer = true;
    bool showWideFovLayer = true;

    bool *showLayers = new bool[maxHiddenLayers];
    for (int i = 0; i < maxHiddenLayers; i++) {
        showLayers[i] = true;
    }

    RenderStats renderStats;
    pose_id_t prevPoseID;
    FrameRateWindow frameRateWindow;
    SceneWindow sceneWindow(localScene, ImVec2(430, 800));
    CameraHeader cameraHeader(remoteCamera, "Remote Camera", true);
    
    guiManager->onRender([&](double now, double dt) {
        static bool showUI = true;
        static bool showLayerPreviews = false;

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
            ImGui::MenuItem("Layer Previews", 0, &showLayerPreviews);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Scene")) {
            ImGui::MenuItem("Scene", 0, &sceneWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        sceneWindow.draw(now, dt);
        if (showUI) {
             ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
            ImGui::Begin(config.title.c_str(), &showUI);
            ImGui::Text("OpenGL Version: %s", glGetString(GL_VERSION));
            ImGui::Text("GPU: %s\n", glGetString(GL_RENDERER));

            ImGui::Separator();

            size_t totalTriangles = hybridStreamer.getNumTriangles();
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

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Proxies URL: %s", proxiesURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            ImGui::Text("Client Pose ID: %d", prevPoseID);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Quad Generation Settings")) {
                auto quadsGenerator = hybridStreamer.getQuadsGenerator();
                ImGui::Checkbox("Correct Extreme Normals", &quadsGenerator->params.correctOrientation);
                ImGui::DragFloat("Depth Threshold", &quadsGenerator->params.depthThreshold, 0.0001f, 0.0f, 1.0f, "%.4f");
                ImGui::DragFloat("Angle Threshold", &quadsGenerator->params.angleThreshold, 0.1f, 0.0f, 180.0f);
                ImGui::DragFloat("Flatten Threshold", &quadsGenerator->params.flattenThreshold, 0.001f, 0.0f, 1.0f);
                ImGui::DragFloat("Plane Similarity Threshold", &quadsGenerator->params.planeSimilarityThreshold, 0.001f, 0.0f, 2.0f);
                ImGui::DragInt("Force Merge Iterations", &quadsGenerator->params.maxIterForceMerge, 1, 0, quadsGenerator->numQuadMaps);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Video Stats")) {
                auto& videoStreamerRT = hybridStreamer.videoAtlasStreamerRT;
                ImGui::TextColored(ImVec4(1,0.5,0,1), "Frame Rate: %.1f FPS (%.3f ms/frame)", videoStreamerRT.getFrameRate(), 1000.0f / videoStreamerRT.getFrameRate());
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy frame: %.3f ms", videoStreamerRT.stats.transferTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to encode frame: %.3f ms", videoStreamerRT.stats.encodeTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to send frame: %.3f ms", videoStreamerRT.stats.sendTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0.5,1), "Bitrate: %.3f Mbps", videoStreamerRT.stats.bitrateMbps);
            }

            ImGui::Separator();

            if (ImGui::Combo("Server Framerate", &serverFPSIndex, serverFPSLabels, IM_ARRAYSIZE(serverFPSLabels))) {
                rerenderIntervalMs = serverFPSIndex == 0 ? 0.0 : MILLISECONDS_IN_SECOND / serverFPSValues[serverFPSIndex];
            }

            ImGui::Separator();

            if (ImGui::DragFloat("View Sphere Diameter", &viewSphereDiameter, 0.025f, 0.1f, 2.0f)) {
                sendFrame = true;
                hybridStreamer.setViewSphereDiameter(viewSphereDiameter);
            }

            ImGui::Separator();
            
            ImGui::Checkbox("Show Visible Layer", &showVisibleLayer);
            ImGui::Checkbox("Show Wide FOV Layer", &showWideFovLayer);
            const int columns = 3;
            for (int layer = 0; layer < maxHiddenLayers; layer++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(layer)).c_str(), &showLayers[layer]);
                if ((layer + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();

            if (showLayerPreviews) {

                if (showVisibleLayer) {
                    ImGui::Begin("Visible Layer Preview##", 0, ImGuiWindowFlags_AlwaysAutoResize);
                    ImGui::Image((void*)(intptr_t)(hybridStreamer.frameRTVisible.colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }
                for (int layer = 0; layer < maxHiddenLayers; layer++) {
                    if (showLayers[layer]) {
                        ImGui::Begin(("Hidden Layer " + std::to_string(layer) + " Preview##").c_str(), 0, ImGuiWindowFlags_AlwaysAutoResize);
                        ImGui::Image((void*)(intptr_t)(hybridStreamer.frameRTsHidLayer[layer].colorTexture.ID),
                                         ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                        ImGui::End();
                    }
                }
                if (showWideFovLayer) {
                    ImGui::Begin("Wide FOV Layer Preview##", 0, ImGuiWindowFlags_AlwaysAutoResize);
                    ImGui::Image((void*)(intptr_t)(hybridStreamer.frameRTVisibleWideFov.colorTexture.ID),
                                     ImVec2(430, 270), ImVec2(0, 1), ImVec2(1, 0));
                    ImGui::End();
                }

            }
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        remoteRendererDP.setWindowSize(windowSize.x, windowSize.y);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        remoteCamera.setAspect(windowSize);
        remoteCamera.updateProjectionMatrix();
    });

    double totalDT = 0.0;
    double lastRenderTime = -INFINITY;
    int frameCounter = 0;

    app.onRender([&](double now, double dt) {
        
        // Handle Keyboard input
        auto keys = window->getKeys();
        if (keys.ESC_PRESSED) {
            window->close();
        }
        totalDT += dt;

        if (rerenderIntervalMs > 0.0 && (now - lastRenderTime) >= timeutils::millisToSeconds(rerenderIntervalMs - 1.0)) {
            sendFrame = true;
        }

        if (sendFrame) {
            remoteScene.updateAnimations(totalDT);
            totalDT = 0.0;
            lastRenderTime = now;

            // Receive pose
            PoseReceiver::PoseInfo poseInfo = poseReceiver.receivePose();
            pose_id_t poseID = poseInfo.pose_id;
            double poseSendTimestamp = poseInfo.send_timestamp;
            double poseRecvTimestamp = poseInfo.recv_timestamp;

            if (poseID != -1 && poseID != prevPoseID) {

                // Offset camera
                remoteCamera.setPosition(remoteCamera.getPosition());
                remoteCamera.updateViewMatrix();

                renderStats = hybridStreamer.generateFrame();

                // Restore camera position
                remoteCamera.setPosition(remoteCamera.getPosition());
                remoteCamera.updateViewMatrix();

                hybridStreamer.sendFrame(poseInfo);
                prevPoseID = poseID;
            }

            sendFrame = false;
        }

        // Offset camera
        remoteCamera.setPosition(remoteCamera.getPosition());
        remoteCamera.updateViewMatrix();

        spdlog::info("Draw scene with camera pose: {}, {}, {}", remoteCamera.getPosition().x, remoteCamera.getPosition().y, remoteCamera.getPosition().z);
        spdlog::info("Draw scene with camera rotation: {}, {}, {}", remoteCamera.getRotationEuler().x, remoteCamera.getRotationEuler().y, remoteCamera.getRotationEuler().z);
        spdlog::info("Draw scene with camera fovy: {}", remoteCamera.getFovyDegrees());
        spdlog::info("--------------------------------");

        // Render genereated meshes
        renderer.drawObjects(localScene, remoteCamera);

        // Restore camera position
        remoteCamera.setPosition(remoteCamera.getPosition());
        remoteCamera.updateViewMatrix();

        // Render to screen
        if (config.showWindow) {
            auto quadsGenerator = hybridStreamer.getQuadsGenerator();
            holeFiller.enableTonemapping(true);
            holeFiller.setDepthThreshold(quadsGenerator->params.depthThreshold);
            holeFiller.drawToScreen(renderer);
        }

    });

    // Run app loop (blocking)
    app.run();

    return 0;
}