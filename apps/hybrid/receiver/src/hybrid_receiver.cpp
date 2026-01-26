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
#include <UI/TexturePreviewWindow.h>

#include <Streamers/QUASARStreamer.h>
#include <Receivers/HybridReceiver.h>
#include <Streamers/PoseStreamer.h>

#include <HoleFiller.h>
#include <PostProcessing/Tonemapper.h>

using namespace quasar;



int main(int argc, char** argv) {
    Config config{};
    config.title = "Hybrid Receiver";
    
    args::ArgumentParser parser(config.title);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag verbose(parser, "verbose", "Enable verbose logging", {'v', "verbose"});
    
    // args for resolution and rendering
    args::ValueFlag<std::string> sizeIn(parser, "size", "Resolution of renderer", {'s', "size"}, "1920x1080");
    args::ValueFlag<std::string> resIn(parser, "rsize", "Resolution of remote renderer", {'r', "rsize"}, "1920x1080");
    args::Flag novsync(parser, "novsync", "Disable VSync", {'V', "novsync"}, false);
    args::ValueFlag<std::string> dataPathIn(parser, "data-path", "Path to data files", {'D', "data-path"}, "../simulator/");
    args::ValueFlag<uint> vertexGroupSizeIn(parser, "vertex", "Size of vertex grouping", {'g', "vertex-group-size"}, 1);
    args::ValueFlag<uint> depthFactorIn(parser, "factor", "Depth Resolution Factor", {'a', "depth-factor"}, 1);
    args::ValueFlag<float> remoteFOVIn(parser, "remote-fov", "Remote field of view", {'f', "remote-fov"}, 80.0f);
    

    args::ValueFlag<int> maxHiddenLayersIn(parser, "layers", "Max hidden layers", {'n', "max-hidden-layers"}, 3);
    
    // args related to URL
    args::ValueFlag<std::string> outputPathIn(parser, "output-path", "Directory to save outputs", {'o', "output-path"}, ".");
    args::ValueFlag<std::string> videoAtlasURLIn(parser, "video", "URL to recv atlas video", {'c', "video-url"}, "0.0.0.0:12345");
    args::ValueFlag<std::string> videoURLIn(parser, "video", "URL to recv video", {'x', "video-url"}, "0.0.0.0:12346");
    args::ValueFlag<std::string> videoWideFovURLIn(parser, "video-wide", "URL to recv wide fov video", {'w', "video-wide-url"}, "0.0.0.0:12347");
    args::ValueFlag<std::string> depthURLIn(parser, "depth", "URL to recv depth", {'e', "depth-url"}, "127.0.0.1:65432");
    args::ValueFlag<std::string> depthWideFovURLIn(parser, "depth-wide", "URL to recv wide fov depth", {'d', "depth-widefov-url"}, "127.0.0.1:65433");
    args::ValueFlag<std::string> proxiesURLIn(parser, "proxies", "URL to recv quad proxy metadata", {'s', "proxies-url"}, "127.0.0.1:65434");
    args::ValueFlag<std::string> poseURLIn(parser, "pose", "URL to recv camera pose", {'p', "pose-url"}, "127.0.0.1:54321");
    
    // Add for using camera path information
    args::Flag saveImages(parser, "save", "Save outputs to disk", {'I', "save-images"});
    args::ValueFlag<std::string> cameraPathFileIn(parser, "camera-path", "Path to camera animation file", {'C', "camera-path"});
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

    config.enableVSync = !args::get(novsync);
    
    // Parse URL for streaming
    Path dataPath = Path(args::get(dataPathIn));
    std::string videoAtlasURL = args::get(videoAtlasURLIn);
    std::string videoURL = args::get(videoURLIn);
    std::string videoWideFovURL = args::get(videoWideFovURLIn);
    std::string depthURL = args::get(depthURLIn);
    std::string depthWideFovURL = args::get(depthWideFovURLIn);
    std::string poseURL = args::get(poseURLIn);
    std::string proxiesURL = args::get(proxiesURLIn);

    // Number of hidden layers
    int hiddenLayers = args::get(maxHiddenLayersIn);

    // setup window
    auto window = std::make_shared<GLFWWindow>(config);
    auto guiManager = std::make_shared<ImGuiManager>(window);

    config.window = window;
    config.guiManager = guiManager;
    
    OpenGLApp app(config);
    ForwardRenderer renderer(config);

    Scene scene;
    PerspectiveCamera camera(windowSize);

    // Post processing
    HoleFiller holeFiller;
    Tonemapper tonemapper;


    Path outputPath = Path(args::get(outputPathIn)); outputPath.mkdirRecursive();
    Path cameraPathFile = args::get(cameraPathFileIn);

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
    CameraAnimator cameraAnimator(cameraPathFile, -1);

    if (cameraPathFileIn) {
        cameraAnimator.copyPoseToCamera(camera);
    }

    uint depthFactor = args::get(depthFactorIn);
    uint vertexGroupSize = args::get(vertexGroupSizeIn);
    
    // Initialize receiver
    QuadSet quadSet(windowSize);
    spdlog::info("Creating HybridReceiver with proxies url: {}", proxiesURL);
    spdlog::info("Creating HybridReceiver with video url: {}", videoURL);
    spdlog::info("Creating HybridReceiver with video wide fov url: {}", videoWideFovURL);
    spdlog::info("Creating HybridReceiver with depth url: {}", depthURL);
    spdlog::info("Creating HybridReceiver with depth wide fov url: {}", depthWideFovURL);
    spdlog::info("Creating HybridReceiver with pose url: {}", poseURL);
    spdlog::info("Creating HybridReceiver with video atlas url: {}", videoAtlasURL);
    HybridReceiver hybridReceiver(
        remoteWindowSize,
        depthFactor,
        vertexGroupSize,
        quadSet, 
        hiddenLayers,
        videoAtlasURL, 
        proxiesURL, 
        videoURL,
        depthURL,
        videoWideFovURL,
        depthWideFovURL);
    
    PoseStreamer poseStreamer(&camera, poseURL);

    // add in reverse order to have correct layering
    Node wideFovNode(&hybridReceiver.getVisibleMeshWideFOV());
    std::vector<Node> refNodes(hiddenLayers);
    Node visibleNode(&hybridReceiver.getVisibleMesh());

    // wideFovNode.frustumCulled = false;
    // visibleNode.frustumCulled = false;
    // // wideFovNode.primitiveType = GL_TRIANGLES;
    // scene.addChildNode(&wideFovNode);
    
    for (int i = hiddenLayers - 1; i >= 0; --i) {
        refNodes[i].addEntity(&hybridReceiver.getMesh(i));
        refNodes[i].frustumCulled = false;
        scene.addChildNode(&refNodes[i]);
    }

    // visibleNode.primitiveType = GL_TRIANGLES;
    scene.addChildNode(&visibleNode);

    // setup visible layer toggles
    bool showVisibleLayer = true;
    bool showWideFovLayer = true;
    bool* showLayers = new bool[hiddenLayers];
    for (int i = 0; i < hiddenLayers; i++) {
        showLayers[i] = true;
    }

    double elapsedTimeColor, elapsedTimeDepth;
    
    RenderStats renderStats;
    FrameRateWindow frameRateWindow;
    FrameCaptureWindow frameCaptureWindow(recorder, ImVec2(430, 270), outputPath);
    TexturePreviewWindow videoPreviewWindow("Video Texture", hybridReceiver.visibleTexture, ImVec2(430, 270));
    CameraHeader cameraHeader(camera);

    guiManager->onRender([&](double now, double dt) {
        static bool showUI = true;

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
            ImGui::MenuItem("Video Preview", 0, &videoPreviewWindow.visible);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        frameRateWindow.draw(now, dt);
        frameCaptureWindow.draw(now, dt);
        videoPreviewWindow.draw(now, dt);

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

            ImGui::Separator();

            cameraHeader.draw(now, dt);

            ImGui::Separator();

            ImGui::Text("Remote Pose ID: RGB (%d), D (%d)", hybridReceiver.poseIdColor, hybridReceiver.poseIdDepth);

            glm::mat4 pose = glm::inverse(hybridReceiver.depthFramePose.mono.view);
            glm::vec3 skew, scale;
            glm::quat rotationQuat;
            glm::vec3 remotePosition;
            glm::vec4 perspective;
            glm::decompose(pose, scale, rotationQuat, remotePosition, skew, perspective);
            glm::vec3 remoteRotation = glm::degrees(glm::eulerAngles(rotationQuat));
            ImGui::BeginDisabled();
            ImGui::InputFloat3("Remote Position", (float*)&remotePosition);
            ImGui::InputFloat3("Remote Rotation", (float*)&remoteRotation);
            ImGui::EndDisabled();

            ImGui::Separator();

            ImGui::Text("Video URL: %s", videoURL.c_str());
            ImGui::Text("Depth URL: %s", depthURL.c_str());
            ImGui::Text("Pose URL: %s", poseURL.c_str());

            ImGui::Separator();

            // if (ImGui::CollapsingHeader("Video Stats")) {
            //     ImGui::TextColored(ImVec4(1,0.5,0,1), "Frame Rate: RGB (%.1f FPS), D (%.1f FPS)",
            //                                             hybridReceiver.visibleTexture.getFrameRate(),
            //                                             hybridReceiver.depthTexture.getFrameRate());
            //     ImGui::TextColored(ImVec4(1,0.5,0.5,1), "E2E Latency: RGB (%.3f ms), D (%.3f ms)", elapsedTimeColor, elapsedTimeDepth);
            //     ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms",
            //                                             hybridReceiver.visibleTexture.stats.receiveTimeMs);
            //     ImGui::TextColored(ImVec4(0,0.5,0.5,1), "Bitrate: RGB (%.3f Mbps), D (%.3f Mbps)",
            //                                             hybridReceiver.visibleTexture.stats.bitrateMbps,
            //                                             hybridReceiver.depthTexture.stats.bitrateMbps);
            // }

            ImGui::TextColored(ImVec4(0,1,1,1), "Total Quads: %ld (%.3f MB)",
                               hybridReceiver.stats.sizes.numQuads,
                               hybridReceiver.stats.sizes.quadsSize / BYTES_PER_MEGABYTE);
            ImGui::TextColored(ImVec4(1,0,1,1), "Total Depth Offsets: %ld (%.3f MB)",
                               hybridReceiver.stats.sizes.numDepthOffsets,
                               hybridReceiver.stats.sizes.depthOffsetsSize / BYTES_PER_MEGABYTE);

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Video Stats")) {
                ImGui::TextColored(ImVec4(1,0.5,0,1), "Frame Rate: %.1f FPS (%.3f ms/frame)",
                                                        hybridReceiver.videoAtlasTexture.getFrameRate(),
                                                        1000.0f / hybridReceiver.videoAtlasTexture.getFrameRate());
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to receive frame: %.3f ms",
                                                        hybridReceiver.videoAtlasTexture.stats.receiveTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0.5,1), "Bitrate: %.3f Mbps",
                                                        hybridReceiver.videoAtlasTexture.stats.bitrateMbps);
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Proxy Stats")) {
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to load data: %.3f ms", hybridReceiver.stats.loadTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to decompress data (async): %.3f ms", hybridReceiver.stats.decompressTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to copy data to GPU: %.3f ms", hybridReceiver.stats.transferTimeMs);
                ImGui::TextColored(ImVec4(0,0.5,0,1), "Time to create mesh: %.3f ms", hybridReceiver.stats.createMeshTimeMs);
            }


            ImGui::Separator();
 
            ImGui::Checkbox("Show Visible Layer", &showVisibleLayer);
            ImGui::Checkbox("Show Wide FOV Layer", &showWideFovLayer);

            const int columns = 3;
            for (int layer = 0; layer < hiddenLayers; layer++) {
                ImGui::Checkbox(("Show Layer " + std::to_string(layer)).c_str(), &showLayers[layer]);
                if ((layer + 1) % columns != 0) {
                    ImGui::SameLine();
                }
            }

            ImGui::End();
        }
    });

    app.onResize([&](uint width, uint height) {
        windowSize = glm::uvec2(width, height);
        renderer.setWindowSize(windowSize.x, windowSize.y);
        camera.setAspect(windowSize);
        camera.updateProjectionMatrix();
    });


    bool updateClient = true;
    
    app.onRender([&](double now, double dt) {
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

        // if we have input file 
        if (cameraAnimator.running) {
            updateClient = cameraAnimator.update(!cameraPathFileIn ? dt : 1.0 / MILLISECONDS_IN_SECOND);
            now = cameraAnimator.now;
            dt = cameraAnimator.dt;
            if (updateClient) {
                cameraAnimator.copyPoseToCamera(camera);
            }
        } else {
            camera.processKeyboard(keys, dt);
            if (keys.ESC_PRESSED) {
                window->close();
            }
            auto scroll = window->getScrollOffset();
            camera.processScroll(scroll.y);
        }

        // Send pose to streamer
        poseStreamer.sendPose();
        hybridReceiver.recvData(poseStreamer, elapsedTimeColor, elapsedTimeDepth);
        poseStreamer.removePosesLessThan(std::min(hybridReceiver.poseIdColor, hybridReceiver.poseIdDepth));
        
        visibleNode.visible = showVisibleLayer;
        wideFovNode.visible = showWideFovLayer;
        for (int i = 0; i < hiddenLayers; i++) {
            refNodes[i].visible = showLayers[i];
        }

        renderStats = renderer.drawObjects(scene, camera);
        holeFiller.drawToScreen(renderer);
    }); 

    app.run();
    return 0;

}


// 6324480
// 37635840