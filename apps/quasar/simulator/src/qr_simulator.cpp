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
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
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

PoseCsvInfo poseCsvInfoFromAnimatorPose(const CameraAnimator::CameraPose& pose) {
    PoseCsvInfo info;
    info.position = pose.position;
    info.rotationQuat = glm::normalize(pose.rotation);
    info.eulerRotationDegrees = glm::degrees(glm::eulerAngles(info.rotationQuat));
    return info;
}

float quaternionAngularDistanceRad(const glm::quat& a, const glm::quat& b) {
    const glm::quat normalizedA = glm::normalize(a);
    const glm::quat normalizedB = glm::normalize(b);
    const float dotAbs = glm::clamp(std::abs(glm::dot(normalizedA, normalizedB)), 0.0f, 1.0f);
    return 2.0f * std::acos(dotAbs);
}

struct MotionProjectionFeatures {
    bool valid = false;
    glm::vec3 cameraFrameTranslation{0.0f};
    glm::vec2 projectedMotionPx{0.0f};
    float projectedMotionMagnitudePx = 0.0f;
    float representativeDepthM = 1.0f;
};

MotionProjectionFeatures computeMotionProjectionFeatures(
    const glm::mat4& sourceViewMatrix,
    const glm::mat4& targetViewMatrix,
    const glm::mat4& projectionMatrix,
    const glm::uvec2& viewportSize,
    float representativeDepthM)
{
    MotionProjectionFeatures features;
    features.representativeDepthM = std::max(0.01f, representativeDepthM);

    const PoseCsvInfo sourcePose = extractPoseCsvInfoFromViewMatrix(sourceViewMatrix);
    const PoseCsvInfo targetPose = extractPoseCsvInfoFromViewMatrix(targetViewMatrix);
    const glm::vec3 worldTranslation = targetPose.position - sourcePose.position;
    features.cameraFrameTranslation = glm::inverse(sourcePose.rotationQuat) * worldTranslation;

    const float fx = std::abs(projectionMatrix[0][0]) * static_cast<float>(viewportSize.x) * 0.5f;
    const float fy = std::abs(projectionMatrix[1][1]) * static_cast<float>(viewportSize.y) * 0.5f;
    features.projectedMotionPx = glm::vec2(
        fx * features.cameraFrameTranslation.x / features.representativeDepthM,
        fy * features.cameraFrameTranslation.y / features.representativeDepthM);
    features.projectedMotionMagnitudePx = glm::length(features.projectedMotionPx);

    features.valid =
        std::isfinite(features.cameraFrameTranslation.x)
        && std::isfinite(features.cameraFrameTranslation.y)
        && std::isfinite(features.cameraFrameTranslation.z)
        && std::isfinite(features.projectedMotionPx.x)
        && std::isfinite(features.projectedMotionPx.y)
        && std::isfinite(features.projectedMotionMagnitudePx);
    return features;
}

struct ScreenSpaceFootprintFeatures {
    bool valid = false;
    glm::vec2 directionPx{1.0f, 0.0f};
    float majorRadiusPx = 0.0f;
    float minorRadiusPx = 0.0f;
    int sampleCount = 0;
};

std::vector<float> parsePositiveFloatList(
    const std::string& values,
    const std::vector<float>& fallback)
{
    std::vector<float> parsed;
    std::stringstream ss(values);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            const float value = std::stof(token);
            if (std::isfinite(value) && value > 0.0f) {
                parsed.push_back(value);
            }
        }
        catch (const std::exception&) {
            spdlog::warn("Ignoring invalid positive-float-list token '{}'", token);
        }
    }

    return parsed.empty() ? fallback : parsed;
}

ScreenSpaceFootprintFeatures computeScreenSpaceFootprintFeatures(
    const glm::mat4& sourceViewMatrix,
    const glm::mat4& targetViewMatrix,
    const glm::mat4& projectionMatrix,
    const glm::uvec2& viewportSize,
    const std::vector<float>& representativeDepthsM,
    float scale,
    float maxRadiusPx)
{
    ScreenSpaceFootprintFeatures features;
    if (viewportSize.x == 0 || viewportSize.y == 0 || representativeDepthsM.empty()) {
        return features;
    }

    const float width = static_cast<float>(viewportSize.x);
    const float height = static_cast<float>(viewportSize.y);
    const glm::mat4 inverseSourceViewMatrix = glm::inverse(sourceViewMatrix);
    const float safeScale = std::max(0.0f, scale);
    const float maxRadius = maxRadiusPx > 0.0f ? maxRadiusPx : std::numeric_limits<float>::infinity();

    const std::vector<glm::vec2> anchorsNdc = {
        glm::vec2(0.0f, 0.0f),
        glm::vec2(-0.9f, -0.9f),
        glm::vec2(0.9f, -0.9f),
        glm::vec2(-0.9f, 0.9f),
        glm::vec2(0.9f, 0.9f),
        glm::vec2(0.0f, -0.9f),
        glm::vec2(0.0f, 0.9f),
        glm::vec2(-0.9f, 0.0f),
        glm::vec2(0.9f, 0.0f),
    };

    std::vector<glm::vec2> displacements;
    displacements.reserve(anchorsNdc.size() * representativeDepthsM.size());
    glm::vec2 dominantDisplacement(0.0f);
    float dominantMagnitude = 0.0f;

    for (const float depth : representativeDepthsM) {
        const float safeDepth = std::max(0.01f, depth);
        for (const glm::vec2& anchorNdc : anchorsNdc) {
            const glm::vec4 sourceViewPosition(
                anchorNdc.x * safeDepth / projectionMatrix[0][0],
                anchorNdc.y * safeDepth / projectionMatrix[1][1],
                -safeDepth,
                1.0f);
            const glm::vec4 sourceWorldPosition = inverseSourceViewMatrix * sourceViewPosition;
            const glm::vec4 targetClipPosition = projectionMatrix * targetViewMatrix * sourceWorldPosition;
            if (std::abs(targetClipPosition.w) <= 1e-5f) {
                continue;
            }

            const glm::vec2 targetNdc = glm::vec2(targetClipPosition) / targetClipPosition.w;
            if (!std::isfinite(targetNdc.x) || !std::isfinite(targetNdc.y)) {
                continue;
            }

            const glm::vec2 sourcePixel(
                (anchorNdc.x + 1.0f) * 0.5f * width,
                (anchorNdc.y + 1.0f) * 0.5f * height);
            const glm::vec2 targetPixel(
                (targetNdc.x + 1.0f) * 0.5f * width,
                (targetNdc.y + 1.0f) * 0.5f * height);
            const glm::vec2 displacement = targetPixel - sourcePixel;
            const float displacementMagnitude = glm::length(displacement);
            if (!std::isfinite(displacementMagnitude)) {
                continue;
            }

            displacements.push_back(displacement);
            if (displacementMagnitude > dominantMagnitude) {
                dominantMagnitude = displacementMagnitude;
                dominantDisplacement = displacement;
            }
        }
    }

    if (displacements.empty() || dominantMagnitude <= 1e-4f) {
        return features;
    }

    features.directionPx = glm::normalize(dominantDisplacement);
    const glm::vec2 minorDirection(-features.directionPx.y, features.directionPx.x);
    for (const glm::vec2& displacement : displacements) {
        features.majorRadiusPx = std::max(features.majorRadiusPx, std::abs(glm::dot(displacement, features.directionPx)));
        features.minorRadiusPx = std::max(features.minorRadiusPx, std::abs(glm::dot(displacement, minorDirection)));
    }

    features.majorRadiusPx = std::min(features.majorRadiusPx * safeScale, maxRadius);
    features.minorRadiusPx = std::min(features.minorRadiusPx * safeScale, maxRadius);
    features.sampleCount = static_cast<int>(displacements.size());
    features.valid =
        std::isfinite(features.majorRadiusPx)
        && std::isfinite(features.minorRadiusPx)
        && features.majorRadiusPx > 1e-4f;
    return features;
}

int positiveZViewOffsetSampleCount(const glm::vec3& viewOffsetUncertaintyM) {
    return viewOffsetUncertaintyM.z > 1e-5f
        && glm::compMax(glm::abs(viewOffsetUncertaintyM)) > 1e-5f
        ? 4
        : 0;
}

std::string describeViewOffsetMode(const glm::vec3& viewOffsetUncertaintyM) {
    if (glm::compMax(glm::abs(viewOffsetUncertaintyM)) <= 1e-5f) {
        return "none";
    }
    if (viewOffsetUncertaintyM.z > 1e-5f) {
        return "positive_z_samples";
    }
    if (viewOffsetUncertaintyM.z < -1e-5f) {
        return "negative_z_scalar";
    }
    return "xy_only_scalar";
}

float uintBitsToFloat(uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool readRGBA32UITexture(const Texture& texture, std::vector<glm::uvec4>& pixels) {
    pixels.resize(static_cast<size_t>(texture.width) * texture.height);

    GLint previousReadFramebuffer = 0;
    GLint previousReadBuffer = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
    glGetIntegerv(GL_READ_BUFFER, &previousReadBuffer);

    GLuint readFramebuffer = 0;
    glGenFramebuffers(1, &readFramebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFramebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.ID, 0);

    const GLenum status = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        spdlog::warn("Failed to read RGBA32UI texture: framebuffer status={}", status);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousReadFramebuffer));
        glReadBuffer(static_cast<GLenum>(previousReadBuffer));
        glDeleteFramebuffers(1, &readFramebuffer);
        return false;
    }

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(
        0,
        0,
        texture.width,
        texture.height,
        GL_RGBA_INTEGER,
        GL_UNSIGNED_INT,
        pixels.data());

    glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousReadFramebuffer));
    glReadBuffer(static_cast<GLenum>(previousReadBuffer));
    glDeleteFramebuffers(1, &readFramebuffer);
    return true;
}

glm::vec3 sourceViewPosFromPixelAndDepth(
    int x,
    int y,
    float depth,
    uint width,
    uint height,
    const glm::mat4& projectionMatrix)
{
    const glm::vec2 ndc =
        ((glm::vec2(static_cast<float>(x), static_cast<float>(y)) + glm::vec2(0.5f))
         / glm::vec2(static_cast<float>(width), static_cast<float>(height)))
        * 2.0f
        - glm::vec2(1.0f);
    return glm::vec3(
        ndc.x * depth / projectionMatrix[0][0],
        ndc.y * depth / projectionMatrix[1][1],
        -depth);
}

float scalarEForViewOffsetDebug(
    float dynamicE,
    bool viewOffsetFootprintEnabled,
    const glm::vec3& viewOffsetUncertaintyM,
    const glm::vec3& blockerViewPos)
{
    if (!viewOffsetFootprintEnabled || viewOffsetUncertaintyM.z >= 0.0f) {
        return dynamicE;
    }
    if (viewOffsetUncertaintyM.z < blockerViewPos.z) {
        return dynamicE;
    }

    const float z2 = std::abs(blockerViewPos.z);
    const float denominator = std::max(std::abs(blockerViewPos.z - viewOffsetUncertaintyM.z), 1e-5f);
    const float frontMotionE =
        z2
        * (std::abs(viewOffsetUncertaintyM.x) + std::abs(blockerViewPos.x))
        / denominator;

    if (!std::isfinite(frontMotionE) || frontMotionE <= 0.0f) {
        return dynamicE;
    }
    return frontMotionE;
}

struct DepthStats {
    size_t count = 0;
    float minNormalized = std::numeric_limits<float>::infinity();
    float maxNormalized = -std::numeric_limits<float>::infinity();
    double sumNormalized = 0.0;
    float minViewDepthM = std::numeric_limits<float>::infinity();
    float maxViewDepthM = -std::numeric_limits<float>::infinity();
    double sumViewDepthM = 0.0;
};

void accumulateDepthStats(
    DepthStats& stats,
    float normalizedDepth,
    float nearPlane,
    float farPlane)
{
    if (!std::isfinite(normalizedDepth) || normalizedDepth <= 0.0f || normalizedDepth >= 0.9999f) {
        return;
    }

    const float viewDepthM = glm::mix(nearPlane, farPlane, normalizedDepth);
    if (!std::isfinite(viewDepthM)) {
        return;
    }

    stats.count++;
    stats.minNormalized = std::min(stats.minNormalized, normalizedDepth);
    stats.maxNormalized = std::max(stats.maxNormalized, normalizedDepth);
    stats.sumNormalized += normalizedDepth;
    stats.minViewDepthM = std::min(stats.minViewDepthM, viewDepthM);
    stats.maxViewDepthM = std::max(stats.maxViewDepthM, viewDepthM);
    stats.sumViewDepthM += viewDepthM;
}

void dumpNormalRenderDepthStatsForRecordedFrame(
    const DeferredRenderer& renderer,
    const PerspectiveCamera& camera,
    const Path& outputPath,
    int recordedFrameID)
{
    std::vector<glm::uvec4> idPixels;
    if (!readRGBA32UITexture(renderer.gBuffer.idTexture, idPixels)) {
        spdlog::warn("Skipping normal-render depth stats for recorded frame {} because ID texture readback failed", recordedFrameID);
        return;
    }

    DepthStats stats;
    const float nearPlane = camera.getNear();
    const float farPlane = camera.getFar();
    for (const glm::uvec4& pixel : idPixels) {
        accumulateDepthStats(stats, uintBitsToFloat(pixel.z), nearPlane, farPlane);
    }

    if (stats.count == 0) {
        spdlog::warn("Skipping normal-render depth stats for recorded frame {} because no valid depth pixels were found", recordedFrameID);
        return;
    }

    const Path csvPath = outputPath / "normal_render_depth_stats.csv";
    const bool writeHeader = !csvPath.exists();
    std::ofstream csv(csvPath.str(), std::ios::app);
    if (writeHeader) {
        csv << "recorded_frame_id,width,height,valid_pixels,"
            << "near_m,far_m,"
            << "min_depth_normalized,max_depth_normalized,avg_depth_normalized,"
            << "min_view_depth_m,max_view_depth_m,avg_view_depth_m\n";
    }

    csv << std::setprecision(9)
        << recordedFrameID << ","
        << renderer.gBuffer.width << ","
        << renderer.gBuffer.height << ","
        << stats.count << ","
        << nearPlane << ","
        << farPlane << ","
        << stats.minNormalized << ","
        << stats.maxNormalized << ","
        << (stats.sumNormalized / static_cast<double>(stats.count)) << ","
        << stats.minViewDepthM << ","
        << stats.maxViewDepthM << ","
        << (stats.sumViewDepthM / static_cast<double>(stats.count))
        << "\n";

    spdlog::info(
        "Normal-render ID depth stats for recorded frame {}: valid_pixels={}, normalized[min={:.9f}, max={:.9f}, avg={:.9f}], view_depth_m[min={:.6f}, max={:.6f}, avg={:.6f}]",
        recordedFrameID,
        stats.count,
        stats.minNormalized,
        stats.maxNormalized,
        stats.sumNormalized / static_cast<double>(stats.count),
        stats.minViewDepthM,
        stats.maxViewDepthM,
        stats.sumViewDepthM / static_cast<double>(stats.count));
}

void dumpDynamicEPerPixelForRemoteFrame(
    const DepthPeelingRenderer& renderer,
    const PerspectiveCamera& camera,
    const Path& outputPath,
    int remoteFrameID,
    int recordedFrameID,
    float dynamicE,
    const glm::vec3& viewOffsetUncertaintyM)
{
    const uint width = renderer.width;
    const uint height = renderer.height;
    if (width == 0 || height == 0 || renderer.peelingLayers.size() < 2) {
        spdlog::warn("Skipping DynamicE pixel dump for remote frame {} because renderer dimensions/layers are invalid", remoteFrameID);
        return;
    }

    const Path dumpDir = outputPath / "dynamic_e_pixel_dump";
    dumpDir.mkdirRecursive();

    const std::string frameTag =
        "remote_frame_" + std::to_string(remoteFrameID)
        + (recordedFrameID >= 0 ? "_recorded_frame_" + std::to_string(recordedFrameID) : "");

    const Path metadataPath = dumpDir / (frameTag + "_metadata.txt");
    std::ofstream metadata(metadataPath.str());
    metadata << "remote_frame_id=" << remoteFrameID << "\n";
    metadata << "recorded_frame_id=" << recordedFrameID << "\n";
    metadata << "width=" << width << "\n";
    metadata << "height=" << height << "\n";
    metadata << "row_order=OpenGL pixel coordinates, y=0 bottom row first\n";
    metadata << "invalid_or_unused_pixels=nan\n";
    metadata << "dynamic_e_m=" << std::setprecision(9) << dynamicE << "\n";
    metadata << "view_offset_uncertainty_m=("
             << viewOffsetUncertaintyM.x << ","
             << viewOffsetUncertaintyM.y << ","
             << viewOffsetUncertaintyM.z << ")\n";
    metadata << "mode=" << describeViewOffsetMode(viewOffsetUncertaintyM) << "\n";
    metadata << "layer_file_semantics=layer_N uses prevIDMap from depth-peeling layer_N_minus_1, matching depth_peeling.glsl\n";

    const glm::mat4& projectionMatrix = camera.getProjectionMatrix();
    const float cameraNear = camera.getNear();
    const float cameraFar = camera.getFar();
    const float maxDepth = 0.9999f;

    for (size_t layer = 1; layer < renderer.peelingLayers.size(); ++layer) {
        std::vector<glm::uvec4> prevIDMap;
        const Texture& prevIDTexture = renderer.peelingLayers[layer - 1].idTexture;
        if (!readRGBA32UITexture(prevIDTexture, prevIDMap)) {
            continue;
        }

        const Path csvPath = dumpDir / (frameTag + "_layer_" + std::to_string(layer) + "_dynamic_e_m.csv");
        std::ofstream csv(csvPath.str());
        csv << std::setprecision(9);

        size_t validPixelCount = 0;
        float minE = std::numeric_limits<float>::infinity();
        float maxE = 0.0f;
        double sumE = 0.0;

        for (uint y = 0; y < height; ++y) {
            for (uint x = 0; x < width; ++x) {
                if (x > 0) {
                    csv << ",";
                }

                const glm::uvec4 q = prevIDMap[static_cast<size_t>(y) * width + x];
                const float blockerDepthNormalized = uintBitsToFloat(q.z);
                if (blockerDepthNormalized == 0.0f || blockerDepthNormalized >= maxDepth) {
                    csv << "nan";
                    continue;
                }

                const float blockerDepth = glm::mix(cameraNear, cameraFar, blockerDepthNormalized);
                const glm::vec3 blockerViewPos = sourceViewPosFromPixelAndDepth(
                    static_cast<int>(x),
                    static_cast<int>(y),
                    blockerDepth,
                    width,
                    height,
                    projectionMatrix);
                const float effectiveE = scalarEForViewOffsetDebug(
                    dynamicE,
                    renderer.eViewOffsetFootprint,
                    viewOffsetUncertaintyM,
                    blockerViewPos);

                csv << effectiveE;
                validPixelCount++;
                minE = std::min(minE, effectiveE);
                maxE = std::max(maxE, effectiveE);
                sumE += effectiveE;
            }
            csv << "\n";
        }

        metadata << "layer_" << layer << "_csv=" << csvPath.name() << "\n";
        metadata << "layer_" << layer << "_valid_pixels=" << validPixelCount << "\n";
        if (validPixelCount > 0) {
            metadata << "layer_" << layer << "_min_e_m=" << minE << "\n";
            metadata << "layer_" << layer << "_max_e_m=" << maxE << "\n";
            metadata << "layer_" << layer << "_mean_e_m=" << (sumE / static_cast<double>(validPixelCount)) << "\n";
        }
    }

    spdlog::info(
        "Dumped per-pixel DynamicE matrices for remote frame {} to {}",
        remoteFrameID,
        dumpDir.absolutePathStr());
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
    args::Flag noFrameDumpIn(
        parser,
        "no-frame-dump",
        "Run batch/path simulation without writing captured frame images; CSV and bitrate logs are still written",
        {"no-frame-dump"});
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
    args::Flag dynamicEdpEIn(
        parser,
        "dynamic-edp-e",
        "Drive depth-peeling E from pose prediction uncertainty instead of always using viewSphereDiameter / 2",
        {"dynamic-edp-e"});
    args::Flag dynamicEdpEGroundTruthErrorIn(
        parser,
        "dynamic-edp-e-ground-truth-error",
        "Drive dynamic depth-peeling E from the actual remote-vs-recorded pose error for offline/oracle analysis",
        {"dynamic-edp-e-ground-truth-error"});
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
    args::Flag dynamicEdpEMotionDirectionIn(
        parser,
        "dynamic-edp-e-motion-direction",
        "Use projected source->target camera-frame translation as the EDP anisotropy direction",
        {"dynamic-edp-e-motion-direction"});
    args::ValueFlag<float> dynamicEdpEMotionRepresentativeDepthIn(
        parser,
        "meters",
        "Representative depth used to project camera-frame translation into screen-space motion",
        {"dynamic-edp-e-motion-representative-depth"},
        1.0f);
    args::ValueFlag<float> dynamicEdpEMotionBoostThresholdPxIn(
        parser,
        "pixels",
        "If positive, projected motion above this threshold boosts dynamic E; disabled when <= 0",
        {"dynamic-edp-e-motion-boost-threshold-px"},
        -1.0f);
    args::ValueFlag<float> dynamicEdpEMotionBoostScaleIn(
        parser,
        "scale",
        "Scale applied to thresholded projected-motion feature when boosting dynamic E",
        {"dynamic-edp-e-motion-boost-scale"},
        0.0f);
    args::ValueFlag<float> dynamicEdpEMotionBoostMaxIn(
        parser,
        "scale",
        "Maximum additive E boost factor from projected-motion features",
        {"dynamic-edp-e-motion-boost-max"},
        1.0f);
    args::Flag dynamicEdpEScreenSpaceFootprintIn(
        parser,
        "dynamic-edp-e-screen-space-footprint",
        "Estimate a screen-space pose-error footprint and use it to enlarge the EDP sampling ellipse",
        {"dynamic-edp-e-screen-space-footprint"});
    args::ValueFlag<std::string> dynamicEdpEScreenSpaceDepthsIn(
        parser,
        "meters_csv",
        "Comma-separated representative depths used for screen-space EDP footprint projection",
        {"dynamic-edp-e-screen-space-depths"},
        "1,2,5,10");
    args::ValueFlag<float> dynamicEdpEScreenSpaceScaleIn(
        parser,
        "scale",
        "Scale applied to the projected screen-space EDP footprint radii",
        {"dynamic-edp-e-screen-space-scale"},
        1.0f);
    args::ValueFlag<float> dynamicEdpEScreenSpaceMaxPxIn(
        parser,
        "pixels",
        "Maximum screen-space EDP footprint radius in pixels; negative means unclamped",
        {"dynamic-edp-e-screen-space-max-px"},
        -1.0f);
    args::Flag dynamicEdpEViewOffsetFootprintIn(
        parser,
        "dynamic-edp-e-view-offset-footprint",
        "Use 4 positive-z view-space uncertainty offset samples for EDP, plus scalar DynamicE/LCOC fallback",
        {"dynamic-edp-e-view-offset-footprint", "dynamic-edp-e-pose-sample-footprint"});
    args::ValueFlag<float> dynamicEdpENonMotionScaleIn(
        parser,
        "scale",
        "Perpendicular scale for directional dynamic E, where 1 keeps the circular EDP radius",
        {"dynamic-edp-e-non-motion-scale"},
        0.5f);
    args::ValueFlag<int> dumpDynamicEFrameIn(
        parser,
        "remote_frame_id",
        "Dump per-pixel effective DynamicE matrices for this QUASAR remote frame id; disabled when < 0",
        {"dump-dynamic-e-frame"},
        -1);
    args::ValueFlag<int> dumpDynamicERecordedFrameIn(
        parser,
        "recorded_frame_id",
        "Recorded frame id label to include in per-pixel DynamicE dump filenames",
        {"dump-dynamic-e-recorded-frame"},
        -1);
    args::ValueFlag<int> dumpNormalRenderDepthStatsFrameIn(
        parser,
        "recorded_frame_id",
        "Render the original scene at this recorded client frame and dump normal-render ID-depth stats; disabled when < 0",
        {"dump-normal-render-depth-stats-frame"},
        -1);
    args::ValueFlag<int> wideFovPoseLagFramesIn(parser, "wide-fov-pose-lag", "Wide-FOV layer renders with camera view from this many frames ago", {'L', "wide-fov-pose-lag"}, 0);
    args::ValueFlag<int> wideFovUpdatePeriodIn(parser, "wide-fov-update-period", "Regenerate wide-FOV layer only when frameID mod N == 0 (1 = every frame)", {'K', "wide-fov-update-period"}, 1);
    args::ValueFlag<std::string> wideFovDumpDirIn(parser, "path", "Dump wide-FOV tonemapped PNG per frame to this folder (empty = off)", {"wide-fov-dump-dir"}, "");
    args::Flag noWideFovDumpIn(
        parser,
        "no-wide-fov-dump",
        "Disable wide-FOV PNG/debug dumps while keeping CSV and bitrate logs",
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
        "Wide-FOV mask method: none, stencil, or recorded-texel-usage",
        {"wide-fov-mask-method"},
        "");
    args::Flag useWideFovGroundTruthIn(
        parser,
        "use-wide-fov-ground-truth",
        "Wide-FOV reprojection uses local (client) view as current pose instead of predicted remote pose",
        {"use-wide-fov-ground-truth"});
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
    const bool frameDumpEnabled =
        args::get(saveImages)
        && !args::get(noFrameDumpIn);
    if (args::get(saveImages) && args::get(noFrameDumpIn)) {
        spdlog::info("Frame image dumping disabled by --no-frame-dump; CSV and bitrate logs will still be written");
    }

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
    const bool dynamicEdpEGroundTruthError = args::get(dynamicEdpEGroundTruthErrorIn);
    const bool dynamicEdpEMotionDirection = args::get(dynamicEdpEMotionDirectionIn);
    const float dynamicEdpEMotionRepresentativeDepth =
        std::max(0.01f, args::get(dynamicEdpEMotionRepresentativeDepthIn));
    const float dynamicEdpEMotionBoostThresholdPx = args::get(dynamicEdpEMotionBoostThresholdPxIn);
    const float dynamicEdpEMotionBoostScale = std::max(0.0f, args::get(dynamicEdpEMotionBoostScaleIn));
    const float dynamicEdpEMotionBoostMax = std::max(0.0f, args::get(dynamicEdpEMotionBoostMaxIn));
    const bool dynamicEdpEScreenSpaceFootprint = args::get(dynamicEdpEScreenSpaceFootprintIn);
    const std::vector<float> dynamicEdpEScreenSpaceDepths = parsePositiveFloatList(
        args::get(dynamicEdpEScreenSpaceDepthsIn),
        std::vector<float>{dynamicEdpEMotionRepresentativeDepth});
    const float dynamicEdpEScreenSpaceScale = std::max(0.0f, args::get(dynamicEdpEScreenSpaceScaleIn));
    const float dynamicEdpEScreenSpaceMaxPx = args::get(dynamicEdpEScreenSpaceMaxPxIn);
    const bool dynamicEdpEViewOffsetFootprint = args::get(dynamicEdpEViewOffsetFootprintIn);
    const float dynamicEdpENonMotionScale =
        glm::clamp(args::get(dynamicEdpENonMotionScaleIn), 0.0f, 1.0f);
    const int dumpDynamicEFrame = args::get(dumpDynamicEFrameIn);
    const int dumpDynamicERecordedFrame = args::get(dumpDynamicERecordedFrameIn);
    const int dumpNormalRenderDepthStatsFrame = args::get(dumpNormalRenderDepthStatsFrameIn);
    int wideFovPoseLagArg = args::get(wideFovPoseLagFramesIn);
    uint wideFovPoseLagFrames = static_cast<uint>(wideFovPoseLagArg < 0 ? 0 : wideFovPoseLagArg);


    spdlog::info("Remote FOV Wide: {}", remoteFOVWide);
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
        if (dynamicEdpEGroundTruthError) {
            spdlog::info("Dynamic EDP E source: ground-truth remote-vs-recorded pose error");
        }
        if (dynamicEdpEMotionDirection) {
            spdlog::info(
                "Dynamic EDP E motion features: enabled, representative_depth={:.3f}m, boost_threshold={:.3f}px, boost_scale={:.3f}, boost_max={:.3f}",
                dynamicEdpEMotionRepresentativeDepth,
                dynamicEdpEMotionBoostThresholdPx,
                dynamicEdpEMotionBoostScale,
                dynamicEdpEMotionBoostMax);
        }
        if (dynamicEdpEScreenSpaceFootprint) {
            std::stringstream depthList;
            for (size_t i = 0; i < dynamicEdpEScreenSpaceDepths.size(); ++i) {
                if (i > 0) {
                    depthList << ",";
                }
                depthList << dynamicEdpEScreenSpaceDepths[i];
            }
            spdlog::info(
                "Dynamic EDP E screen-space footprint: enabled, depths=[{}]m, scale={:.3f}, max_radius_px={:.3f}",
                depthList.str(),
                dynamicEdpEScreenSpaceScale,
                dynamicEdpEScreenSpaceMaxPx);
        }
        if (dynamicEdpEViewOffsetFootprint) {
            spdlog::info(
                "Dynamic EDP E view-offset footprint: enabled, using 4 positive-z source-view offset samples plus scalar DynamicE fallback");
        }
        if (dumpDynamicEFrame >= 0) {
            spdlog::info(
                "Per-pixel DynamicE dump: enabled for remote frame {}, recorded-frame label {}",
                dumpDynamicEFrame,
                dumpDynamicERecordedFrame);
        }
    }
    if (dumpNormalRenderDepthStatsFrame >= 0) {
        spdlog::info(
            "Normal-render depth stats dump: enabled for recorded frame {}",
            dumpNormalRenderDepthStatsFrame);
    }

    // hardcode wide fov pose lag frames to 1
    wideFovPoseLagFrames = 0;

    int wideFovUpdatePeriodArg = args::get(wideFovUpdatePeriodIn);
    uint wideFovUpdatePeriodFrames = static_cast<uint>(wideFovUpdatePeriodArg < 1 ? 1 : wideFovUpdatePeriodArg);
    WideFovMaskMethod wideFovMaskMethod = WideFovMaskMethod::None;
    const std::string wideFovMaskMethodValue = args::get(wideFovMaskMethodIn);
    if (!wideFovMaskMethodValue.empty()) {
        if (!tryParseWideFovMaskMethod(wideFovMaskMethodValue, wideFovMaskMethod)) {
            spdlog::error(
                "Invalid wide-FOV mask method '{}'. Expected one of: none, stencil, recorded-texel-usage",
                wideFovMaskMethodValue);
            return 1;
        }
    }
    else if (args::get(trimWideFovIn)) {
        wideFovMaskMethod = WideFovMaskMethod::Stencil;
    }
    bool trimWideFov = wideFovMaskMethod == WideFovMaskMethod::Stencil;
    bool useWideFovGroundTruth = args::get(useWideFovGroundTruthIn);
    const auto wideFovGroundTruthPoseRecords =
        loadWideFovGroundTruthPoseRecords(kWideFovGroundTruthPoseRecordsPath);

    spdlog::info("Wide FOV Update Period Frames: {}", wideFovUpdatePeriodFrames);
    spdlog::info("Trim Wide FOV: {}", trimWideFov ? "Enabled" : "Disabled");
    spdlog::info("Wide FOV Mask Method: {}", wideFovMaskMethodToString(wideFovMaskMethod));
    spdlog::info("Use Wide FOV Ground Truth: {}", useWideFovGroundTruth ? "Enabled" : "Disabled");

    //
    std::string wideFovDumpDir = args::get(wideFovDumpDirIn);
    if (args::get(noWideFovDumpIn)) {
        wideFovDumpDir.clear();
        spdlog::info("Wide-FOV image/debug dumping: disabled");
    }
    else if (wideFovDumpDir.empty()) {
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
            .datasetOutputDir = outputPath.str(),
        });

    quasar.addMeshesToScene(localScene);

    std::ofstream recvPoseToRenderFile((outputPath / "recv_pose_to_render.csv").str());
    recvPoseToRenderFile
        << "recorded_frame_id,pose_timestamp_us,"
        << "tx,ty,tz,qx,qy,qz,qw,rx_deg,ry_deg,rz_deg" << std::endl;
    std::ofstream predictionPoseTimestampsFile((outputPath / "pose_prediction_timestamps.csv").str());
    predictionPoseTimestampsFile
        << "recorded_frame_id,prediction_used,"
        << "latest_pose_timestamp_us,prev_pose_timestamp_us,prev_but_two_pose_timestamp_us,predicted_pose_timestamp_us,"
        << "position_uncertainty_m,rotation_uncertainty_rad,dynamic_edp_e_m,"
        << "dynamic_edp_direction_x,dynamic_edp_direction_y,dynamic_edp_perpendicular_scale,"
        << "groundtruth_position_error_m,groundtruth_rotation_error_rad,dynamic_edp_e_source,"
        << "groundtruth_target_source,groundtruth_target_x_m,groundtruth_target_y_m,groundtruth_target_z_m,"
        << "prediction_position_uncertainty_view_x_m,prediction_position_uncertainty_view_y_m,prediction_position_uncertainty_view_z_m,"
        << "groundtruth_position_error_view_x_m,groundtruth_position_error_view_y_m,groundtruth_position_error_view_z_m,"
        << "view_offset_uncertainty_x_m,view_offset_uncertainty_y_m,view_offset_uncertainty_z_m,"
        << "view_offset_mode,"
        << "motion_delta_cam_x_m,motion_delta_cam_y_m,motion_delta_cam_z_m,"
        << "motion_projected_dx_px,motion_projected_dy_px,motion_projected_magnitude_px,"
        << "motion_representative_depth_m,motion_e_boost_factor,"
        << "screen_footprint_major_radius_px,screen_footprint_minor_radius_px,"
        << "screen_footprint_direction_x,screen_footprint_direction_y,screen_footprint_sample_count,"
        << "view_offset_sample_count"
        << std::endl;
    std::ofstream recordedFramePoseFile((outputPath / "recorded_frame_render_poses.csv").str());
    recordedFramePoseFile
        << "recorded_frame_id,pose_timestamp_us,render_now_us,"
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

    if (frameDumpEnabled) {
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
    std::optional<Pose> activeRemoteRenderPose;
    float currentDynamicEdpE = baseEdpERadius;
    glm::vec2 currentDynamicEdpDirectionPx(0.0f);
    float currentDynamicEdpNonMotionScale = 1.0f;
    float currentGroundTruthPositionErrorM = 0.0f;
    float currentGroundTruthRotationErrorRad = 0.0f;
    std::string currentDynamicEdpESource = "base";
    MotionProjectionFeatures currentMotionProjectionFeatures;
    float currentMotionEBoostFactor = 1.0f;
    ScreenSpaceFootprintFeatures currentScreenSpaceFootprintFeatures;
    glm::vec3 currentPredictionPositionUncertaintyViewM{0.0f};
    glm::vec3 currentGroundTruthPositionErrorViewM{0.0f};
    PoseCsvInfo currentGroundTruthTargetPoseInfo;
    std::string currentGroundTruthTargetSource = "current_camera";
    glm::vec3 currentViewOffsetUncertaintyM{0.0f};
    int currentViewOffsetSampleCount = 0;
    glm::mat4 previousDynamicEdpViewMatrix = remoteCamera.getViewMatrix();
    bool hasPreviousDynamicEdpViewMatrix = false;
    int nonDumpedRecordedFrameID = 0;

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
            if (!preventCopyingLocalPose) {
                // "Receive" a predicted pose to render a new frame. this will wait until latency+/-jitter ms have passed
                Pose clientPosePred;
                if (poseSendRecvSimulator.recvPoseToRender(clientPosePred, now)) {
                    remoteCamera.setViewMatrix(clientPosePred.mono.view);
                    activeRemoteRenderPose = clientPosePred;
                    activePredictionDebugInfo = poseSendRecvSimulator.getLastPredictionDebugInfo();
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

            if (useWideFovGroundTruth) {
                const int nextQuasarRemoteFrameID = quasar.frameID + 1;
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

            if (dynamicEdpE) {
                currentDynamicEdpE = baseEdpERadius;
                currentDynamicEdpDirectionPx = glm::vec2(0.0f);
                currentDynamicEdpNonMotionScale = 1.0f;
                currentGroundTruthPositionErrorM = 0.0f;
                currentGroundTruthRotationErrorRad = 0.0f;
                currentMotionProjectionFeatures = {};
                currentMotionEBoostFactor = 1.0f;
                currentScreenSpaceFootprintFeatures = {};
                currentPredictionPositionUncertaintyViewM = glm::vec3(0.0f);
                currentGroundTruthPositionErrorViewM = glm::vec3(0.0f);
                currentGroundTruthTargetPoseInfo = extractPoseCsvInfoFromViewMatrix(camera.getViewMatrix());
                currentGroundTruthTargetSource = "current_camera";
                if (cameraPathFileIn) {
                    if (const auto nextPose = cameraAnimator.getNextPose()) {
                        currentGroundTruthTargetPoseInfo = poseCsvInfoFromAnimatorPose(*nextPose);
                        currentGroundTruthTargetSource = "next_render_pose";
                    }
                }
                currentViewOffsetUncertaintyM = glm::vec3(0.0f);
                currentViewOffsetSampleCount = 0;
                currentDynamicEdpESource = "base";
                remoteRendererDP.clearEScreenSpaceFootprint();
                remoteRendererDP.clearEViewOffsetFootprint();
                const glm::mat4& dynamicEdpSourceViewMatrix = remoteCamera.getViewMatrix();
                if (dynamicEdpEGroundTruthError) {
                    const PoseCsvInfo remoteRenderPoseInfo =
                        extractPoseCsvInfoFromViewMatrix(remoteCamera.getViewMatrix());
                    currentGroundTruthPositionErrorM =
                        glm::length(remoteRenderPoseInfo.position - currentGroundTruthTargetPoseInfo.position);
                    currentGroundTruthRotationErrorRad =
                        quaternionAngularDistanceRad(
                            remoteRenderPoseInfo.rotationQuat,
                            currentGroundTruthTargetPoseInfo.rotationQuat);
                    currentGroundTruthPositionErrorViewM =
                        glm::inverse(remoteRenderPoseInfo.rotationQuat)
                        * (currentGroundTruthTargetPoseInfo.position - remoteRenderPoseInfo.position);
                    currentViewOffsetUncertaintyM =
                        dynamicEdpEPositionScale * currentGroundTruthPositionErrorViewM;
                    const float groundTruthErrorM =
                        dynamicEdpEPositionScale * currentGroundTruthPositionErrorM
                        + dynamicEdpERotationLeverArm * currentGroundTruthRotationErrorRad;
                    currentDynamicEdpE = glm::clamp(groundTruthErrorM, dynamicEdpEMin, dynamicEdpEMax);
                    currentDynamicEdpESource = "groundtruth";
                }
                else if (activePredictionDebugInfo.valid && activePredictionDebugInfo.usedPrediction) {
                    const float predictionUncertaintyM =
                        dynamicEdpEPositionScale * activePredictionDebugInfo.predictionPositionUncertaintyM
                        + dynamicEdpERotationLeverArm * activePredictionDebugInfo.predictionRotationUncertaintyRad;
                    currentDynamicEdpE = glm::clamp(predictionUncertaintyM, dynamicEdpEMin, dynamicEdpEMax);
                    currentDynamicEdpESource = dynamicEdpEGroundTruthError ? "prediction_fallback" : "prediction";
                    currentPredictionPositionUncertaintyViewM =
                        activePredictionDebugInfo.predictionPositionUncertaintyViewM;
                    currentViewOffsetUncertaintyM =
                        dynamicEdpEPositionScale
                        * currentPredictionPositionUncertaintyViewM;
                }
                if (dynamicEdpEMotionDirection) {
                    currentMotionProjectionFeatures = computeMotionProjectionFeatures(
                        dynamicEdpSourceViewMatrix,
                        camera.getViewMatrix(),
                        remoteCamera.getProjectionMatrix(),
                        remoteWindowSize,
                        dynamicEdpEMotionRepresentativeDepth);
                    if (currentMotionProjectionFeatures.valid
                        && dynamicEdpEMotionBoostThresholdPx > 0.0f
                        && dynamicEdpEMotionBoostScale > 0.0f
                        && currentMotionProjectionFeatures.projectedMotionMagnitudePx > dynamicEdpEMotionBoostThresholdPx)
                    {
                        const float normalizedExcess =
                            (currentMotionProjectionFeatures.projectedMotionMagnitudePx - dynamicEdpEMotionBoostThresholdPx)
                            / dynamicEdpEMotionBoostThresholdPx;
                        const float additiveBoost = std::min(
                            dynamicEdpEMotionBoostMax,
                            dynamicEdpEMotionBoostScale * normalizedExcess);
                        currentMotionEBoostFactor = 1.0f + additiveBoost;
                        currentDynamicEdpE = glm::clamp(
                            currentDynamicEdpE * currentMotionEBoostFactor,
                            dynamicEdpEMin,
                            dynamicEdpEMax);
                        currentDynamicEdpESource += "_motion_boost";
                    }
                }
                if (dynamicEdpEScreenSpaceFootprint) {
                    currentScreenSpaceFootprintFeatures = computeScreenSpaceFootprintFeatures(
                        dynamicEdpSourceViewMatrix,
                        camera.getViewMatrix(),
                        remoteCamera.getProjectionMatrix(),
                        remoteWindowSize,
                        dynamicEdpEScreenSpaceDepths,
                        dynamicEdpEScreenSpaceScale,
                        dynamicEdpEScreenSpaceMaxPx);
                    if (currentScreenSpaceFootprintFeatures.valid) {
                        remoteRendererDP.setEScreenSpaceFootprint(
                            currentScreenSpaceFootprintFeatures.majorRadiusPx,
                            currentScreenSpaceFootprintFeatures.minorRadiusPx);
                        currentDynamicEdpESource += "_screen_footprint";
                    }
                    else {
                        remoteRendererDP.clearEScreenSpaceFootprint();
                    }
                }
                if (dynamicEdpEViewOffsetFootprint) {
                    remoteRendererDP.setEViewOffsetFootprint(currentViewOffsetUncertaintyM);
                    currentViewOffsetSampleCount =
                        positiveZViewOffsetSampleCount(currentViewOffsetUncertaintyM);
                    if (currentViewOffsetSampleCount > 0) {
                        currentDynamicEdpESource += "_view_offset_footprint";
                    }
                }
                remoteRendererDP.setEOverride(currentDynamicEdpE);
                const bool screenSpaceFootprintHasDirection =
                    currentScreenSpaceFootprintFeatures.valid
                    && glm::length(currentScreenSpaceFootprintFeatures.directionPx) > 1e-4f;
                if ((dynamicEdpEDirectional && (hasPreviousDynamicEdpViewMatrix || dynamicEdpEMotionDirection || screenSpaceFootprintHasDirection))
                    || (dynamicEdpEDirectional && dynamicEdpEViewOffsetFootprint)
                    || screenSpaceFootprintHasDirection)
                {
                    if (screenSpaceFootprintHasDirection) {
                        currentDynamicEdpDirectionPx =
                            glm::normalize(currentScreenSpaceFootprintFeatures.directionPx);
                    }
                    else if (dynamicEdpEMotionDirection
                        && currentMotionProjectionFeatures.valid
                        && currentMotionProjectionFeatures.projectedMotionMagnitudePx > 1e-4f)
                    {
                        currentDynamicEdpDirectionPx =
                            glm::normalize(currentMotionProjectionFeatures.projectedMotionPx);
                    }
                    else if (dynamicEdpEViewOffsetFootprint) {
                        currentDynamicEdpDirectionPx = glm::vec2(1.0f, 0.0f);
                    }
                    else {
                        currentDynamicEdpDirectionPx = computeScreenMotionDirectionPx(
                            previousDynamicEdpViewMatrix,
                            remoteCamera.getViewMatrix(),
                            remoteCamera.getProjectionMatrix(),
                            remoteWindowSize,
                            std::max(0.01f, dynamicEdpERotationLeverArm));
                    }
                    if (glm::length(currentDynamicEdpDirectionPx) > 1e-4f) {
                        currentDynamicEdpNonMotionScale = dynamicEdpEDirectional
                            ? dynamicEdpENonMotionScale
                            : 1.0f;
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
                    "Dynamic EDP E for QUASAR frame {}: E={:.6f}m, base={:.6f}m, source={}, pos_uncertainty={:.6f}m, rot_uncertainty={:.6f}rad, gt_pos_error={:.6f}m, gt_rot_error={:.6f}rad, view_offset_uncertainty=({:.6f}, {:.6f}, {:.6f})m, motion_projected=({:.3f}, {:.3f})px, motion_mag={:.3f}px, motion_boost={:.3f}, screen_footprint=({:.3f}, {:.3f})px, view_offset_sample_count={}, direction=({:.3f}, {:.3f}), perpendicular_scale={:.3f}",
                    quasar.frameID + 1,
                    currentDynamicEdpE,
                    baseEdpERadius,
                    currentDynamicEdpESource,
                    activePredictionDebugInfo.predictionPositionUncertaintyM,
                    activePredictionDebugInfo.predictionRotationUncertaintyRad,
                    currentGroundTruthPositionErrorM,
                    currentGroundTruthRotationErrorRad,
                    currentViewOffsetUncertaintyM.x,
                    currentViewOffsetUncertaintyM.y,
                    currentViewOffsetUncertaintyM.z,
                    currentMotionProjectionFeatures.projectedMotionPx.x,
                    currentMotionProjectionFeatures.projectedMotionPx.y,
                    currentMotionProjectionFeatures.projectedMotionMagnitudePx,
                    currentMotionEBoostFactor,
                    currentScreenSpaceFootprintFeatures.majorRadiusPx,
                    currentScreenSpaceFootprintFeatures.minorRadiusPx,
                    currentViewOffsetSampleCount,
                    currentDynamicEdpDirectionPx.x,
                    currentDynamicEdpDirectionPx.y,
                    currentDynamicEdpNonMotionScale);
                spdlog::info(
                    "Dynamic EDP E offset debug for QUASAR frame {}: mode={}, gt_target_source={}, gt_target_pos=({:.6f}, {:.6f}, {:.6f})m, prediction_view_uncertainty=({:.6f}, {:.6f}, {:.6f})m, groundtruth_view_error=({:.6f}, {:.6f}, {:.6f})m, scaled_shader_offset=({:.6f}, {:.6f}, {:.6f})m",
                    quasar.frameID + 1,
                    describeViewOffsetMode(currentViewOffsetUncertaintyM),
                    currentGroundTruthTargetSource,
                    currentGroundTruthTargetPoseInfo.position.x,
                    currentGroundTruthTargetPoseInfo.position.y,
                    currentGroundTruthTargetPoseInfo.position.z,
                    currentPredictionPositionUncertaintyViewM.x,
                    currentPredictionPositionUncertaintyViewM.y,
                    currentPredictionPositionUncertaintyViewM.z,
                    currentGroundTruthPositionErrorViewM.x,
                    currentGroundTruthPositionErrorViewM.y,
                    currentGroundTruthPositionErrorViewM.z,
                    currentViewOffsetUncertaintyM.x,
                    currentViewOffsetUncertaintyM.y,
                    currentViewOffsetUncertaintyM.z);
            }
            else {
                currentDynamicEdpE = baseEdpERadius;
                currentDynamicEdpDirectionPx = glm::vec2(0.0f);
                currentDynamicEdpNonMotionScale = 1.0f;
                currentGroundTruthPositionErrorM = 0.0f;
                currentGroundTruthRotationErrorRad = 0.0f;
                currentMotionProjectionFeatures = {};
                currentMotionEBoostFactor = 1.0f;
                currentScreenSpaceFootprintFeatures = {};
                currentPredictionPositionUncertaintyViewM = glm::vec3(0.0f);
                currentGroundTruthPositionErrorViewM = glm::vec3(0.0f);
                currentGroundTruthTargetPoseInfo = extractPoseCsvInfoFromViewMatrix(camera.getViewMatrix());
                currentGroundTruthTargetSource = "current_camera";
                currentViewOffsetUncertaintyM = glm::vec3(0.0f);
                currentViewOffsetSampleCount = 0;
                currentDynamicEdpESource = "disabled";
                remoteRendererDP.clearEOverride();
                remoteRendererDP.clearEAnisotropy();
                remoteRendererDP.clearEScreenSpaceFootprint();
                remoteRendererDP.clearEViewOffsetFootprint();
            }

            quasar.generateFrame(
                sendResidualFrame,
                showNormals,
                showDepth,
                wideFovGroundTruthView);
            if (dumpDynamicEFrame >= 0 && quasar.frameID == dumpDynamicEFrame) {
                dumpDynamicEPerPixelForRemoteFrame(
                    remoteRendererDP,
                    remoteCamera,
                    outputPath,
                    quasar.frameID,
                    dumpDynamicERecordedFrame,
                    currentDynamicEdpE,
                    currentViewOffsetUncertaintyM);
            }
            previousDynamicEdpViewMatrix = remoteCamera.getViewMatrix();
            hasPreviousDynamicEdpViewMatrix = true;
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
            predictionPoseTimestampsFile
                << recordedFrameID << ","
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
                << currentDynamicEdpNonMotionScale << ","
                << currentGroundTruthPositionErrorM << ","
                << currentGroundTruthRotationErrorRad << ","
                << currentDynamicEdpESource << ","
                << currentGroundTruthTargetSource << ","
                << currentGroundTruthTargetPoseInfo.position.x << ","
                << currentGroundTruthTargetPoseInfo.position.y << ","
                << currentGroundTruthTargetPoseInfo.position.z << ","
                << currentPredictionPositionUncertaintyViewM.x << ","
                << currentPredictionPositionUncertaintyViewM.y << ","
                << currentPredictionPositionUncertaintyViewM.z << ","
                << currentGroundTruthPositionErrorViewM.x << ","
                << currentGroundTruthPositionErrorViewM.y << ","
                << currentGroundTruthPositionErrorViewM.z << ","
                << currentViewOffsetUncertaintyM.x << ","
                << currentViewOffsetUncertaintyM.y << ","
                << currentViewOffsetUncertaintyM.z << ","
                << describeViewOffsetMode(currentViewOffsetUncertaintyM) << ","
                << currentMotionProjectionFeatures.cameraFrameTranslation.x << ","
                << currentMotionProjectionFeatures.cameraFrameTranslation.y << ","
                << currentMotionProjectionFeatures.cameraFrameTranslation.z << ","
                << currentMotionProjectionFeatures.projectedMotionPx.x << ","
                << currentMotionProjectionFeatures.projectedMotionPx.y << ","
                << currentMotionProjectionFeatures.projectedMotionMagnitudePx << ","
                << currentMotionProjectionFeatures.representativeDepthM << ","
                << currentMotionEBoostFactor << ","
                << currentScreenSpaceFootprintFeatures.majorRadiusPx << ","
                << currentScreenSpaceFootprintFeatures.minorRadiusPx << ","
                << currentScreenSpaceFootprintFeatures.directionPx.x << ","
                << currentScreenSpaceFootprintFeatures.directionPx.y << ","
                << currentScreenSpaceFootprintFeatures.sampleCount << ","
                << currentViewOffsetSampleCount
                << std::endl;

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

            if (activeRemoteRenderPose.has_value()) {
                const PoseCsvInfo remoteRenderPoseInfo = extractPoseCsvInfoFromViewMatrix(activeRemoteRenderPose->mono.view);
                recvPoseToRenderFile
                    << recordedFrameID << ","
                    << static_cast<int64_t>(activeRemoteRenderPose->send_timestamp) << ","
                    << remoteRenderPoseInfo.position.x << "," << remoteRenderPoseInfo.position.y << "," << remoteRenderPoseInfo.position.z << ","
                    << remoteRenderPoseInfo.rotationQuat.x << "," << remoteRenderPoseInfo.rotationQuat.y << ","
                    << remoteRenderPoseInfo.rotationQuat.z << "," << remoteRenderPoseInfo.rotationQuat.w << ","
                    << remoteRenderPoseInfo.eulerRotationDegrees.x << "," << remoteRenderPoseInfo.eulerRotationDegrees.y << ","
                    << remoteRenderPoseInfo.eulerRotationDegrees.z
                    << std::endl;
            }
            else {
                recvPoseToRenderFile
                    << recordedFrameID << ","
                    << -1 << ","
                    << -1 << "," << -1 << "," << -1 << ","
                    << -1 << "," << -1 << "," << -1 << "," << -1 << ","
                    << -1 << "," << -1 << "," << -1
                    << std::endl;
            }
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
            const int recordedFrameID = frameDumpEnabled ? recorder.getNextFrameID() : nonDumpedRecordedFrameID++;
            dumpPosePredictionAndRecordedPoseCsvRows(recordedFrameID);
            dumpWideFovTexelUsageForCapture(recordedFrameID);
            if (recordedFrameID == dumpNormalRenderDepthStatsFrame) {
                remoteRenderer.drawObjectsNoLighting(remoteScene, camera);
                dumpNormalRenderDepthStatsForRecordedFrame(remoteRenderer, camera, outputPath, recordedFrameID);
            }
            if (frameDumpEnabled) {
                recorder.captureFrame(camera);
            }

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
            if (recordedFrameID == dumpNormalRenderDepthStatsFrame) {
                remoteRenderer.drawObjectsNoLighting(remoteScene, camera);
                dumpNormalRenderDepthStatsForRecordedFrame(remoteRenderer, camera, outputPath, recordedFrameID);
            }
            recorder.captureFrame(camera);
        }
    });

    // Run app loop (blocking)
    app.run();

    return 0;
}
