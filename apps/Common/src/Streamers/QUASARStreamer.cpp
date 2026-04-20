#include "RenderTargets/FrameRenderTarget.h"
#include <Streamers/QUASARStreamer.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using quasar::FileIO;
using quasar::FrameRenderTarget;
using quasar::Path;
using quasar::WideFovMaskMethod;

namespace {

constexpr const char* kWideFovTexelUsageRoot =
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/test/robot_lab";
constexpr const char* kRemoteToRecordedFrameMapCsv =
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/profileCode/imageQualityCorrelation/robot_lab/"
    "quasarTrimWithMaskGT_remote_to_recorded_frame_map.csv";

struct RecordedWideFovMaskFrame {
    int recordedFrameID = -1;
    int recordedFrameIDPNG = -1;
};

using RemoteToRecordedFrameMap = std::unordered_map<uint, std::vector<RecordedWideFovMaskFrame>>;

struct DepthStencilPixel {
    float depth = 1.0f;
    std::uint32_t stencil = 0u;
};

struct ExternalWideFovUsageMask {
    std::string sourcePath;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
    size_t usefulTexels = 0;

    explicit operator bool() const {
        return !rgba.empty();
    }
};

struct PoseDebugInfo {
    bool valid = false;
    glm::vec3 position{0.0f};
    glm::vec3 rotationDegrees{0.0f};
};

/// Shoelace area for quad vertex order (0,1), (1,3), (3,2), (2,0) — matches quad_mask.vert triangle strip.
double narrowReprojectionQuadAreaPx2(const glm::vec3 c[4]) {
    const glm::vec2 v[4] = {
        glm::vec2(c[0]), glm::vec2(c[1]), glm::vec2(c[3]), glm::vec2(c[2]),
    };
    double a = 0.0;
    for (int i = 0; i < 4; i++) {
        const int j = (i + 1) % 4;
        a += static_cast<double>(v[i].x) * static_cast<double>(v[j].y);
        a -= static_cast<double>(v[j].x) * static_cast<double>(v[i].y);
    }
    return std::abs(a) * 0.5;
}

constexpr float kPoseOnlyReprojectionMinDenominator = 1e-4f;
constexpr size_t kMinReliableTrimmedWideFovAlphaTexels = 10000u;

glm::mat3 buildAtwStyleCurrentToPreviousHomography(
    const glm::mat4& currentViewMatrix,
    const glm::mat4& currentViewMatrixInverse,
    const glm::mat4& previousViewMatrix,
    const glm::mat4& currentProjectionMatrix,
    const glm::mat4& previousProjectionMatrix,
    float viewportWidth,
    float viewportHeight)
{
    const glm::mat3 currentCameraToWorldRotation = glm::mat3(currentViewMatrixInverse);
    const glm::mat3 previousWorldToCameraRotation = glm::mat3(previousViewMatrix);
    const glm::mat3 currentToPreviousRotation = previousWorldToCameraRotation * currentCameraToWorldRotation;

    auto buildIntrinsicsFromProjection = [&](const glm::mat4& projectionMatrix) {
        const float fx = 0.5f * viewportWidth * projectionMatrix[0][0];
        const float fy = 0.5f * viewportHeight * projectionMatrix[1][1];
        const float cx = 0.5f * viewportWidth;
        const float cy = 0.5f * viewportHeight;

        glm::mat3 intrinsics(1.0f);
        intrinsics[0][0] = fx;
        intrinsics[1][1] = fy;
        intrinsics[2][0] = cx;
        intrinsics[2][1] = cy;
        return intrinsics;
    };

    const glm::mat3 currentIntrinsics = buildIntrinsicsFromProjection(currentProjectionMatrix);
    const glm::mat3 previousIntrinsics = buildIntrinsicsFromProjection(previousProjectionMatrix);
    return previousIntrinsics * currentToPreviousRotation * glm::inverse(currentIntrinsics);
}

glm::vec3 applyPoseOnlyHomographyToImagePoint(const glm::mat3& homography, const glm::vec3& imagePoint) {
    glm::vec3 warpedPoint = homography * imagePoint;
    if (std::abs(warpedPoint.z) >= kPoseOnlyReprojectionMinDenominator) {
        warpedPoint /= warpedPoint.z;
    }
    else {
        warpedPoint.z = 1.0f;
    }
    return warpedPoint;
}

PoseDebugInfo extractPoseDebugInfo(const glm::mat4& viewMatrix) {
    PoseDebugInfo info;
    glm::vec3 scale;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::quat rotationQuat;
    const glm::mat4 poseMatrix = glm::inverse(viewMatrix);
    if (glm::decompose(poseMatrix, scale, rotationQuat, translation, skew, perspective)) {
        rotationQuat = glm::normalize(rotationQuat);
        info.valid = true;
        info.position = translation;
        info.rotationDegrees = glm::degrees(glm::eulerAngles(rotationQuat));
    }
    return info;
}

void logPoseForDebug(const char* label, const glm::mat4& viewMatrix) {
    const PoseDebugInfo info = extractPoseDebugInfo(viewMatrix);
    if (info.valid) {
        spdlog::info(
            "{} pose: pos=({:.6f}, {:.6f}, {:.6f}), rot_deg=({:.6f}, {:.6f}, {:.6f})",
            label,
            info.position.x,
            info.position.y,
            info.position.z,
            info.rotationDegrees.x,
            info.rotationDegrees.y,
            info.rotationDegrees.z);
    }
    else {
        spdlog::warn("Failed to decompose {} pose for debugging", label);
    }
}

std::string getOutputDirFromWideFovDumpDir(const std::string& wideFovImageDumpDir) {
    if (wideFovImageDumpDir.empty()) {
        return "";
    }
    const size_t slashPos = wideFovImageDumpDir.find_last_of('/');
    if (slashPos == std::string::npos) {
        return wideFovImageDumpDir;
    }
    return wideFovImageDumpDir.substr(0, slashPos);
}

Path getReprojectionMaskCompareDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "reprojection_mask_compare";
}

Path getTrimWideFovTimewarpDebugDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "trim_widefov_timewarp_debug";
}

Path getTrimWideFovAtwDebugDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "trim_widefov_atw_debug";
}

glm::vec3 clampImagePointToViewport(const glm::vec3& imagePoint, float viewportWidth, float viewportHeight) {
    return glm::vec3(
        std::clamp(imagePoint.x, 0.0f, viewportWidth),
        std::clamp(imagePoint.y, 0.0f, viewportHeight),
        1.0f);
}

bool pointInTriangle(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    const glm::vec2 v0 = c - a;
    const glm::vec2 v1 = b - a;
    const glm::vec2 v2 = p - a;

    const float dot00 = glm::dot(v0, v0);
    const float dot01 = glm::dot(v0, v1);
    const float dot02 = glm::dot(v0, v2);
    const float dot11 = glm::dot(v1, v1);
    const float dot12 = glm::dot(v1, v2);

    const float denom = dot00 * dot11 - dot01 * dot01;
    if (std::abs(denom) < 1e-8f) {
        return false;
    }

    const float invDenom = 1.0f / denom;
    const float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f;
}

bool pointInQuadTriangleStrip(const glm::vec2& p, const glm::vec3 quad[4]) {
    return pointInTriangle(p, glm::vec2(quad[0]), glm::vec2(quad[1]), glm::vec2(quad[3]))
        || pointInTriangle(p, glm::vec2(quad[0]), glm::vec2(quad[3]), glm::vec2(quad[2]));
}

std::vector<unsigned char> maskToRgbImage(const std::vector<unsigned char>& mask, unsigned char r, unsigned char g, unsigned char b) {
    std::vector<unsigned char> rgb(mask.size() * 3u, 0u);
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 0u) {
            continue;
        }
        const size_t base = i * 3u;
        rgb[base + 0] = r;
        rgb[base + 1] = g;
        rgb[base + 2] = b;
    }
    return rgb;
}

std::vector<unsigned char> rasterizeQuadMask(int width, int height, const glm::vec3 quad[4]) {
    std::vector<unsigned char> mask(static_cast<size_t>(width) * height, 0u);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec2 pixelCenter(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            if (!pointInQuadTriangleStrip(pixelCenter, quad)) {
                continue;
            }

            const size_t idx = static_cast<size_t>(y) * width + x;
            mask[idx] = 255u;
        }
    }
    return mask;
}

std::vector<unsigned char> subtractBinaryMasks(
    const std::vector<unsigned char>& minuendMask,
    const std::vector<unsigned char>& subtrahendMask)
{
    std::vector<unsigned char> result(minuendMask.size(), 0u);
    for (size_t i = 0; i < minuendMask.size(); ++i) {
        result[i] = (minuendMask[i] != 0u && subtrahendMask[i] == 0u) ? 255u : 0u;
    }
    return result;
}

std::vector<unsigned char> overlayBinaryMasksRgb(
    const std::vector<unsigned char>& currentMask,
    const std::vector<unsigned char>& warpedPreviousMask)
{
    std::vector<unsigned char> rgb(currentMask.size() * 3u, 0u);
    for (size_t i = 0; i < currentMask.size(); ++i) {
        const bool current = currentMask[i] != 0u;
        const bool warpedPrevious = warpedPreviousMask[i] != 0u;
        const size_t base = i * 3u;
        if (current && warpedPrevious) {
            rgb[base + 0] = 255u;
            rgb[base + 1] = 255u;
        }
        else if (current) {
            rgb[base + 0] = 255u;
        }
        else if (warpedPrevious) {
            rgb[base + 1] = 255u;
        }
    }
    return rgb;
}

void dumpTrimWideFovTimewarpDebugImages(
    uint frameID,
    const std::string& wideFovImageDumpDir,
    int normalWidth,
    int normalHeight,
    const glm::vec3 currentNormalQuad[4],
    const glm::vec3 warpedPreviousNormalQuad[4],
    int wideWidth,
    int wideHeight,
    const glm::vec3 currentWideQuad[4],
    const glm::vec3 warpedPreviousWideQuad[4])
{
    if (wideFovImageDumpDir.empty()) {
        return;
    }

    Path dumpDir = getTrimWideFovTimewarpDebugDir(wideFovImageDumpDir);
    dumpDir.mkdirRecursive();

    const std::vector<unsigned char> currentNormalMask =
        rasterizeQuadMask(normalWidth, normalHeight, currentNormalQuad);
    const std::vector<unsigned char> warpedPreviousNormalMask =
        rasterizeQuadMask(normalWidth, normalHeight, warpedPreviousNormalQuad);
    const std::vector<unsigned char> uncoveredNormalMask =
        subtractBinaryMasks(currentNormalMask, warpedPreviousNormalMask);
    const std::vector<unsigned char> normalOverlay =
        overlayBinaryMasksRgb(currentNormalMask, warpedPreviousNormalMask);

    const std::vector<unsigned char> currentWideMask =
        rasterizeQuadMask(wideWidth, wideHeight, currentWideQuad);
    const std::vector<unsigned char> warpedPreviousWideMask =
        rasterizeQuadMask(wideWidth, wideHeight, warpedPreviousWideQuad);
    const std::vector<unsigned char> uncoveredWideMask =
        subtractBinaryMasks(currentWideMask, warpedPreviousWideMask);
    const std::vector<unsigned char> wideOverlay =
        overlayBinaryMasksRgb(currentWideMask, warpedPreviousWideMask);

    const std::string frameSuffix = std::to_string(frameID);
    FileIO::writeToPNG(
        (dumpDir / ("timewarp_prev_to_current_normal_overlay_" + frameSuffix + ".png")).str(),
        normalWidth,
        normalHeight,
        3,
        normalOverlay.data());
    FileIO::writeToPNG(
        (dumpDir / ("timewarp_prev_to_current_normal_uncovered_" + frameSuffix + ".png")).str(),
        normalWidth,
        normalHeight,
        3,
        maskToRgbImage(uncoveredNormalMask, 255u, 255u, 255u).data());
    FileIO::writeToPNG(
        (dumpDir / ("timewarp_prev_to_current_widefov_overlay_" + frameSuffix + ".png")).str(),
        wideWidth,
        wideHeight,
        3,
        wideOverlay.data());
    FileIO::writeToPNG(
        (dumpDir / ("timewarp_prev_to_current_widefov_uncovered_" + frameSuffix + ".png")).str(),
        wideWidth,
        wideHeight,
        3,
        maskToRgbImage(uncoveredWideMask, 255u, 255u, 255u).data());
}

void dumpTrimWideFovAtwReprojectionDebug(
    uint frameID,
    const std::string& wideFovImageDumpDir,
    quasar::DeferredRenderer& renderer,
    quasar::Shader& atwDebugShader,
    quasar::FrameRenderTarget& outputRT,
    quasar::FrameRenderTarget& sourceRT,
    const glm::mat4& sourceViewMatrix,
    const glm::mat4& sourceProjectionMatrix,
    const glm::mat4& targetViewMatrix,
    const glm::mat4& targetProjectionMatrix)
{
    if (wideFovImageDumpDir.empty()) {
        return;
    }

    Path dumpDir = getTrimWideFovAtwDebugDir(wideFovImageDumpDir);
    dumpDir.mkdirRecursive();

    logPoseForDebug("ATW trim source", sourceViewMatrix);
    logPoseForDebug("ATW trim target", targetViewMatrix);

    const PoseDebugInfo sourcePose = extractPoseDebugInfo(sourceViewMatrix);
    const PoseDebugInfo targetPose = extractPoseDebugInfo(targetViewMatrix);
    const Path poseCsvPath = dumpDir / "pose_pairs.csv";
    const bool shouldWriteHeader = !poseCsvPath.exists();
    std::ofstream poseCsv(poseCsvPath.str(), std::ios::app);
    if (shouldWriteHeader) {
        poseCsv << "frame_id,"
                << "source_pos_x,source_pos_y,source_pos_z,"
                << "source_rot_x_deg,source_rot_y_deg,source_rot_z_deg,"
                << "target_pos_x,target_pos_y,target_pos_z,"
                << "target_rot_x_deg,target_rot_y_deg,target_rot_z_deg"
                << std::endl;
    }
    poseCsv << frameID << ","
            << sourcePose.position.x << "," << sourcePose.position.y << "," << sourcePose.position.z << ","
            << sourcePose.rotationDegrees.x << "," << sourcePose.rotationDegrees.y << "," << sourcePose.rotationDegrees.z << ","
            << targetPose.position.x << "," << targetPose.position.y << "," << targetPose.position.z << ","
            << targetPose.rotationDegrees.x << "," << targetPose.rotationDegrees.y << "," << targetPose.rotationDegrees.z
            << std::endl;

    outputRT.bind();
    outputRT.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    outputRT.unbind();

    atwDebugShader.bind();
    atwDebugShader.setBool("atwEnabled", true);
    atwDebugShader.setMat4("projectionInverse", glm::inverse(targetProjectionMatrix));
    atwDebugShader.setMat4("viewInverse", glm::inverse(targetViewMatrix));
    atwDebugShader.setMat4("remoteProjection", sourceProjectionMatrix);
    atwDebugShader.setMat4("remoteView", sourceViewMatrix);
    atwDebugShader.setTexture("videoTexture", sourceRT.colorTexture, 5);
    atwDebugShader.setTexture("depthTexture", sourceRT.depthStencilTexture, 6);
    renderer.drawToRenderTarget(atwDebugShader, outputRT);

    const std::string frameSuffix = std::to_string(frameID);
    sourceRT.writeColorAsPNG((dumpDir / ("source_frame_" + frameSuffix + ".png")).str());
    outputRT.writeColorAsPNG((dumpDir / ("warped_to_effectiveWideFovGt_" + frameSuffix + ".png")).str());
}

void dumpReprojectionMaskComparison(
    uint frameID,
    const std::string& wideFovImageDumpDir,
    int width,
    int height,
    const glm::mat3& currentToPreviousHomography,
    const glm::vec3 coveredCurrentQuad[4],
    const PoseDebugInfo& previousPose,
    const PoseDebugInfo& currentPose)
{
    if (wideFovImageDumpDir.empty()) {
        return;
    }

    Path dumpDir = getReprojectionMaskCompareDir(wideFovImageDumpDir);
    dumpDir.mkdirRecursive();

    std::vector<unsigned char> quasarEstimateMask(static_cast<size_t>(width) * height, 0u);
    std::vector<unsigned char> timewarpMask(static_cast<size_t>(width) * height, 0u);
    std::vector<unsigned char> overlayRgb(static_cast<size_t>(width) * height * 3u, 0u);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec2 pixelCenter(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            const size_t idx = static_cast<size_t>(y) * width + x;

            const bool coveredByQuasarEstimate = pointInQuadTriangleStrip(pixelCenter, coveredCurrentQuad);
            const bool uncoveredByQuasarEstimate = !coveredByQuasarEstimate;
            quasarEstimateMask[idx] = uncoveredByQuasarEstimate ? 255u : 0u;

            const glm::vec3 mappedToPrevious = applyPoseOnlyHomographyToImagePoint(
                currentToPreviousHomography,
                glm::vec3(pixelCenter, 1.0f));
            const bool uncoveredByTimewarp = mappedToPrevious.x < 0.0f
                || mappedToPrevious.x >= static_cast<float>(width)
                || mappedToPrevious.y < 0.0f
                || mappedToPrevious.y >= static_cast<float>(height);
            timewarpMask[idx] = uncoveredByTimewarp ? 255u : 0u;

            const size_t rgbIdx = idx * 3u;
            if (uncoveredByQuasarEstimate && uncoveredByTimewarp) {
                overlayRgb[rgbIdx + 0] = 255u;
                overlayRgb[rgbIdx + 2] = 255u;
            }
            else if (uncoveredByQuasarEstimate) {
                overlayRgb[rgbIdx + 0] = 255u;
            }
            else if (uncoveredByTimewarp) {
                overlayRgb[rgbIdx + 2] = 255u;
            }
        }
    }

    const std::string frameSuffix = std::to_string(frameID);
    FileIO::writeToPNG(
        (dumpDir / ("quasar_estimate_mask_" + frameSuffix + ".png")).str(),
        width,
        height,
        3,
        maskToRgbImage(quasarEstimateMask, 255u, 255u, 255u).data());
    FileIO::writeToPNG(
        (dumpDir / ("timewarp_mask_" + frameSuffix + ".png")).str(),
        width,
        height,
        3,
        maskToRgbImage(timewarpMask, 255u, 255u, 255u).data());
    FileIO::writeToPNG(
        (dumpDir / ("mask_overlay_" + frameSuffix + ".png")).str(),
        width,
        height,
        3,
        overlayRgb.data());

    std::ofstream posePairsCsv((dumpDir / "pose_pairs.csv").str(), std::ios::app);
    posePairsCsv << frameID << ","
                 << previousPose.position.x << "," << previousPose.position.y << "," << previousPose.position.z << ","
                 << previousPose.rotationDegrees.x << "," << previousPose.rotationDegrees.y << "," << previousPose.rotationDegrees.z << ","
                 << currentPose.position.x << "," << currentPose.position.y << "," << currentPose.position.z << ","
                 << currentPose.rotationDegrees.x << "," << currentPose.rotationDegrees.y << "," << currentPose.rotationDegrees.z
                 << std::endl;
}

bool shouldApplyExternalWideFovUsageMask(WideFovMaskMethod wideFovMaskMethod) {
    return wideFovMaskMethod == WideFovMaskMethod::RecordedTexelUsage;
}

bool tryParseCSVInt(const std::string& value, int& parsedValue) {
    if (value.empty()) {
        return false;
    }

    try {
        parsedValue = std::stoi(value);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }

    return fields;
}

RemoteToRecordedFrameMap loadRemoteToRecordedFrameMap() {
    RemoteToRecordedFrameMap remoteToRecordedFrameMap;

    std::ifstream csvFile(kRemoteToRecordedFrameMapCsv);
    if (!csvFile.is_open()) {
        spdlog::warn("Failed to open remote-to-recorded frame map CSV: {}", kRemoteToRecordedFrameMapCsv);
        return remoteToRecordedFrameMap;
    }

    std::string headerLine;
    if (!std::getline(csvFile, headerLine)) {
        spdlog::warn("Remote-to-recorded frame map CSV is empty: {}", kRemoteToRecordedFrameMapCsv);
        return remoteToRecordedFrameMap;
    }

    const std::vector<std::string> headers = splitCSVLine(headerLine);
    auto findColumnIndex = [&](const std::string& columnName) -> int {
        for (size_t idx = 0; idx < headers.size(); ++idx) {
            if (headers[idx] == columnName) {
                return static_cast<int>(idx);
            }
        }
        return -1;
    };

    const int remoteFrameIDColumn = findColumnIndex("remote_frame_id");
    const int recordedFrameIDColumn = findColumnIndex("recorded_frame_id");
    const int recordedFrameIDPNGColumn = findColumnIndex("recorded_frame_id_png");

    if (remoteFrameIDColumn < 0 || recordedFrameIDColumn < 0 || recordedFrameIDPNGColumn < 0) {
        spdlog::warn(
            "Remote-to-recorded frame map CSV is missing required columns: {}",
            kRemoteToRecordedFrameMapCsv);
        return remoteToRecordedFrameMap;
    }

    std::string line;
    while (std::getline(csvFile, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = splitCSVLine(line);
        if (fields.size() < headers.size()) {
            fields.resize(headers.size());
        }

        int remoteFrameID = -1;
        if (!tryParseCSVInt(fields[remoteFrameIDColumn], remoteFrameID) || remoteFrameID < 0) {
            continue;
        }

        RecordedWideFovMaskFrame recordedMaskFrame;
        const bool hasRecordedFrameID = tryParseCSVInt(
            fields[recordedFrameIDColumn],
            recordedMaskFrame.recordedFrameID);
        const bool hasRecordedFrameIDPNG = tryParseCSVInt(
            fields[recordedFrameIDPNGColumn],
            recordedMaskFrame.recordedFrameIDPNG);

        auto& recordedMaskFrames = remoteToRecordedFrameMap[static_cast<uint>(remoteFrameID)];
        if (!hasRecordedFrameID && !hasRecordedFrameIDPNG) {
            continue;
        }

        bool alreadyRecorded = false;
        for (const RecordedWideFovMaskFrame& existingMaskFrame : recordedMaskFrames) {
            if (existingMaskFrame.recordedFrameID == recordedMaskFrame.recordedFrameID
                && existingMaskFrame.recordedFrameIDPNG == recordedMaskFrame.recordedFrameIDPNG)
            {
                alreadyRecorded = true;
                break;
            }
        }

        if (!alreadyRecorded) {
            recordedMaskFrames.push_back(recordedMaskFrame);
        }
    }

    return remoteToRecordedFrameMap;
}

const RemoteToRecordedFrameMap& getRemoteToRecordedFrameMap() {
    static const RemoteToRecordedFrameMap remoteToRecordedFrameMap = loadRemoteToRecordedFrameMap();
    return remoteToRecordedFrameMap;
}

std::vector<Path> getWideFovTexelUsageMaskDirs(const std::string&) {
    std::vector<Path> maskDirs;

    auto appendMaskDirIfPresent = [&](const Path& maskDir) {
        if (!maskDir.exists()) {
            return;
        }

        for (const Path& existingMaskDir : maskDirs) {
            if (existingMaskDir.str() == maskDir.str()) {
                return;
            }
        }

        maskDirs.push_back(maskDir);
    };

    const Path maskRoot{std::string(kWideFovTexelUsageRoot)};
    appendMaskDirIfPresent(maskRoot / "quasarWideFoVUsage" / "widefov_texel_usage");

    return maskDirs;
}

bool loadExternalWideFovUsageMaskImage(
    const Path& maskPath,
    uint expectedWidth,
    uint expectedHeight,
    ExternalWideFovUsageMask& outMask)
{
    if (!maskPath.exists()) {
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    FileIO::flipVerticallyOnLoad(true);
    unsigned char* maskData = FileIO::loadImage(maskPath.str(), &width, &height, &channels, 4);
    FileIO::flipVerticallyOnLoad(false);

    if (maskData == nullptr) {
        spdlog::warn("Failed to load wide-FOV texel-usage mask: {}", maskPath.str());
        return false;
    }

    if (static_cast<uint>(width) != expectedWidth || static_cast<uint>(height) != expectedHeight) {
        spdlog::warn(
            "Wide-FOV texel-usage mask {} has unexpected size {}x{} (expected {}x{}); skipping it",
            maskPath.str(),
            width,
            height,
            expectedWidth,
            expectedHeight);
        FileIO::freeImage(maskData);
        return false;
    }

    outMask.sourcePath = maskPath.str();
    outMask.width = width;
    outMask.height = height;
    outMask.rgba.assign(maskData, maskData + (static_cast<size_t>(width) * height * 4u));
    outMask.usefulTexels = 0;
    for (size_t px = 0; px < static_cast<size_t>(width) * height; ++px) {
        const size_t offset = px * 4u;
        if (outMask.rgba[offset + 0] != 0u
            || outMask.rgba[offset + 1] != 0u
            || outMask.rgba[offset + 2] != 0u
            || outMask.rgba[offset + 3] != 0u)
        {
            outMask.usefulTexels++;
        }
    }

    FileIO::freeImage(maskData);
    return true;
}

size_t countUsefulTexels(const std::vector<unsigned char>& rgba) {
    size_t usefulTexels = 0;
    for (size_t px = 0; px + 3 < rgba.size(); px += 4u) {
        if (rgba[px + 0] != 0u
            || rgba[px + 1] != 0u
            || rgba[px + 2] != 0u
            || rgba[px + 3] != 0u)
        {
            usefulTexels++;
        }
    }
    return usefulTexels;
}

void unionExternalWideFovUsageMask(ExternalWideFovUsageMask& unionMask, const ExternalWideFovUsageMask& maskToUnion) {
    if (!maskToUnion) {
        return;
    }

    if (!unionMask) {
        unionMask = maskToUnion;
        return;
    }

    if (unionMask.width != maskToUnion.width || unionMask.height != maskToUnion.height) {
        spdlog::warn(
            "Cannot union wide-FOV texel-usage masks with mismatched sizes: {}x{} vs {}x{}",
            unionMask.width,
            unionMask.height,
            maskToUnion.width,
            maskToUnion.height);
        return;
    }

    if (!unionMask.sourcePath.empty()) {
        unionMask.sourcePath += " | ";
    }
    unionMask.sourcePath += maskToUnion.sourcePath;

    for (size_t px = 0; px < unionMask.rgba.size(); px += 4u) {
        const bool sourceMaskHasTexel = maskToUnion.rgba[px + 0] != 0u
            || maskToUnion.rgba[px + 1] != 0u
            || maskToUnion.rgba[px + 2] != 0u
            || maskToUnion.rgba[px + 3] != 0u;
        if (!sourceMaskHasTexel) {
            continue;
        }

        unionMask.rgba[px + 0] = maskToUnion.rgba[px + 0];
        unionMask.rgba[px + 1] = maskToUnion.rgba[px + 1];
        unionMask.rgba[px + 2] = maskToUnion.rgba[px + 2];
        unionMask.rgba[px + 3] = maskToUnion.rgba[px + 3];
    }

    unionMask.usefulTexels = countUsefulTexels(unionMask.rgba);
}

void addNormalViewRectToExternalWideFovUsageMask(
    ExternalWideFovUsageMask& mask,
    const glm::vec3 normalViewCornersInWideFoVImage[4])
{
    if (!mask) {
        return;
    }

    float minX = normalViewCornersInWideFoVImage[0].x;
    float maxX = normalViewCornersInWideFoVImage[0].x;
    float minY = normalViewCornersInWideFoVImage[0].y;
    float maxY = normalViewCornersInWideFoVImage[0].y;
    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, normalViewCornersInWideFoVImage[i].x);
        maxX = std::max(maxX, normalViewCornersInWideFoVImage[i].x);
        minY = std::min(minY, normalViewCornersInWideFoVImage[i].y);
        maxY = std::max(maxY, normalViewCornersInWideFoVImage[i].y);
    }

    const int startX = std::clamp(static_cast<int>(std::floor(minX)), 0, mask.width);
    const int endX = std::clamp(static_cast<int>(std::ceil(maxX)), 0, mask.width);
    const int startY = std::clamp(static_cast<int>(std::floor(minY)), 0, mask.height);
    const int endY = std::clamp(static_cast<int>(std::ceil(maxY)), 0, mask.height);

    if (startX >= endX || startY >= endY) {
        spdlog::warn(
            "Normal-view rect projected into wide-FOV image is empty after clamping: x=[{}, {}), y=[{}, {})",
            startX,
            endX,
            startY,
            endY);
        return;
    }

    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(mask.width)
                                   + static_cast<size_t>(x))
                * 4u;
            mask.rgba[offset + 0] = 255u;
            mask.rgba[offset + 1] = 255u;
            mask.rgba[offset + 2] = 255u;
            mask.rgba[offset + 3] = 255u;
        }
    }

    if (!mask.sourcePath.empty()) {
        mask.sourcePath += " | ";
    }
    mask.sourcePath += "normal-view rect";
    mask.usefulTexels = countUsefulTexels(mask.rgba);
}

ExternalWideFovUsageMask createEmptyExternalWideFovUsageMask(
    uint expectedWidth,
    uint expectedHeight,
    const std::string& sourceDescription)
{
    ExternalWideFovUsageMask emptyMask;
    emptyMask.sourcePath = sourceDescription;
    emptyMask.width = static_cast<int>(expectedWidth);
    emptyMask.height = static_cast<int>(expectedHeight);
    emptyMask.rgba.assign(static_cast<size_t>(expectedWidth) * expectedHeight * 4u, 0u);
    emptyMask.usefulTexels = 0;
    return emptyMask;
}

bool loadExternalWideFovUsageMaskForRemoteFrame(
    uint remoteFrameID,
    const std::string& wideFovImageDumpDir,
    uint expectedWidth,
    uint expectedHeight,
    const glm::vec3 normalViewCornersInWideFoVImage[4],
    ExternalWideFovUsageMask& outMask)
{
    const RemoteToRecordedFrameMap& remoteToRecordedFrameMap = getRemoteToRecordedFrameMap();
    const auto remoteFrameIt = remoteToRecordedFrameMap.find(remoteFrameID);
    if (remoteFrameIt == remoteToRecordedFrameMap.end()) {
        spdlog::info(
            "Remote frame {} has no mapping; skipping external wide-FOV mask for this frame",
            remoteFrameID);
        return false;
    }

    const std::vector<RecordedWideFovMaskFrame>& recordedMaskFrames = remoteFrameIt->second;
    if (recordedMaskFrames.empty()) {
        spdlog::info(
            "Remote frame {} has no recorded frames; skipping external wide-FOV mask for this frame",
            remoteFrameID);
        return false;
    }

    const std::vector<Path> maskDirs = getWideFovTexelUsageMaskDirs(wideFovImageDumpDir);
    if (maskDirs.empty()) {
        spdlog::warn("No wide-FOV texel-usage directories are available for remote frame {}", remoteFrameID);
        return false;
    }

    bool loadedAnyMask = false;
    for (const RecordedWideFovMaskFrame& recordedMaskFrame : recordedMaskFrames) {
        std::vector<int> candidateMaskIDs;
        if (recordedMaskFrame.recordedFrameIDPNG >= 0) {
            candidateMaskIDs.push_back(recordedMaskFrame.recordedFrameIDPNG);
        }
        if (recordedMaskFrame.recordedFrameID > 0) {
            bool alreadyAdded = false;
            for (int candidateMaskID : candidateMaskIDs) {
                if (candidateMaskID == recordedMaskFrame.recordedFrameID) {
                    alreadyAdded = true;
                    break;
                }
            }
            if (!alreadyAdded) {
                candidateMaskIDs.push_back(recordedMaskFrame.recordedFrameID);
            }
        }

        bool loadedMaskForRecordedFrame = false;
        for (const Path& maskDir : maskDirs) {
            for (int candidateMaskID : candidateMaskIDs) {
                ExternalWideFovUsageMask maskForRecordedFrame;
                const Path maskPath = maskDir / ("widefov_texel_usage_" + std::to_string(candidateMaskID) + ".png");
                if (!loadExternalWideFovUsageMaskImage(
                        maskPath,
                        expectedWidth,
                        expectedHeight,
                        maskForRecordedFrame))
                {
                    continue;
                }

                unionExternalWideFovUsageMask(outMask, maskForRecordedFrame);
                loadedAnyMask = true;
                loadedMaskForRecordedFrame = true;
                break;
            }

            if (loadedMaskForRecordedFrame) {
                break;
            }
        }

        if (!loadedMaskForRecordedFrame) {
            spdlog::warn(
                "Failed to find a wide-FOV texel-usage mask for remote frame {} (recorded_frame_id={}, recorded_frame_id_png={})",
                remoteFrameID,
                recordedMaskFrame.recordedFrameID,
                recordedMaskFrame.recordedFrameIDPNG);
        }
    }

    if (loadedAnyMask) {
        addNormalViewRectToExternalWideFovUsageMask(outMask, normalViewCornersInWideFoVImage);
        outMask.usefulTexels = countUsefulTexels(outMask.rgba);
    }

    return loadedAnyMask;
}

size_t applyExternalWideFovUsageMask(FrameRenderTarget& frameRT, const ExternalWideFovUsageMask& mask) {
    if (!mask) {
        return 0;
    }

    if (static_cast<uint>(mask.width) != frameRT.width || static_cast<uint>(mask.height) != frameRT.height) {
        spdlog::warn(
            "Wide-FOV texel-usage mask size {}x{} does not match render target size {}x{}",
            mask.width,
            mask.height,
            frameRT.width,
            frameRT.height);
        return 0;
    }

    const size_t pixelCount = static_cast<size_t>(frameRT.width) * frameRT.height;

    std::vector<float> colorData(pixelCount * frameRT.colorTexture.channels);
    frameRT.colorTexture.readPixels(reinterpret_cast<unsigned char*>(colorData.data()), true);

    std::vector<unsigned char> alphaData(pixelCount * frameRT.alphaTexture.channels);
    frameRT.alphaTexture.readPixels(alphaData.data(), false);

    std::vector<DepthStencilPixel> depthStencilData(pixelCount);
    frameRT.bind();
    glReadPixels(
        0,
        0,
        frameRT.width,
        frameRT.height,
        GL_DEPTH_STENCIL,
        GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
        depthStencilData.data());
    frameRT.unbind();

    size_t trimmedTexels = 0;
    for (size_t px = 0; px < pixelCount; ++px) {
        const size_t maskOffset = px * 4u;
        const bool keepTexel = mask.rgba[maskOffset + 0] != 0u
            || mask.rgba[maskOffset + 1] != 0u
            || mask.rgba[maskOffset + 2] != 0u
            || mask.rgba[maskOffset + 3] != 0u;
        if (keepTexel) {
            continue;
        }

        trimmedTexels++;

        const size_t colorOffset = px * frameRT.colorTexture.channels;
        for (uint channel = 0; channel < frameRT.colorTexture.channels; ++channel) {
            colorData[colorOffset + channel] = 0.0f;
        }

        alphaData[px] = 0u;
        depthStencilData[px].depth = 1.0f;
        depthStencilData[px].stencil = 0u;
    }

    frameRT.colorTexture.loadFromData(colorData.data());
    frameRT.alphaTexture.loadFromData(alphaData.data());
    frameRT.depthStencilTexture.loadFromData(depthStencilData.data());

    return trimmedTexels;
}

size_t countNonZeroAlphaTexels(FrameRenderTarget& frameRT) {
    const size_t pixelCount = static_cast<size_t>(frameRT.width) * frameRT.height;
    std::vector<unsigned char> alphaData(pixelCount * frameRT.alphaTexture.channels);
    frameRT.alphaTexture.readPixels(alphaData.data(), false);

    size_t nonZeroAlphaTexels = 0;
    for (size_t px = 0; px < pixelCount; ++px) {
        const size_t alphaOffset = px * frameRT.alphaTexture.channels;
        bool hasAlpha = false;
        for (uint channel = 0; channel < frameRT.alphaTexture.channels; ++channel) {
            if (alphaData[alphaOffset + channel] != 0u) {
                hasAlpha = true;
                break;
            }
        }
        if (hasAlpha) {
            nonZeroAlphaTexels++;
        }
    }

    return nonZeroAlphaTexels;
}

} // namespace

using namespace quasar;

QUASARStreamer::QUASARStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QUASARStreamerCreateParams& params)
    : quadSet(quadSet)
    , videoURL(params.videoURL)
    , proxiesURL(params.proxiesURL)
    , wideFovImageDumpDir(params.wideFovImageDumpDir)
    , maxLayers(params.maxLayers)
    , wideFovPoseLagFrames(params.wideFovPoseLagFrames)
    , wideFovUpdatePeriodFrames(params.wideFovUpdatePeriodFrames != 0u ? params.wideFovUpdatePeriodFrames : 1u)
    , wideFovMaskMethod(params.wideFovMaskMethod)
    , trimWideFov(params.trimWideFov)
    , useWideFovGroundTruth(params.useWideFovGroundTruth)
    , remoteRenderer(remoteRenderer)
    , remoteRendererDP(remoteRendererDP)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
    , frameGenerator(quadSet)
    , referenceFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , referenceFrameRT_noTone({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameMaskRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualFrameRT_noTone({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , videoAtlasStreamerRT({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.videoURL, params.targetFramerate, params.targetBitRate)
    , alphaAtlasRT({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_R8,
        .format = GL_RED,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , quadMaskShader({
        .vertexCodeData = SHADER_COMMON_QUAD_MASK_VERT,
        .vertexCodeSize = SHADER_COMMON_QUAD_MASK_VERT_len,
        .fragmentCodeData = SHADER_COMMON_QUAD_MASK_FRAG,
        .fragmentCodeSize = SHADER_COMMON_QUAD_MASK_FRAG_len,
    })
    , atwDebugShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_COMMON_ATW_FRAG,
        .fragmentCodeSize = SHADER_COMMON_ATW_FRAG_len,
    })
    , debugMaskRT({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , alphaCodec(alphaAtlasRT.width, alphaAtlasRT.height)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture, residualFrameRT_noTone.alphaTexture)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(params.proxiesURL)
{
    if (wideFovMaskMethod == WideFovMaskMethod::Stencil) {
        trimWideFov = true;
    }
    else {
        trimWideFov = false;
    }

    if (wideFovUpdatePeriodFrames > 1u && trimWideFov) {
        trimWideFov = false;
    }

    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    wideFovNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    referenceFrames.resize(maxLayers);
    geometryMetadatas.resize(maxLayers);

    uint numHidLayers = maxLayers - 1;
    frameRTsHidLayer.reserve(numHidLayers);
    frameRTsHidLayer_noTone.reserve(numHidLayers);
    meshesHidLayer.reserve(numHidLayers);
    depthMeshesHidLayer.reserve(numHidLayers);
    nodesHidLayer.reserve(numHidLayers);
    wireframesHidLayer.reserve(numHidLayers);
    depthNodesHidLayer.reserve(numHidLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    remoteCameraWideFOV.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
    wideFovReuseViewMatrix = remoteCamera.getViewMatrix();

    // Setup hidden layers and wide fov RTs
    RenderTargetCreateParams rtParams = {
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    };
    for (int layer = 0; layer < numHidLayers; layer++) {
        frameRTsHidLayer.emplace_back(rtParams);
        frameRTsHidLayer_noTone.emplace_back(rtParams);
    }

    // Setup visible layer for reference frame
    for (int i = 0; i < meshScenes.size(); i++) {
        referenceFrameMeshes.emplace_back(
            quadSet, referenceFrameRT_noTone.colorTexture, referenceFrameRT_noTone.alphaTexture);

        referenceFrameNodes.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&referenceFrameNodes[i]);

        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodesLocal[i].frustumCulled = false;

        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameWireframesLocal[i].frustumCulled = false;
        referenceFrameWireframesLocal[i].wireframe = true;
        referenceFrameWireframesLocal[i].visible = false;
        referenceFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
    }

    // Setup masks for residual frame
    residualFrameNode.addEntity(&residualFrameMesh);
    residualFrameNode.frustumCulled = false;
    residualFrameNodeLocal.addEntity(&residualFrameMesh);
    residualFrameNodeLocal.frustumCulled = false;

    residualFrameWireframeLocal.addEntity(&residualFrameMesh);
    residualFrameWireframeLocal.frustumCulled = false;
    residualFrameWireframeLocal.wireframe = true;
    residualFrameWireframeLocal.visible = false;
    residualFrameWireframeLocal.overrideMaterial = &maskWireframeMaterial;

    // Setup depth mesh
    depthNode.addEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    for (int layer = 0; layer < numHidLayers; layer++) {
        meshesHidLayer.emplace_back(
            quadSet, 
            frameRTsHidLayer_noTone[layer].colorTexture, 
            frameRTsHidLayer_noTone[layer].alphaTexture);
        if (layer == numHidLayers - 1) {
            // Increase expand amount by 3px for wide FOV
            // This makes it so that we can merge more and still cover holes
            meshesHidLayer[layer].setExpandQuadAmount(3.0f);
        }

        nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
        nodesHidLayer[layer].frustumCulled = false;

        const glm::vec4& color = colors[(layer + 1) % colors.size()];

        wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
        wireframesHidLayer[layer].frustumCulled = false;
        wireframesHidLayer[layer].wireframe = true;
        wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

        depthMeshesHidLayer.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        depthNodesHidLayer.emplace_back(&depthMeshesHidLayer[layer]);
        depthNodesHidLayer[layer].frustumCulled = false;
        depthNodesHidLayer[layer].visible = false;
        depthNodesHidLayer[layer].primitiveType = GL_POINTS;
    }

    if (numHidLayers > 0) {
        const uint wideLayerIdx = numHidLayers - 1;
        wideFovQuadTexelUsageMaterial = std::make_unique<QuadTexelUsageMaterial>(QuadMaterialCreateParams{
            .baseColorTexture = &frameRTsHidLayer_noTone[wideLayerIdx].colorTexture,
            .alphaTexture = &frameRTsHidLayer_noTone[wideLayerIdx].alphaTexture,
        });
        nodesHidLayer[wideLayerIdx].overrideMaterial = wideFovQuadTexelUsageMaterial.get();
    }

    // Setup scene to use as mask for wide fov camera
    for (int i = 0; i < meshScenes.size(); i++) {
        wideFovNodes.emplace_back(&referenceFrameMeshes[i]);
        wideFovNodes[i].frustumCulled = false;
        sceneWideFov.addChildNode(&wideFovNodes[i]);
    }
    for (int i = 0; i < numHidLayers - 1; i++) {
        sceneWideFov.addChildNode(&nodesHidLayer[i]);
    }
    sceneWideFov.addChildNode(&residualFrameNode);

    alphaImageData.resize(alphaAtlasRT.width * alphaAtlasRT.height);

    setViewSphereDiameter(params.viewSphereDiameter);

    if (!videoURL.empty() && !proxiesURL.empty()) {
        spdlog::info("Created QUASARStreamer that sends to URL: tcp://{}", proxiesURL);
    }

    const std::string outputDir = getOutputDirFromWideFovDumpDir(wideFovImageDumpDir);

    quasarStatsCSVFileName = outputDir + "/quasar_stats.csv";
    quasarStatsCSVFile.open(quasarStatsCSVFileName);
    quasarStatsCSVFile << "frame_id";
    quasarStatsCSVFile << ",visible_render";
    for (int layer = 0; layer < maxLayers; layer++) {
        quasarStatsCSVFile << ",layer_" << layer << "_create_proxies";
        quasarStatsCSVFile << ",layer_" << layer << "_compress";
        quasarStatsCSVFile << ",layer_" << layer << "_create_mesh";
    }
    quasarStatsCSVFile << ",total_compress" << std::endl;
    quasarStatsCSVFile.close();

    bandwidthstats.proxy_size_by_layer.resize(maxLayers);
    bandwidthstats.depth_offset_size_by_layer.resize(maxLayers);
    bandwidthstats.alphaSize = 0;
    bandwidthstats.totalSize = 0;
    
    // add output dir to bandwidth stats

    bandwidthStatsCSVFileName = outputDir + "/quasar_streamer_bitrate.csv";
    bandwidthStatsCSVFile.open(bandwidthStatsCSVFileName);
    bandwidthStatsCSVFile << "frameID";
    bandwidthStatsCSVFile << ",atlas_bitrate";
    bandwidthStatsCSVFile << ",depth_peeling_bitrate";
    // Add for bandwidth reference of each layer, above information is enough
    for (int layer = 0; layer < maxLayers; layer++) {
        bandwidthStatsCSVFile << ",layer_" << layer << "_proxy_size";
        bandwidthStatsCSVFile << ",layer_" << layer << "_depth_offset_size";
    }
    bandwidthStatsCSVFile << std::endl;
    bandwidthStatsCSVFile.close();

    if (trimWideFov && !outputDir.empty()) {
        Path reprojectionMaskCompareDir = getReprojectionMaskCompareDir(wideFovImageDumpDir);
        reprojectionMaskCompareDir.mkdirRecursive();
        std::ofstream posePairsCsv((reprojectionMaskCompareDir / "pose_pairs.csv").str());
        posePairsCsv << "frame_id,"
                     << "prev_pos_x,prev_pos_y,prev_pos_z,"
                     << "prev_rot_x_deg,prev_rot_y_deg,prev_rot_z_deg,"
                     << "curr_pos_x,curr_pos_y,curr_pos_z,"
                     << "curr_rot_x_deg,curr_rot_y_deg,curr_rot_z_deg"
                     << std::endl;
    }

    // Given the projection matrix of wideFov and the normal projection matrix, 
    // we can pre-compute the corners of the normal view space in the wide fov space
    glm::mat4 wideFovProjectionMatrix = remoteCameraWideFOV.getProjectionMatrix();
    glm::mat4 normalProjectionMatrix = remoteCamera.getProjectionMatrix();
    glm::vec3 corners[4] = {
        glm::vec3(0, 0, 1.0),
        glm::vec3(quadSet.getSize().x, 0, 1.0),
        glm::vec3(0, quadSet.getSize().y, 1.0),
        glm::vec3(quadSet.getSize().x, quadSet.getSize().y, 1.0),
    };
    for (int i = 0; i < 4; i++) {
        corners[i] = glm::vec3(corners[i].x, corners[i].y, corners[i].z);
        corners[i] = glm::unProject(
            corners[i], 
            glm::mat4(1.0f), 
            normalProjectionMatrix,
            glm::vec4(0.0f, 0.0f, remoteRenderer.width, remoteRenderer.height));
        normalViewCornersInWideFoVImage[i] = glm::project(
            corners[i],
            glm::mat4(1.0f), 
            wideFovProjectionMatrix,
            glm::vec4(0.0f, 0.0f, remoteRenderer.width, remoteRenderer.height));
    }
    spdlog::info("Precomputed normal view corners in wide FOV image space:");
    for (int i = 0; i < 4; i++) {
        spdlog::info("  Corner {}: ({:.3f}, {:.3f})", i, normalViewCornersInWideFoVImage[i].x, normalViewCornersInWideFoVImage[i].y);
    }
}

QUASARStreamer::~QUASARStreamer() {
    videoAtlasStreamerRT.stop();
}

uint QUASARStreamer::getNumTriangles() const {
    int currMeshIndex  = meshIndex % 2;
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    uint numTriangles = refMeshSizes.numIndices / 3; // Each triangle has 3 indices
    for (const auto& mesh : meshesHidLayer) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QUASARStreamer::setDrawState(QuadMesh::DrawState drawState) {
    for (auto& mesh : referenceFrameMeshes) {
        mesh.setDrawState(drawState);
    }
    residualFrameMesh.setDrawState(drawState);
    for (auto& mesh : meshesHidLayer) {
        mesh.setDrawState(drawState);
    }
}

void QUASARStreamer::addMeshesToScene(Scene& localScene) {
    // Add in reverse order to have correct layering
    for (int layer = nodesHidLayer.size() - 1; layer >= 0; layer--) {
        localScene.addChildNode(&nodesHidLayer[layer]);
        localScene.addChildNode(&wireframesHidLayer[layer]);
        localScene.addChildNode(&depthNodesHidLayer[layer]);
    }

    for (int i = 0; i < meshScenes.size(); i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNodeLocal);
    localScene.addChildNode(&residualFrameWireframeLocal);
    localScene.addChildNode(&depthNode);
}

void QUASARStreamer::setViewSphereDiameter(float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;
    remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);
}

RenderStats QUASARStreamer::generateFrame(
    bool createResidualFrame,
    bool showNormals,
    bool showDepth,
    const glm::mat4* wideFovGroundTruthView) {
    const glm::mat4* const effectiveWideFovGt =
        (useWideFovGroundTruth && wideFovGroundTruthView != nullptr) ? wideFovGroundTruthView : nullptr;

    // Reset stats
    Stats prevStats = stats;
    stats = { 0 };
    stats.frameSize = prevStats.frameSize; // Keep previous frame size

    // Draw all meshes for proper masking
    setDrawState(QuadMesh::DrawState::BOTH);

    int currMeshIndex  = meshIndex % 2;
    int prevMeshIndex  = (meshIndex + 1) % 2;

    // Wide-FOV pose lag: ring buffer of remote views; pickIndex selects the lagged view when we refresh the snapshot.
    wideFovCameraViewHistory.push_back(remoteCamera.getViewMatrix());
    const size_t lag = static_cast<size_t>(wideFovPoseLagFrames);
    const size_t maxHistory = lag + 1u;
    while (wideFovCameraViewHistory.size() > maxHistory) {
        wideFovCameraViewHistory.pop_front();
    }

    auto quadsGenerator = frameGenerator.getQuadsGenerator();

    /*
    ============================
    Render scene normally to create Reference Frame textures
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);
    stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    
    frameID++;
    spdlog::info("Frame ID: {}", frameID);
    // open file 
    quasarStatsCSVFile.open(quasarStatsCSVFileName, std::ios::app);
    quasarStatsCSVFile << frameID;
    quasarStatsCSVFile << "," << stats.totalRenderTimeMs;

    const uint wideFovFrameId = static_cast<uint>(frameID);
    const bool renderWideFovThisFrame =
        (wideFovFrameId % wideFovUpdatePeriodFrames == 0u)
        || (maxLayers >= 2 && referenceFrames[maxLayers - 1].getTotalNumQuads() == 0);
    spdlog::info("Render wide FOV this frame: {}, frameID: {}, wideFovFrameId: {}, useTrimmedWideFov: {}", renderWideFovThisFrame, frameID, wideFovFrameId, trimWideFov);
    for (int layer = 0; layer < maxLayers; layer++) {
        int hiddenLayerIndex = layer - 1;
        const bool isWideFovLayer = (layer == maxLayers - 1);
        const bool refreshWideFovLayerQuads =
            !isWideFovLayer || maxLayers < 2 || renderWideFovThisFrame;

        auto& remoteCameraToUse = (layer == 0 && createResidualFrame)
                                    ? remoteCameraPrev
                                    : ((layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV);

        auto& renderTargetToUse        = (layer == 0) ? referenceFrameRT        : frameRTsHidLayer[hiddenLayerIndex];
        auto& renderTargetToUse_noTone = (layer == 0) ? referenceFrameRT_noTone : frameRTsHidLayer_noTone[hiddenLayerIndex];

        auto& meshToUse      = (layer == 0) ? referenceFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenLayerIndex];
        auto& meshToUseDepth = (layer == 0) ? depthMesh                           : depthMeshesHidLayer[hiddenLayerIndex];
        const bool useExternalWideFovUsageMask = isWideFovLayer && shouldApplyExternalWideFovUsageMask(wideFovMaskMethod);
        ExternalWideFovUsageMask externalWideFovUsageMask;
        bool hasExternalWideFovUsageMask = false;

        startTime = timeutils::getTimeMicros();
        if (layer == 0) {
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
            remoteRenderer.copyToFrameRT(renderTargetToUse);
            // if (trimWideFov && effectiveWideFovGt != nullptr && !wideFovImageDumpDir.empty()) {
            //     dumpTrimWideFovAtwReprojectionDebug(
            //         static_cast<uint>(frameID),
            //         wideFovImageDumpDir,
            //         remoteRenderer,
            //         atwDebugShader,
            //         debugMaskRT,
            //         renderTargetToUse,
            //         remoteCameraToUse.getViewMatrix(),
            //         remoteCameraToUse.getProjectionMatrix(),
            //         *effectiveWideFovGt,
            //         remoteCamera.getProjectionMatrix());
            // }
        }
        else if (layer < maxLayers - 1) {
            // Hidden layers need to use the noTone render targets to generate quads for some reason...
            remoteRendererDP.peelingLayers[hiddenLayerIndex+1].blit(renderTargetToUse_noTone);
            // renderTargetToUse_noTone.writeColorAsPNG("quasar_hid_layer_no_tone_" + std::to_string(layer) + ".png");
        }
        // Wide fov camera
        else if (renderWideFovThisFrame) {
            
            // Draw old center mesh at new remoteCamera layer, filling stencil buffer with 1
            glm::mat4 prevViewMatrix = effectiveWideFovGt != nullptr
                ? remoteCamera.getViewMatrix()
                : remoteCameraPrev.getViewMatrix();
            glm::mat4 prevViewMatrixInverse = glm::inverse(prevViewMatrix);
            // glm::mat4 prevViewMatrix = remoteCameraPrev.getViewMatrix();
            glm::mat4 prevProjectionMatrix = remoteCameraPrev.getProjectionMatrix();
            
            glm::mat4 currentViewMatrix = effectiveWideFovGt != nullptr
                ? *effectiveWideFovGt
                : remoteCamera.getViewMatrix();
            glm::mat4 currentProjectionMatrix = remoteCamera.getProjectionMatrix();

            const PoseDebugInfo previousPoseDebug = extractPoseDebugInfo(prevViewMatrix);
            const PoseDebugInfo currentPoseDebug = extractPoseDebugInfo(currentViewMatrix);
            logPoseForDebug("Previous", prevViewMatrix);
            logPoseForDebug("Current", currentViewMatrix);
            spdlog::info("Current viewport size: ({}, {})", quadSet.getSize().x, quadSet.getSize().y);


            glm::mat4 currentViewMatrixInverse = glm::inverse(currentViewMatrix);
            glm::mat4 currentProjectionMatrixInverse = glm::inverse(currentProjectionMatrix);
            
            // Keep the wide-FOV proxy generation camera aligned with the pose that produced the
            // wide-FOV render target; otherwise the dumped wide-FOV image can be correct while
            // the reconstructed wide-FOV quads still project into the wrong part of the final frame.
            remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

            glm::mat4 remoteCameraWideFoVViewMatrix = remoteCameraWideFOV.getViewMatrix();
            glm::mat4 remoteCameraWideFoVProjectionMatrix = remoteCameraWideFOV.getProjectionMatrix();
        
            // if (totalBlackArea > 10000.0f) {
            if (trimWideFov) {
            const glm::vec2 viewportSize(remoteRenderer.width, remoteRenderer.height);
            const glm::vec4 normalViewport(0.0f, 0.0f, quadSet.getSize().x, quadSet.getSize().y);
            const glm::vec3 normalViewportCorners[4] = {
                glm::vec3(-1.0f, -1.0f, 1.0f),
                glm::vec3(1.0f, -1.0f, 1.0f),
                glm::vec3(-1.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 1.0f, 1.0f),
            };

            // compute the transformation matrix
            glm::vec2 timewarpedPreviousCornersInNormalView[4];
            glm::vec2 timewarpedPreviousCornersInWideFov[4];
            for (int i = 0; i < 4; i++) {
                glm::vec3 corner = normalViewportCorners[i];

                // we assume the corners are in the new frame (currentView)
                // project it back to the previous frame's coordinate to see whether it is visible in the previous frame
                glm::vec4 cornerInWorld = currentViewMatrixInverse * currentProjectionMatrixInverse * glm::vec4(corner, 1.0f);
                
                glm::vec4 reprojectedCorner = prevProjectionMatrix * prevViewMatrix * cornerInWorld;

                spdlog::info(
                    "Corner {}: world=({:.3f}, {:.3f}, {:.3f}), reprojected=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
                    i,
                    cornerInWorld.x, cornerInWorld.y, cornerInWorld.z,
                    reprojectedCorner.x, reprojectedCorner.y, reprojectedCorner.z, reprojectedCorner.w);
                glm::vec4 reprojectedCornerNDC = reprojectedCorner/reprojectedCorner.w;

                reprojectedCornerNDC.x = (reprojectedCornerNDC.x + 1.0f) * 0.5f * quadSet.getSize().x;
                reprojectedCornerNDC.y = (reprojectedCornerNDC.y + 1.0f) * 0.5f * quadSet.getSize().y;
                timewarpedPreviousCornersInNormalView[i] = glm::vec2{reprojectedCornerNDC.x, reprojectedCornerNDC.y};

                glm::vec4 reprojectedCornerInWideFov = remoteCameraWideFoVProjectionMatrix * prevViewMatrix * cornerInWorld;

                spdlog::info(
                    "Corner {}: reprojected widefov=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
                    i,
                    reprojectedCornerInWideFov.x, reprojectedCornerInWideFov.y, 
                    reprojectedCornerInWideFov.z, reprojectedCornerInWideFov.w);
                glm::vec4 reprojectedCornerInWideFovNDC = reprojectedCornerInWideFov / reprojectedCornerInWideFov.w;

                reprojectedCornerInWideFovNDC.x = (reprojectedCornerInWideFovNDC.x + 1.0f) * 0.5f * quadSet.getSize().x;
                reprojectedCornerInWideFovNDC.y = (reprojectedCornerInWideFovNDC.y + 1.0f) * 0.5f * quadSet.getSize().y;
                timewarpedPreviousCornersInWideFov[i] = glm::vec2{reprojectedCornerInWideFovNDC.x, reprojectedCornerInWideFovNDC.y};
            }

            spdlog::info("Timewarped previous corners projected into normal view:");
            for (int i = 0; i < 4; ++i) {
                const glm::vec2& normalCorner = timewarpedPreviousCornersInNormalView[i];
                spdlog::info(
                    "  Corner {}: raw=({:.3f}, {:.3f})",
                    i,
                    normalCorner.x,
                    normalCorner.y);
            }

            spdlog::info("Timewarped previous corners projected into wide FOV:");
            for (int i = 0; i < 4; ++i) {
                const glm::vec2& wideFovCorner = timewarpedPreviousCornersInWideFov[i];
                spdlog::info(
                    "  Corner {}: raw=({:.3f}, {:.3f})",
                    i,
                    wideFovCorner.x,
                    wideFovCorner.y);
            }

            remoteRenderer.gBuffer.bind();
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.stencilState.stencilRef = 1;
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            remoteRenderer.pipeline.apply();

            glClearStencil(0);
            glClear(GL_STENCIL_BUFFER_BIT);
            quadMaskShader.bind();
            quadMaskShader.setVec2("uViewport", viewportSize);
            quadMaskShader.setBool("expand", false);

            // Pass 1: states regions that are reprojected back from current/future view as stencil 1
            quadMaskShader.setVec4("uDebugColor", glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
            quadMaskShader.setVec2("uCorners[0]", timewarpedPreviousCornersInWideFov[0]);
            quadMaskShader.setVec2("uCorners[1]", timewarpedPreviousCornersInWideFov[1]);
            quadMaskShader.setVec2("uCorners[2]", timewarpedPreviousCornersInWideFov[2]);
            quadMaskShader.setVec2("uCorners[3]", timewarpedPreviousCornersInWideFov[3]);
            quadMaskQuad.draw();

            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.stencilState.stencilRef = 0;
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            remoteRenderer.pipeline.apply();

            // Pass 2: mask out the current-frame coverage to show only regions that are outside of the normal view.
            quadMaskShader.setVec4("uDebugColor", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
            quadMaskShader.setVec2("uCorners[0]", normalViewCornersInWideFoVImage[0]);
            quadMaskShader.setVec2("uCorners[1]", normalViewCornersInWideFoVImage[1]);
            quadMaskShader.setVec2("uCorners[2]", normalViewCornersInWideFoVImage[2]);
            quadMaskShader.setVec2("uCorners[3]", normalViewCornersInWideFoVImage[3]);
            quadMaskQuad.draw();
            remoteRenderer.gBuffer.unbind();

            // Render only the uncovered difference region so the wide-FOV color footprint matches
            // the red empty-slot region from the synthetic warp reference.
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_EQUAL, 1);
            } else {
            
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                // remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                wideFovNodes[currMeshIndex].visible = true;
                wideFovNodes[prevMeshIndex].visible = false;
                renderStats += remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);
                // remoteRendereer.outputRT.writeColorAsPNG("quasar_wide_fov_no_tone.png");

                // Render remoteScene using stencil buffer as a mask
                // At values where stencil buffer is not 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            }
            
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();
            remoteRenderer.copyToFrameRT(renderTargetToUse);
            if (useExternalWideFovUsageMask) {
                hasExternalWideFovUsageMask = loadExternalWideFovUsageMaskForRemoteFrame(
                    static_cast<uint>(frameID),
                    wideFovImageDumpDir,
                    renderTargetToUse.width,
                    renderTargetToUse.height,
                    normalViewCornersInWideFoVImage,
                    externalWideFovUsageMask);
                if (hasExternalWideFovUsageMask) {
                    const size_t trimmedTexels = applyExternalWideFovUsageMask(renderTargetToUse, externalWideFovUsageMask);
                    spdlog::info(
                        "Applied external wide-FOV texel-usage mask to {} (kept {} / {}, trimmed {})",
                        externalWideFovUsageMask.sourcePath,
                        externalWideFovUsageMask.usefulTexels,
                        static_cast<size_t>(renderTargetToUse.width) * renderTargetToUse.height,
                        trimmedTexels);
                }
            }

            // log out the rendered wide fov image to the dump directory
            if (!wideFovImageDumpDir.empty()) {
                Path dumpDir(wideFovImageDumpDir);
                dumpDir.mkdirRecursive();
                Path pngPath = dumpDir / ("widefov_" + std::to_string(frameID) + ".png");
                renderTargetToUse.writeColorAsPNG(pngPath.str());
            }

        }
        stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        spdlog::info("Generating quads for layer {}, refreshWideFovLayerQuads: {}", layer, refreshWideFovLayerQuads);
        if (refreshWideFovLayerQuads) {
            auto oldParams = quadsGenerator->params;
            // Wide FOV has very loose parameters to reduce data size
            if (layer == maxLayers - 1) {
                // if (trimWideFov) {
                //     // Keep trimmed wide-FOV compensation patches geometrically faithful; the usual
                //     // aggressive wide-FOV simplification can erase or smear the small region that
                //     // is supposed to fill the black gap in the final reconstructed frame.
                //     quadsGenerator->params.expandEdges = false;
                // }
                // else {
                    quadsGenerator->params.planeSimilarityThreshold *= 4.0f;
                    quadsGenerator->params.expandEdges = true;
                // }
            }
            // Hidden layers have looser parameters to reduce data size
            else if (layer > 0) {
                quadsGenerator->params.planeSimilarityThreshold *= (layer * 2.0f);
                quadsGenerator->params.expandEdges = false;
            }
            ReferenceFrame dummyFrame;
            glFinish();
            frameGenerator.createReferenceFrame(
                (layer != 0 && layer != maxLayers - 1) ? renderTargetToUse_noTone : renderTargetToUse,
                remoteCameraToUse,
                meshToUse,
                (layer == 0 && createResidualFrame) ? dummyFrame : referenceFrames[layer] // Don't save output of this reference frame if we are making a residual frame
            );
            if (!showNormals) {
                if (layer == 0) {
                    remoteRenderer.copyToFrameRT(referenceFrameRT_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, referenceFrameRT);
                }
                else if (layer < maxLayers - 1) {
                    tonemapper.setUniforms(renderTargetToUse_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
                }
                else {
                    if (hasExternalWideFovUsageMask) {
                        applyExternalWideFovUsageMask(renderTargetToUse_noTone, externalWideFovUsageMask);
                    }
                    tonemapper.setUniforms(renderTargetToUse_noTone);
                    remoteRenderer.copyToFrameRT(renderTargetToUse_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse);
                    if (hasExternalWideFovUsageMask) {
                        applyExternalWideFovUsageMask(renderTargetToUse, externalWideFovUsageMask);
                    }
                }
            }
            else {
                showNormalsEffect.drawToRenderTarget(remoteRenderer, renderTargetToUse_noTone);
            }
            quadsGenerator->params = oldParams;

            stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
            stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
            stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

            stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
            stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
            stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

            stats.createProxiesTimeMsByLayer.push_back(frameGenerator.stats.createQuadsTimeMs);
            stats.compressTimeMsByLayer.push_back(frameGenerator.stats.compressTimeMs);
            stats.createMeshTimeMsByLayer.push_back(frameGenerator.stats.createMeshTimeMs);

            if (!createResidualFrame || layer != 0) {
                stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
            }
        }
        else {
            stats.createProxiesTimeMsByLayer.push_back(0.0);
            stats.compressTimeMsByLayer.push_back(0.0);
            stats.createMeshTimeMsByLayer.push_back(0.0);
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (layer == 0) {
            if (createResidualFrame) {
                /*
                ============================
                Generate masked Residual Frame textures
                ============================
                */
                frameGenerator.updateResidualRenderTargets(
                    residualFrameMaskRT, residualFrameRT,
                    remoteRenderer, remoteScene,
                    meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                    remoteCamera, remoteCameraPrev
                );

                /*
                ============================
                Generate Residual Frame
                ============================
                */
                quadsGenerator->params.expandEdges = true;
                frameGenerator.createResidualFrame(
                    residualFrameMaskRT, residualFrameRT,
                    remoteCamera, remoteCameraPrev,
                    referenceFrameMeshes[prevMeshIndex], residualFrameMesh,
                    residualFrame
                );
                if (!showNormals) {
                    residualFrameRT.blit(residualFrameRT_noTone);
                    tonemapper.setUniforms(residualFrameRT_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, residualFrameRT, false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, residualFrameRT_noTone);
                }

                stats.totalRenderTimeMs += frameGenerator.stats.updateRTsTimeMs;

                stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
                stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
                stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
                stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

                stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
                stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
                stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

                stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
            }
            else {
                // Only update the previous camera pose if we are not generating a Residual Frame
                remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
                lastMeshIndex = meshIndex;
                meshIndex++;
            }

            residualFrameNode.visible = createResidualFrame;
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth && refreshWideFovLayerQuads) {
            meshToUseDepth.update((layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV, renderTargetToUse);
            stats.totalGenDepthTimeMs += meshToUseDepth.stats.genDepthTime;
        }

        if (!(createResidualFrame && layer == 0)) {
            stats.proxySizes.numQuads += referenceFrames[layer].getTotalNumQuads();
            stats.proxySizes.numDepthOffsets += referenceFrames[layer].getTotalNumDepthOffsets();
            stats.proxySizes.quadsSize += referenceFrames[layer].getTotalQuadsSize();
            stats.proxySizes.depthOffsetsSize += referenceFrames[layer].getTotalDepthOffsetsSize();
            spdlog::debug("Reference frame generated with {} quads ({:.3f}MB), {} depth offsets ({:.3f}MB)",
                          referenceFrames[layer].getTotalNumQuads(), referenceFrames[layer].getTotalQuadsSize() / BYTES_PER_MEGABYTE,
                          referenceFrames[layer].getTotalNumDepthOffsets(), referenceFrames[layer].getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
        }
        else {
            stats.proxySizes.numQuads += residualFrame.getTotalNumQuads();
            stats.proxySizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
            stats.proxySizes.quadsSize += residualFrame.getTotalQuadsSize();
            stats.proxySizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
            spdlog::debug("Residual frame generated with {} updated quads ({:.3f}MB) and {} revealed quads ({:.3f}MB), {} updated depth offsets ({:.3f}MB) and {} revealed depth offsets ({:.3f}MB)",
                          residualFrame.getTotalNumQuadsUpdated(), residualFrame.getTotalQuadsUpdatedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumQuadsRevealed(), residualFrame.getTotalQuadsRevealedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumDepthOffsetsUpdated(), residualFrame.getTotalDepthOffsetsUpdatedSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumDepthOffsetsRevealed(), residualFrame.getTotalDepthOffsetsRevealedSize() / BYTES_PER_MEGABYTE);
        }
    }
     
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    for (int layer = 0; layer < maxLayers; layer++) {
        quasarStatsCSVFile << "," << stats.createProxiesTimeMsByLayer[layer];
        quasarStatsCSVFile << "," << stats.compressTimeMsByLayer[layer];
        quasarStatsCSVFile << "," << stats.createMeshTimeMsByLayer[layer];
    }
    quasarStatsCSVFile << "," << stats.totalCompressTimeMs << std::endl;
    quasarStatsCSVFile.close();

    // Update color and alpha atlases (tile frames side by side)
    uint row = 0, col = 0;
    uint dstWidth = referenceFrameRT.width, dstHeight = referenceFrameRT.height;
    for (int layer = 0; layer < maxLayers; layer++) {
        if (layer == 0) {
            referenceFrameRT.blit(videoAtlasStreamerRT,
                0, 0, referenceFrameRT.width, referenceFrameRT.height,
                col, row, dstWidth, dstHeight
            );
            referenceFrameRT.blit(alphaAtlasRT,
                0, 0, referenceFrameRT.width, referenceFrameRT.height,
                col, row, dstWidth, dstHeight
            );
        }
        else {
            int hiddenLayerIndex = layer - 1;
            frameRTsHidLayer[hiddenLayerIndex].blit(videoAtlasStreamerRT,
                0, 0, frameRTsHidLayer[hiddenLayerIndex].width, frameRTsHidLayer[hiddenLayerIndex].height,
                col, row, dstWidth, dstHeight
            );
            frameRTsHidLayer_noTone[hiddenLayerIndex].blit(alphaAtlasRT,
                0, 0, frameRTsHidLayer_noTone[hiddenLayerIndex].width, frameRTsHidLayer_noTone[hiddenLayerIndex].height,
                col, row, dstWidth, dstHeight
            );
        }
        col += referenceFrameRT.width;
        dstWidth += referenceFrameRT.width;
        if (col >= videoAtlasStreamerRT.width) {
            col = 0;
            dstWidth = referenceFrameRT.width;

            row += referenceFrameRT.height;
            dstHeight += referenceFrameRT.height;
            if (row >= videoAtlasStreamerRT.height) {
                row = 0;
                dstHeight = referenceFrameRT.height;
            }
        }
    }
    residualFrameRT.blit(videoAtlasStreamerRT,
        0, 0, residualFrameRT.width, residualFrameRT.height,
        col, row, dstWidth, dstHeight
    );
    residualFrameRT_noTone.blit(alphaAtlasRT,
        0, 0, residualFrameRT_noTone.width, residualFrameRT_noTone.height,
        col, row, dstWidth, dstHeight
    );

    // videoAtlasStreamerRT.writeColorAsPNG("debug_quasar_video_atlas.png");
    // alphaAtlasRT.writeAlphaAsPNG("debug_quasar_alpha_atlas.png");

    return renderStats;
}

void QUASARStreamer::sendFrame(PoseReceiver::PoseInfo poseInfo, bool createResidualFrame) {

    stats.frameSize = writeToMemory(poseInfo, createResidualFrame, compressedData);
    size_t compressedDataSize = compressedData.size();
    if (!videoURL.empty() && !proxiesURL.empty()) {
        // Send atlas frame
        videoAtlasStreamerRT.sendFrame(poseInfo.pose_id);
        // Send proxies
        send(compressedData);
    }

    if (prevSendTimeMs != 0.0) {
        
        double compressedDataSendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevSendTimeMs);
        double bitrateMbps = ((8.0 * compressedDataSize) / BYTES_PER_MEGABYTE) / timeutils::millisToSeconds(compressedDataSendTimeMs);

        bandwidthStatsCSVFile.open(bandwidthStatsCSVFileName, std::ios::app);
        bandwidthStatsCSVFile << frameID;
        bandwidthStatsCSVFile << "," << videoAtlasStreamerRT.stats.bitrateMbps;
        bandwidthStatsCSVFile << "," << bitrateMbps;
        for (int layer = 0; layer < maxLayers; layer++) {
            bandwidthStatsCSVFile << "," << bandwidthstats.proxy_size_by_layer[layer];
            bandwidthStatsCSVFile << "," << bandwidthstats.depth_offset_size_by_layer[layer];
        }
        bandwidthStatsCSVFile << std::endl;
        bandwidthStatsCSVFile.close();
    }
    prevSendTimeMs = timeutils::getTimeMicros();
}

void QUASARStreamer::writeTexturesToFiles(const Path& outputPath) {
    // Save color
    Path colorFileName = (outputPath / "color.jpg");
    videoAtlasStreamerRT.writeColorAsJPG(colorFileName);

    // Save alpha
    Path alphaFileName = (outputPath / "alpha.png");
    alphaAtlasRT.writeAlphaAsPNG(alphaFileName);
}

size_t QUASARStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = outputPath / "camera.bin";
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    Path cameraFileNamePrev = outputPath / "camera_prev.bin";
    cameraPose.setProjectionMatrix(remoteCameraPrev.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCameraPrev.getViewMatrix());
    cameraPose.writeToFile(cameraFileNamePrev);

    // Save metadata (viewSphereDiameter and wide FOV)
    QUASARReceiver::Params params = {
        .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
        .viewSphereDiameter = viewSphereDiameter,
        .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
    };
    FileIO::writeToBinaryFile(outputPath / "metadata.bin", &params, sizeof(params));

    writeTexturesToFiles(outputPath);

    // Save proxies
    size_t totalOutputSize = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        totalOutputSize += referenceFrames[layer].writeToFiles(outputPath, layer);
    }
    totalOutputSize += residualFrame.writeToFiles(outputPath);
    return totalOutputSize;
}

size_t QUASARStreamer::writeToMemory(PoseReceiver::PoseInfo poseInfo, bool writeResidualFrame, std::vector<char>& outputData) {
    // Save camera data
    Pose cameraPose;
    pose_id_t poseID = poseInfo.pose_id;
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save alpha data
    alphaAtlasRT.writeAlphaToMemory(alphaImageData);
    alphaCodec.compress(alphaImageData.data(), alphaData, alphaImageData.size());

    // Save geometry data
    // Save visible layer
    if (!writeResidualFrame) {
        referenceFrames[0].writeToMemory(geometryMetadatas[0]);
    }
    else {
        residualFrame.writeToMemory(geometryMetadatas[0]);
    }
    // Save hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; layer++) {
        referenceFrames[layer].writeToMemory(geometryMetadatas[layer]);
    }

    uint32_t geometrySize = 0;
    for (const auto& layerData : geometryMetadatas) {
        geometrySize += sizeof(uint32_t) + static_cast<uint32_t>(layerData.size());
    }

    double timestamp = double(timeutils::getTimeMicros());
    spdlog::info("Timestamp: {}", timestamp);

    QUASARReceiver::Header header{
        .poseID = poseID,
        .frameType = !writeResidualFrame ? QuadFrame::FrameType::REFERENCE : QuadFrame::FrameType::RESIDUAL,
        .params {
            .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
        },
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .alphaSize = static_cast<uint32_t>(alphaData.size()),
        .geometrySize = geometrySize,
        .pose_send_timestamp = poseInfo.send_timestamp,
        .pose_recv_timestamp = poseInfo.recv_timestamp,
        .frame_send_timestamp = timestamp,
    };

    spdlog::debug("Writing camera size: {:.3f}MB", static_cast<float>(header.cameraSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing alpha size: {:.3f}MB", static_cast<float>(header.alphaSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing geometry size: {:.3f}MB", static_cast<float>(header.geometrySize) / BYTES_PER_MEGABYTE);

    outputData.resize(header.getSize());
    char* ptr = outputData.data();

    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();

    // Write alpha data
    std::memcpy(ptr, alphaData.data(), alphaData.size());
    ptr += alphaData.size();

    // Write geometry data
    for (const auto& layerData : geometryMetadatas) {
        uint32_t layerSize = static_cast<uint32_t>(layerData.size());

        // Write size of layer
        std::memcpy(ptr, &layerSize, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Write layer data
        std::memcpy(ptr, layerData.data(), layerSize);
        ptr += layerSize;
    }

    spdlog::debug("Total data size: {:.3f}MB", static_cast<float>(outputData.size()) / BYTES_PER_MEGABYTE);

    for (int layer = 0; layer < maxLayers; layer++) {
        bandwidthstats.proxy_size_by_layer[layer] = referenceFrames[layer].getTotalQuadsSize();
        bandwidthstats.depth_offset_size_by_layer[layer] = referenceFrames[layer].getTotalDepthOffsetsSize();
    }
    bandwidthstats.alphaSize = alphaData.size();
    bandwidthstats.totalSize = outputData.size();
    return outputData.size();
}
