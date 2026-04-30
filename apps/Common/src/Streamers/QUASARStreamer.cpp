#include "RenderTargets/FrameRenderTarget.h"
#include <Streamers/QUASARStreamer.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

using quasar::FileIO;
using quasar::FrameRenderTarget;
using quasar::Path;
using quasar::Texture;
using quasar::WideFovMaskMethod;

namespace {

constexpr const char* kWideFovTexelUsageRoot =
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/test/robot_lab";
constexpr const char* kRemoteToRecordedFrameMapCsv =
    "/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/profileCode/imageQualityCorrelation/robot_lab/"
    "quasarTrimWithMaskGT_remote_to_recorded_frame_map.csv";
constexpr const char* kRemoteToRecordedFrameMapFilename =
    "quasarTrimWithMaskGT_remote_to_recorded_frame_map.csv";
constexpr const char* kRecvPoseToRenderFilename = "recv_pose_to_render.csv";
constexpr const char* kWideFovGroundTruthPoseRecordsFilename =
    "prev_curr_pose_records.json";
constexpr const char* kLegacyWideFovGroundTruthPoseRecordsFilename =
    "quasarTrimWithRealRender_prev_curr_pose_records.json";

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

struct CornerUncertaintyInfo {
    bool valid = false;
    glm::vec3 cameraPoint{0.0f};
    double sigmaU00 = 0.0;
    double sigmaU01 = 0.0;
    double sigmaU11 = 0.0;
    double lambdaMax = 0.0;
    double unclampedRadiusPx = 0.0;
    float radiusPx = 0.0f;
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
constexpr double kWideFovChi2Confidence68 = 2.30;
constexpr double kWideFovChi2Confidence95 = 5.99;
constexpr double kWideFovChi2Confidence99 = 9.21;
constexpr double kWideFovUncertaintyZEpsilon = 1e-5;

double chi2ForWideFovConfidence(double confidence) {
    if (confidence > 1.0 && confidence <= 100.0) {
        confidence /= 100.0;
    }

    if (std::abs(confidence - 0.68) < 0.02 || std::abs(confidence - 68.0) < 0.5) {
        return kWideFovChi2Confidence68;
    }
    if (std::abs(confidence - 0.95) < 0.02 || std::abs(confidence - 95.0) < 0.5) {
        return kWideFovChi2Confidence95;
    }
    if (std::abs(confidence - 0.99) < 0.01 || std::abs(confidence - 99.0) < 0.5) {
        return kWideFovChi2Confidence99;
    }

    if (std::isfinite(confidence) && confidence > 0.0 && confidence < 1.0) {
        return -2.0 * std::log(1.0 - confidence);
    }

    spdlog::warn(
        "Invalid wide-FOV reprojection confidence {}; defaulting to 68% chi2={}",
        confidence,
        kWideFovChi2Confidence68);
    return kWideFovChi2Confidence68;
}

std::array<double, 36> transformPoseCovarianceToCameraFrame(
    const std::array<double, 36>& poseCovariance,
    const glm::mat4& viewMatrix)
{
    double transform[6][6] = {};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double value = static_cast<double>(viewMatrix[col][row]);
            transform[row][col] = value;
            transform[row + 3][col + 3] = value;
        }
    }

    std::array<double, 36> transformed{};
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            double value = 0.0;
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    value += transform[row][k] * poseCovariance[k * 6 + l] * transform[col][l];
                }
            }
            transformed[row * 6 + col] = value;
        }
    }
    return transformed;
}

CornerUncertaintyInfo computeCornerReprojectionUncertainty(
    const glm::vec4& cornerInWorld,
    const glm::mat4& targetViewMatrix,
    const glm::mat4& projectionMatrix,
    const glm::uvec2& imageSize,
    const std::array<double, 36>& poseCovarianceCameraFrame,
    double chi2Threshold,
    float minRadiusPx,
    float maxRadiusPx)
{
    CornerUncertaintyInfo info;
    if (imageSize.x == 0u || imageSize.y == 0u || minRadiusPx > maxRadiusPx) {
        return info;
    }

    const glm::dvec4 cameraPointGL = glm::dmat4(targetViewMatrix) * glm::dvec4(cornerInWorld);
    const double x = cameraPointGL.x;
    const double y = cameraPointGL.y;
    const double z = -cameraPointGL.z;
    info.cameraPoint = glm::vec3(
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z));

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= kWideFovUncertaintyZEpsilon) {
        return info;
    }

    const double fx = std::abs(static_cast<double>(projectionMatrix[0][0]))
        * static_cast<double>(imageSize.x) * 0.5;
    const double fy = std::abs(static_cast<double>(projectionMatrix[1][1]))
        * static_cast<double>(imageSize.y) * 0.5;
    if (!std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.0 || fy <= 0.0) {
        return info;
    }

    const double z2 = z * z;
    const double jacobian[2][6] = {
        {
            fx / z,
            0.0,
            -fx * x / z2,
            -fx * x * y / z2,
            fx + fx * x * x / z2,
            -fx * y / z,
        },
        {
            0.0,
            fy / z,
            -fy * y / z2,
            -fy - fy * y * y / z2,
            fy * x * y / z2,
            fy * x / z,
        },
    };

    double sigmaU[2][2] = {};
    for (int outRow = 0; outRow < 2; ++outRow) {
        for (int outCol = 0; outCol < 2; ++outCol) {
            double value = 0.0;
            for (int poseRow = 0; poseRow < 6; ++poseRow) {
                for (int poseCol = 0; poseCol < 6; ++poseCol) {
                    value += jacobian[outRow][poseRow]
                        * poseCovarianceCameraFrame[poseRow * 6 + poseCol]
                        * jacobian[outCol][poseCol];
                }
            }
            sigmaU[outRow][outCol] = value;
        }
    }

    const double a = sigmaU[0][0];
    const double b = 0.5 * (sigmaU[0][1] + sigmaU[1][0]);
    const double c = sigmaU[1][1];
    const double trace = a + c;
    const double det = a * c - b * b;
    const double discriminant = std::max(0.0, trace * trace - 4.0 * det);
    const double lambdaMax = std::max(0.0, 0.5 * (trace + std::sqrt(discriminant)));
    const double unclampedRadius = std::sqrt(std::max(0.0, chi2Threshold * lambdaMax));
    if (!std::isfinite(unclampedRadius)) {
        return info;
    }

    info.valid = true;
    info.sigmaU00 = a;
    info.sigmaU01 = b;
    info.sigmaU11 = c;
    info.lambdaMax = lambdaMax;
    info.unclampedRadiusPx = unclampedRadius;
    info.radiusPx = std::clamp(static_cast<float>(unclampedRadius), minRadiusPx, maxRadiusPx);
    return info;
}

void expandCornerQuadByUncertainty(glm::vec2 corners[4], const CornerUncertaintyInfo uncertainty[4]) {
    const glm::vec2 signs[4] = {
        glm::vec2(-1.0f, -1.0f),
        glm::vec2(1.0f, -1.0f),
        glm::vec2(-1.0f, 1.0f),
        glm::vec2(1.0f, 1.0f),
    };

    for (int i = 0; i < 4; ++i) {
        if (!uncertainty[i].valid || uncertainty[i].radiusPx <= 0.0f) {
            continue;
        }
        corners[i] += signs[i] * uncertainty[i].radiusPx;
    }
}

float decodePackedProxyDepth(uint32_t normalAndDepthPacked) {
    return static_cast<float>(normalAndDepthPacked & 0xFFFFu) / 65535.0f;
}

bool decodePackedProxyFootprint(uint32_t packedMetadata, glm::uvec2& outOffset, uint32_t& outHalfSize) {
    outOffset.x = (packedMetadata >> 20) & 0xFFFu;
    outOffset.y = (packedMetadata >> 8) & 0xFFFu;
    const uint32_t sizeAlphaFlattened = packedMetadata & 0xFFu;
    const uint32_t size = (sizeAlphaFlattened >> 2) & 0x7Fu;
    if (size == 0u || size >= 31u) {
        return false;
    }

    outHalfSize = 1u << (size - 1u);
    return outHalfSize > 0u;
}

bool tryGetCurrentNormalViewCornerPlaneDepths(
    const quasar::QuadMesh& normalViewMesh,
    const glm::uvec2& viewportSize,
    float outDepths[4])
{
    if (viewportSize.x == 0u || viewportSize.y == 0u) {
        return false;
    }

    const glm::ivec2 cornerPixels[4] = {
        glm::ivec2(0, 0),
        glm::ivec2(static_cast<int>(viewportSize.x) - 1, 0),
        glm::ivec2(0, static_cast<int>(viewportSize.y) - 1),
        glm::ivec2(static_cast<int>(viewportSize.x) - 1, static_cast<int>(viewportSize.y) - 1),
    };

    const quasar::QuadBuffers& quadBuffers = normalViewMesh.getQuadBuffers();
    std::vector<uint32_t> packedNormalAndDepth;
    std::vector<uint32_t> packedMetadatas;
    packedNormalAndDepth.resize(quadBuffers.normalAndDepthBuffer.getSize());
    packedMetadatas.resize(quadBuffers.metadatasBuffer.getSize());
    quadBuffers.normalAndDepthBuffer.bind();
    quadBuffers.normalAndDepthBuffer.getData(packedNormalAndDepth.data());
    quadBuffers.normalAndDepthBuffer.unbind();
    quadBuffers.metadatasBuffer.bind();
    quadBuffers.metadatasBuffer.getData(packedMetadatas.data());
    quadBuffers.metadatasBuffer.unbind();

    bool foundAnyDepth = false;
    for (int i = 0; i < 4; ++i) {
        outDepths[i] = 1.0f;
        float bestDistanceSquared = std::numeric_limits<float>::max();
        int bestQuadIndex = -1;

        for (uint32_t quadIndex = 0; quadIndex < quadBuffers.numProxies; ++quadIndex) {
            if (quadIndex >= packedNormalAndDepth.size() || quadIndex >= packedMetadatas.size()) {
                break;
            }

            const float depth = decodePackedProxyDepth(packedNormalAndDepth[quadIndex]);
            if (!(depth > 0.0f && depth < 1.0f)) {
                continue;
            }

            glm::uvec2 offset{0u};
            uint32_t halfSize = 0u;
            if (!decodePackedProxyFootprint(packedMetadatas[quadIndex], offset, halfSize)) {
                continue;
            }

            const float minX = static_cast<float>(offset.x);
            const float minY = static_cast<float>(offset.y);
            const float maxX = minX + static_cast<float>(halfSize) - 1.0f;
            const float maxY = minY + static_cast<float>(halfSize) - 1.0f;
            const float px = static_cast<float>(cornerPixels[i].x);
            const float py = static_cast<float>(cornerPixels[i].y);

            const float dx = (px < minX) ? (minX - px) : ((px > maxX) ? (px - maxX) : 0.0f);
            const float dy = (py < minY) ? (minY - py) : ((py > maxY) ? (py - maxY) : 0.0f);
            const float distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < bestDistanceSquared) {
                bestDistanceSquared = distanceSquared;
                bestQuadIndex = static_cast<int>(quadIndex);
                outDepths[i] = depth;
            }
        }

        if (bestQuadIndex >= 0) {
            spdlog::info(
                "Trim-wideFov corner {} matched nearest proxy quad {} with plane depth {:.6f} (distance^2 {:.3f})",
                i,
                bestQuadIndex,
                outDepths[i],
                bestDistanceSquared);
            foundAnyDepth = true;
        }
        else {
            spdlog::warn(
                "Trim-wideFov corner {} failed to find any valid proxy-plane depth; falling back to far-plane depth",
                i);
        }
    }

    return foundAnyDepth;
}

bool readDepthPixelsFromDepthStencilTexture(
    const Texture& depthStencilTexture,
    uint width,
    uint height,
    std::vector<float>& outDepths)
{
    if (width == 0 || height == 0) {
        return false;
    }
    if (depthStencilTexture.array) {
        spdlog::warn("Depth-texture readback does not support array textures yet");
        return false;
    }

    outDepths.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 1.0f);

    GLint previousReadFramebuffer = 0;
    GLint previousDrawFramebuffer = 0;
    GLint previousReadBuffer = 0;
    GLint previousDrawBuffer = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
    glGetIntegerv(GL_READ_BUFFER, &previousReadBuffer);
    glGetIntegerv(GL_DRAW_BUFFER, &previousDrawBuffer);

    GLuint readFramebuffer = 0;
    glGenFramebuffers(1, &readFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, readFramebuffer);

    const GLenum textureTarget = depthStencilTexture.multiSampled ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, textureTarget, depthStencilTexture.ID, 0);
    glReadBuffer(GL_NONE);
    glDrawBuffer(GL_NONE);

    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        spdlog::warn(
            "Depth-texture readback framebuffer incomplete (status={} width={} height={})",
            static_cast<unsigned int>(status),
            width,
            height);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousReadFramebuffer));
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(previousDrawFramebuffer));
        glReadBuffer(static_cast<GLenum>(previousReadBuffer));
        glDrawBuffer(static_cast<GLenum>(previousDrawBuffer));
        glDeleteFramebuffers(1, &readFramebuffer);
        return false;
    }

    glReadPixels(
        0,
        0,
        static_cast<GLsizei>(width),
        static_cast<GLsizei>(height),
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        outDepths.data());

    glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousReadFramebuffer));
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(previousDrawFramebuffer));
    glReadBuffer(static_cast<GLenum>(previousReadBuffer));
    glDrawBuffer(static_cast<GLenum>(previousDrawBuffer));
    glDeleteFramebuffers(1, &readFramebuffer);
    return true;
}

bool getFrameRenderTargetCornerDepths(const FrameRenderTarget& frameRT, std::array<float, 4>& outDepths) {
    if (frameRT.width == 0 || frameRT.height == 0) {
        return false;
    }

    std::vector<float> depths;
    if (!readDepthPixelsFromDepthStencilTexture(frameRT.depthStencilTexture, frameRT.width, frameRT.height, depths)) {
        return false;
    }

    const GLint cornerPixels[4][2] = {
        {0, 0},
        {static_cast<GLint>(frameRT.width) - 1, 0},
        {0, static_cast<GLint>(frameRT.height) - 1},
        {static_cast<GLint>(frameRT.width) - 1, static_cast<GLint>(frameRT.height) - 1},
    };

    for (int i = 0; i < 4; ++i) {
        const size_t depthIndex =
            static_cast<size_t>(cornerPixels[i][1]) * static_cast<size_t>(frameRT.width) +
            static_cast<size_t>(cornerPixels[i][0]);
        outDepths[i] = depths[depthIndex];
    }

    return true;
}

void logFrameRenderTargetCornerDepths(const FrameRenderTarget& frameRT, const char* label) {
    if (frameRT.width == 0 || frameRT.height == 0) {
        spdlog::warn("{} corner depth read skipped because frame RT has invalid size {}x{}", label, frameRT.width, frameRT.height);
        return;
    }

    std::array<float, 4> cornerDepths{};
    if (!getFrameRenderTargetCornerDepths(frameRT, cornerDepths)) {
        spdlog::warn("{} corner depth read skipped because depthStencilTexture readback failed", label);
        return;
    }

    const GLint cornerPixels[4][2] = {
        {0, 0},
        {static_cast<GLint>(frameRT.width) - 1, 0},
        {0, static_cast<GLint>(frameRT.height) - 1},
        {static_cast<GLint>(frameRT.width) - 1, static_cast<GLint>(frameRT.height) - 1},
    };
    const char* cornerNames[4] = {
        "bottom-left",
        "bottom-right",
        "top-left",
        "top-right",
    };

    for (int i = 0; i < 4; ++i) {
        spdlog::info(
            "{} corner depth {}: pixel=({}, {}), depth={:.6f}",
            label,
            cornerNames[i],
            cornerPixels[i][0],
            cornerPixels[i][1],
            cornerDepths[i]);
    }
}

void logFrameRenderTargetDepthRange(const FrameRenderTarget& frameRT, const char* label) {
    if (frameRT.width == 0 || frameRT.height == 0) {
        spdlog::warn("{} depth-range read skipped because frame RT has invalid size {}x{}", label, frameRT.width, frameRT.height);
        return;
    }

    std::vector<float> depths;
    if (!readDepthPixelsFromDepthStencilTexture(frameRT.depthStencilTexture, frameRT.width, frameRT.height, depths)) {
        spdlog::warn("{} depth-range read skipped because depthStencilTexture readback failed", label);
        return;
    }

    float minDepth = std::numeric_limits<float>::max();
    float maxDepth = std::numeric_limits<float>::lowest();
    for (const float depth : depths) {
        minDepth = std::min(minDepth, depth);
        maxDepth = std::max(maxDepth, depth);
    }

    spdlog::info(
        "{} depth range: min={:.6f}, max={:.6f}",
        label,
        minDepth,
        maxDepth);
}

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

Path getLoggedInfoDirFromWideFovDumpDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "loggedInfo";
}

Path getReprojectionMaskCompareLoggedInfoDir(const std::string& wideFovImageDumpDir) {
    return getLoggedInfoDirFromWideFovDumpDir(wideFovImageDumpDir) / "reprojection_mask_compare";
}

Path getTrimWideFovTimewarpDebugDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "trim_widefov_timewarp_debug";
}

Path getTrimWideFovAtwDebugDir(const std::string& wideFovImageDumpDir) {
    return Path(getOutputDirFromWideFovDumpDir(wideFovImageDumpDir)) / "trim_widefov_atw_debug";
}

Path getTrimWideFovAtwDebugLoggedInfoDir(const std::string& wideFovImageDumpDir) {
    return getLoggedInfoDirFromWideFovDumpDir(wideFovImageDumpDir) / "trim_widefov_atw_debug";
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
    Path infoDir = getTrimWideFovAtwDebugLoggedInfoDir(wideFovImageDumpDir);
    infoDir.mkdirRecursive();
    const Path poseCsvPath = infoDir / "pose_pairs.csv";
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
    outputRT.writeColorAsPNG((dumpDir / ("warped_to_target_view_" + frameSuffix + ".png")).str());
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

    Path infoDir = getReprojectionMaskCompareLoggedInfoDir(wideFovImageDumpDir);
    infoDir.mkdirRecursive();
    std::ofstream posePairsCsv((infoDir / "pose_pairs.csv").str(), std::ios::app);
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

std::vector<std::string> resolveRemoteToRecordedFrameMapCandidates(
    const std::string& wideFovGroundTruthDir)
{
    if (!wideFovGroundTruthDir.empty()) {
        const Path groundTruthDir(wideFovGroundTruthDir);
        const Path loggedInfoDir = groundTruthDir / "loggedInfo";
        return {
            (groundTruthDir / kWideFovGroundTruthPoseRecordsFilename).str(),
            (loggedInfoDir / kWideFovGroundTruthPoseRecordsFilename).str(),
            (groundTruthDir / kLegacyWideFovGroundTruthPoseRecordsFilename).str(),
            (loggedInfoDir / kLegacyWideFovGroundTruthPoseRecordsFilename).str(),
            (groundTruthDir / kRemoteToRecordedFrameMapFilename).str(),
            (loggedInfoDir / kRemoteToRecordedFrameMapFilename).str(),
            (groundTruthDir / kRecvPoseToRenderFilename).str(),
            (loggedInfoDir / kRecvPoseToRenderFilename).str(),
        };
    }
    return { kRemoteToRecordedFrameMapCsv };
}

RemoteToRecordedFrameMap loadRemoteToRecordedFrameMap(const std::string& csvPath) {
    RemoteToRecordedFrameMap remoteToRecordedFrameMap;

    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        spdlog::warn("Failed to open remote-to-recorded frame map CSV: {}", csvPath);
        return remoteToRecordedFrameMap;
    }

    std::string headerLine;
    if (!std::getline(csvFile, headerLine)) {
        spdlog::warn("Remote-to-recorded frame map CSV is empty: {}", csvPath);
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

    int remoteFrameIDColumn = findColumnIndex("remote_frame_id");
    if (remoteFrameIDColumn < 0) {
        remoteFrameIDColumn = findColumnIndex("frame_id");
    }
    const int recordedFrameIDColumn = findColumnIndex("recorded_frame_id");
    const int recordedFrameIDPNGColumn = findColumnIndex("recorded_frame_id_png");

    if (remoteFrameIDColumn < 0 || recordedFrameIDColumn < 0) {
        spdlog::warn(
            "Remote-to-recorded frame map CSV is missing required columns: {}",
            csvPath);
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
        bool hasRecordedFrameIDPNG = false;
        if (recordedFrameIDPNGColumn >= 0) {
            hasRecordedFrameIDPNG = tryParseCSVInt(
                fields[recordedFrameIDPNGColumn],
                recordedMaskFrame.recordedFrameIDPNG);
        }
        else if (hasRecordedFrameID) {
            recordedMaskFrame.recordedFrameIDPNG = recordedMaskFrame.recordedFrameID;
            hasRecordedFrameIDPNG = true;
        }

        if ((!hasRecordedFrameID || recordedMaskFrame.recordedFrameID < 0)
            && (!hasRecordedFrameIDPNG || recordedMaskFrame.recordedFrameIDPNG < 0))
        {
            continue;
        }

        auto& recordedMaskFrames = remoteToRecordedFrameMap[static_cast<uint>(remoteFrameID)];
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

    spdlog::info(
        "Loaded {} remote-frame -> recorded-frame wide-FOV mask mappings from {}",
        remoteToRecordedFrameMap.size(),
        csvPath);
    return remoteToRecordedFrameMap;
}

RemoteToRecordedFrameMap loadRemoteToRecordedFrameMapFromJson(const std::string& jsonPath) {
    RemoteToRecordedFrameMap remoteToRecordedFrameMap;

    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        spdlog::warn("Failed to open remote-to-recorded frame map JSON: {}", jsonPath);
        return remoteToRecordedFrameMap;
    }

    nlohmann::json root;
    try {
        jsonFile >> root;
    }
    catch (const std::exception& e) {
        spdlog::warn("Failed to parse remote-to-recorded frame map JSON {}: {}", jsonPath, e.what());
        return remoteToRecordedFrameMap;
    }

    const auto remoteRenderFramesIt = root.find("remote_render_frames");
    if (remoteRenderFramesIt == root.end() || !remoteRenderFramesIt->is_object()) {
        spdlog::warn("Remote-to-recorded frame map JSON is missing remote_render_frames: {}", jsonPath);
        return remoteToRecordedFrameMap;
    }

    for (const auto& [remoteFrameIDText, remoteFrameEntry] : remoteRenderFramesIt->items()) {
        int remoteFrameID = -1;
        try {
            remoteFrameID = std::stoi(remoteFrameIDText);
        }
        catch (const std::exception&) {
            continue;
        }
        if (remoteFrameID < 0 || !remoteFrameEntry.is_object()) {
            continue;
        }

        auto& recordedMaskFrames = remoteToRecordedFrameMap[static_cast<uint>(remoteFrameID)];
        const auto recordedFramesIt = remoteFrameEntry.find("Recorded Frame");
        if (recordedFramesIt == remoteFrameEntry.end()
            || !recordedFramesIt->is_object()
            || recordedFramesIt->empty())
        {
            continue;
        }

        for (const auto& [recordedFrameIDPNGText, recordedFramePose] : recordedFramesIt->items()) {
            RecordedWideFovMaskFrame recordedMaskFrame;
            try {
                recordedMaskFrame.recordedFrameIDPNG = std::stoi(recordedFrameIDPNGText);
            }
            catch (const std::exception&) {
                continue;
            }

            recordedMaskFrame.recordedFrameID = recordedMaskFrame.recordedFrameIDPNG;
            if (recordedFramePose.is_object()) {
                const auto recordedFrameIDIt = recordedFramePose.find("recorded_frame_id");
                if (recordedFrameIDIt != recordedFramePose.end() && recordedFrameIDIt->is_number_integer()) {
                    recordedMaskFrame.recordedFrameID = recordedFrameIDIt->get<int>();
                }
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
    }

    spdlog::info(
        "Loaded {} remote-frame -> recorded-frame wide-FOV mask mappings from {}",
        remoteToRecordedFrameMap.size(),
        jsonPath);
    return remoteToRecordedFrameMap;
}

RemoteToRecordedFrameMap loadRemoteToRecordedFrameMapFromCandidates(
    const std::vector<std::string>& metadataPaths)
{
    for (const std::string& metadataPath : metadataPaths) {
        if (!Path(metadataPath).exists()) {
            continue;
        }

        RemoteToRecordedFrameMap remoteToRecordedFrameMap =
            Path(metadataPath).extension() == ".json"
                ? loadRemoteToRecordedFrameMapFromJson(metadataPath)
                : loadRemoteToRecordedFrameMap(metadataPath);
        if (!remoteToRecordedFrameMap.empty()) {
            return remoteToRecordedFrameMap;
        }
    }

    std::string candidateList;
    for (const std::string& metadataPath : metadataPaths) {
        if (!candidateList.empty()) {
            candidateList += ", ";
        }
        candidateList += metadataPath;
    }
    spdlog::warn("Failed to load any remote-to-recorded frame map metadata from: {}", candidateList);
    return {};
}

const RemoteToRecordedFrameMap& getRemoteToRecordedFrameMap(const std::string& wideFovGroundTruthDir) {
    static std::unordered_map<std::string, RemoteToRecordedFrameMap> cachedMapsByCsvPath;
    const std::vector<std::string> metadataPaths =
        resolveRemoteToRecordedFrameMapCandidates(wideFovGroundTruthDir);
    std::string cacheKey;
    for (const std::string& metadataPath : metadataPaths) {
        if (!cacheKey.empty()) {
            cacheKey += ";";
        }
        cacheKey += metadataPath;
    }

    auto mapIt = cachedMapsByCsvPath.find(cacheKey);
    if (mapIt == cachedMapsByCsvPath.end()) {
        mapIt = cachedMapsByCsvPath.emplace(
            cacheKey,
            loadRemoteToRecordedFrameMapFromCandidates(metadataPaths)).first;
    }
    return mapIt->second;
}

std::vector<Path> getWideFovTexelUsageMaskDirs(
    const std::string& wideFovGroundTruthDir,
    const std::string& wideFovTexelUsageMaskDir)
{
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

    if (!wideFovTexelUsageMaskDir.empty()) {
        appendMaskDirIfPresent(Path(wideFovTexelUsageMaskDir));
    }

    if (!wideFovGroundTruthDir.empty()) {
        const Path groundTruthDir(wideFovGroundTruthDir);
        appendMaskDirIfPresent(groundTruthDir / "widefov_texel_usage");
        appendMaskDirIfPresent(groundTruthDir.parent() / "widefov_texel_usage");
        appendMaskDirIfPresent(groundTruthDir / "quasarWideFoVUsage" / "widefov_texel_usage");
    }

    if (maskDirs.empty() && wideFovTexelUsageMaskDir.empty() && wideFovGroundTruthDir.empty()) {
        const Path maskRoot{std::string(kWideFovTexelUsageRoot)};
        appendMaskDirIfPresent(maskRoot / "quasarWideFoVUsage" / "widefov_texel_usage");
    }

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
    const std::string& wideFovGroundTruthDir,
    const std::string& wideFovTexelUsageMaskDir,
    uint expectedWidth,
    uint expectedHeight,
    const glm::vec3 normalViewCornersInWideFoVImage[4],
    ExternalWideFovUsageMask& outMask)
{
    const RemoteToRecordedFrameMap& remoteToRecordedFrameMap =
        getRemoteToRecordedFrameMap(wideFovGroundTruthDir);
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

    const std::vector<Path> maskDirs = getWideFovTexelUsageMaskDirs(
        wideFovGroundTruthDir,
        wideFovTexelUsageMaskDir);
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

struct PoseCsvRowInfo {
    glm::vec3 position{0.0f};
    glm::vec3 eulerRotationDegrees{0.0f};
};

PoseCsvRowInfo extractPoseCsvRowInfoFromViewMatrix(const glm::mat4& viewMatrix) {
    PoseCsvRowInfo info;
    glm::vec3 scale;
    glm::quat rotationQuat;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(viewMatrix), scale, rotationQuat, info.position, skew, perspective);
    info.eulerRotationDegrees = glm::degrees(glm::eulerAngles(glm::normalize(rotationQuat)));
    return info;
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
    , wideFovGroundTruthDir(params.wideFovGroundTruthDir)
    , wideFovTexelUsageMaskDir(params.wideFovTexelUsageMaskDir)
    , datasetOutputDir(params.datasetOutputDir)
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
    const std::string loggedOutputDir = !datasetOutputDir.empty() ? datasetOutputDir : outputDir;
    if (!loggedOutputDir.empty()) {
        Path(loggedOutputDir).mkdirRecursive();
    }

    quasarStatsCSVFileName = loggedOutputDir + "/quasar_stats.csv";
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

    bandwidthStatsCSVFileName = loggedOutputDir + "/quasar_streamer_bitrate.csv";
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

    if (!datasetOutputDir.empty()) {
        Path(datasetOutputDir).mkdirRecursive();
        cornerDepthDatasetCSVFileName = (Path(datasetOutputDir) / "render_corner_depths.csv").str();
        cornerDepthDatasetCSVFile.open(cornerDepthDatasetCSVFileName);
        cornerDepthDatasetCSVFile
            << "frame_id,pass_name,rendered_this_frame,pose_source,"
            << "pos_x,pos_y,pos_z,rot_x_deg,rot_y_deg,rot_z_deg,"
            << "depth_bottom_left,depth_bottom_right,depth_top_left,depth_top_right,"
            << "depth_read_success"
            << std::endl;
        cornerDepthDatasetCSVFile.close();

        wideFovCornerUncertaintyCSVFileName = (Path(datasetOutputDir) / "widefov_corner_uncertainty.csv").str();
        wideFovCornerUncertaintyCSVFile.open(wideFovCornerUncertaintyCSVFileName);
        wideFovCornerUncertaintyCSVFile
            << "frame_id,corner_id,confidence,chi2_threshold,depth,valid,"
            << "camera_x,camera_y,camera_z,"
            << "normal_px_x,normal_px_y,wide_px_x,wide_px_y,"
            << "expanded_wide_px_x,expanded_wide_px_y,"
            << "sigma_u_00,sigma_u_01,sigma_u_11,lambda_max,"
            << "unclamped_radius_px,radius_px,min_radius_px,max_radius_px"
            << std::endl;
        wideFovCornerUncertaintyCSVFile.close();
    }

    if (trimWideFov && !outputDir.empty()) {
        Path reprojectionMaskCompareInfoDir = getReprojectionMaskCompareLoggedInfoDir(wideFovImageDumpDir);
        reprojectionMaskCompareInfoDir.mkdirRecursive();
        std::ofstream posePairsCsv((reprojectionMaskCompareInfoDir / "pose_pairs.csv").str());
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
    const glm::mat4* wideFovGroundTruthView,
    const std::vector<glm::mat4>* debugWideFovMaskTargetViews,
    const WideFovReprojectionUncertainty* wideFovReprojectionUncertainty) {
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
    const auto appendCornerDepthRow = [&](const char* passName,
                                          bool renderedThisFrame,
                                          const char* poseSource,
                                          const glm::mat4& poseViewMatrix,
                                          const FrameRenderTarget& frameRT) {
        if (cornerDepthDatasetCSVFileName.empty()) {
            return;
        }

        std::array<float, 4> cornerDepths{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
        };
        const bool depthReadSuccess = getFrameRenderTargetCornerDepths(frameRT, cornerDepths);
        const PoseCsvRowInfo poseInfo = extractPoseCsvRowInfoFromViewMatrix(poseViewMatrix);

        cornerDepthDatasetCSVFile.open(cornerDepthDatasetCSVFileName, std::ios::app);
        cornerDepthDatasetCSVFile << frameID
            << "," << passName
            << "," << (renderedThisFrame ? 1 : 0)
            << "," << poseSource
            << "," << poseInfo.position.x
            << "," << poseInfo.position.y
            << "," << poseInfo.position.z
            << "," << poseInfo.eulerRotationDegrees.x
            << "," << poseInfo.eulerRotationDegrees.y
            << "," << poseInfo.eulerRotationDegrees.z
            << "," << cornerDepths[0]
            << "," << cornerDepths[1]
            << "," << cornerDepths[2]
            << "," << cornerDepths[3]
            << "," << (depthReadSuccess ? 1 : 0)
            << std::endl;
        cornerDepthDatasetCSVFile.close();
    };
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
            remoteRenderer.gBuffer.blitDepth(renderTargetToUse);
            logFrameRenderTargetDepthRange(renderTargetToUse, "Normal-view render");
            logFrameRenderTargetCornerDepths(renderTargetToUse, "Normal-view render");
            appendCornerDepthRow(
                createResidualFrame ? "normal_view_residual_pose" : "normal_view",
                true,
                createResidualFrame ? "remoteCameraPrev" : "remoteCamera",
                remoteCameraToUse.getViewMatrix(),
                renderTargetToUse);
            // if (trimWideFov && useWideFovGroundTruth && wideFovGroundTruthView != nullptr && !wideFovImageDumpDir.empty()) {
            //     dumpTrimWideFovAtwReprojectionDebug(
            //         static_cast<uint>(frameID),
            //         wideFovImageDumpDir,
            //         remoteRenderer,
            //         atwDebugShader,
            //         debugMaskRT,
            //         renderTargetToUse,
            //         remoteCameraToUse.getViewMatrix(),
            //         remoteCameraToUse.getProjectionMatrix(),
            //         *wideFovGroundTruthView,
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
            
            // Reproject from the actual remote-render source pose. Ground-truth/covariance
            // targets only change the destination pose used to compute the wide-FOV mask.
            glm::mat4 prevViewMatrix = remoteCamera.getViewMatrix();
            glm::mat4 prevViewMatrixInverse = glm::inverse(prevViewMatrix);
            glm::mat4 prevProjectionMatrix = remoteCamera.getProjectionMatrix();

            glm::mat4 currentViewMatrix = remoteCamera.getViewMatrix();
            const char* wideFovTargetPoseSource = "remoteCamera";
            std::vector<glm::mat4> activeWideFovMaskTargetViews;
            const bool useAnalyticReprojectionUncertainty =
                wideFovReprojectionUncertainty != nullptr
                && wideFovReprojectionUncertainty->enabled
                && wideFovReprojectionUncertainty->valid;
            if (useAnalyticReprojectionUncertainty) {
                currentViewMatrix = wideFovReprojectionUncertainty->meanViewMatrix;
                wideFovTargetPoseSource = "wideFovReprojectionUncertainty";
                activeWideFovMaskTargetViews.push_back(currentViewMatrix);
                if (debugWideFovMaskTargetViews != nullptr && debugWideFovMaskTargetViews->size() > 1u) {
                    activeWideFovMaskTargetViews.insert(
                        activeWideFovMaskTargetViews.end(),
                        debugWideFovMaskTargetViews->begin() + 1,
                        debugWideFovMaskTargetViews->end());
                    spdlog::info(
                        "Using analytic wide-FOV reprojection uncertainty plus {} debug covariance-sampled poses on frame {}",
                        debugWideFovMaskTargetViews->size() - 1u,
                        frameID);
                }
            }
            else if (debugWideFovMaskTargetViews != nullptr && !debugWideFovMaskTargetViews->empty()) {
                activeWideFovMaskTargetViews = *debugWideFovMaskTargetViews;
                currentViewMatrix = activeWideFovMaskTargetViews.front();
                wideFovTargetPoseSource = "debugWideFovMaskTargetViews";
                spdlog::info(
                    "Using {} debug covariance-sampled target poses for wide-FOV stencil union on frame {}",
                    activeWideFovMaskTargetViews.size(),
                    frameID);
            }
            else if (useWideFovGroundTruth && wideFovGroundTruthView != nullptr) {
                currentViewMatrix = *wideFovGroundTruthView;
                wideFovTargetPoseSource = "wideFovGroundTruthView";
                activeWideFovMaskTargetViews.push_back(currentViewMatrix);
            }
            else {
                activeWideFovMaskTargetViews.push_back(currentViewMatrix);
            }
            glm::mat4 currentProjectionMatrix = remoteCamera.getProjectionMatrix();

            logPoseForDebug("Previous", prevViewMatrix);
            logPoseForDebug("Current", currentViewMatrix);
            spdlog::info("Current viewport size: ({}, {})", quadSet.getSize().x, quadSet.getSize().y);

            glm::mat4 prevProjectionMatrixInverse = glm::inverse(prevProjectionMatrix);
            
            // Keep the wide-FOV proxy generation camera aligned with the pose that produced the
            // wide-FOV render target; otherwise the dumped wide-FOV image can be correct while
            // the reconstructed wide-FOV quads still project into the wrong part of the final frame.
            remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

            glm::mat4 remoteCameraWideFoVProjectionMatrix = remoteCameraWideFOV.getProjectionMatrix();
        
            // if (totalBlackArea > 10000.0f) {
            if (trimWideFov) {
            const glm::vec2 viewportSize(remoteRenderer.width, remoteRenderer.height);
            float cornerPlaneDepths[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            const bool cornerDepthsReliable =
                tryGetCurrentNormalViewCornerPlaneDepths(referenceFrameMeshes[currMeshIndex], quadSet.getSize(), cornerPlaneDepths);
            const glm::vec3 normalViewportCorners[4] = {
                glm::vec3(-1.0f, -1.0f, 1.0f),
                glm::vec3(1.0f, -1.0f, 1.0f),
                glm::vec3(-1.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 1.0f, 1.0f),
            };

            glm::vec4 cornerInWorldPoints[4];
            glm::vec2 timewarpedPreviousCornersInNormalView[4];
            glm::vec2 timewarpedPreviousCornersInWideFov[4];
            for (int i = 0; i < 4; i++) {
                glm::vec3 corner = normalViewportCorners[i];
                const float cornerPlaneDepthNdc = cornerPlaneDepths[i] * 2.0f - 1.0f;
                corner.z = cornerPlaneDepthNdc;
                spdlog::info(
                    "Trim-wideFov corner {} using proxy-plane depth {:.6f} (ndc z {:.6f})",
                    i,
                    cornerPlaneDepths[i],
                    cornerPlaneDepthNdc);

                glm::vec4 cornerInWorld = prevViewMatrixInverse * prevProjectionMatrixInverse * glm::vec4(corner, 1.0f);
                cornerInWorldPoints[i] = cornerInWorld;
                glm::vec4 reprojectedCorner = currentProjectionMatrix * currentViewMatrix * cornerInWorld;

                spdlog::info(
                    "Corner {}: world=({:.3f}, {:.3f}, {:.3f}), reprojected=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
                    i,
                    cornerInWorld.x, cornerInWorld.y, cornerInWorld.z,
                    reprojectedCorner.x, reprojectedCorner.y, reprojectedCorner.z, reprojectedCorner.w);
                glm::vec4 reprojectedCornerNDC = reprojectedCorner/reprojectedCorner.w;

                reprojectedCornerNDC.x = (reprojectedCornerNDC.x + 1.0f) * 0.5f * quadSet.getSize().x;
                reprojectedCornerNDC.y = (reprojectedCornerNDC.y + 1.0f) * 0.5f * quadSet.getSize().y;
                timewarpedPreviousCornersInNormalView[i] = glm::vec2{reprojectedCornerNDC.x, reprojectedCornerNDC.y};

                glm::vec4 reprojectedCornerInWideFov = remoteCameraWideFoVProjectionMatrix * currentViewMatrix * cornerInWorld;

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
                spdlog::info("  Corner {}: raw=({:.3f}, {:.3f})", i, normalCorner.x, normalCorner.y);
            }

            spdlog::info("Timewarped previous corners projected into wide FOV:");
            for (int i = 0; i < 4; ++i) {
                const glm::vec2& wideFovCorner = timewarpedPreviousCornersInWideFov[i];
                spdlog::info("  Corner {}: raw=({:.3f}, {:.3f})", i, wideFovCorner.x, wideFovCorner.y);
            }

            glm::vec2 normalViewRectMin = glm::vec2(normalViewCornersInWideFoVImage[0]);
            glm::vec2 normalViewRectMax = glm::vec2(normalViewCornersInWideFoVImage[0]);
            for (int i = 1; i < 4; ++i) {
                const glm::vec2 normalViewRectCorner = glm::vec2(normalViewCornersInWideFoVImage[i]);
                normalViewRectMin = glm::min(normalViewRectMin, normalViewRectCorner);
                normalViewRectMax = glm::max(normalViewRectMax, normalViewRectCorner);
            }
            spdlog::info(
                "Normal view rect in wide FOV image: min=({:.3f}, {:.3f}), max=({:.3f}, {:.3f})",
                normalViewRectMin.x, normalViewRectMin.y,
                normalViewRectMax.x, normalViewRectMax.y);

            glm::vec2 quadWindowCorners[4] = {
                glm::vec2(0.0f, 0.0f),
                glm::vec2(quadSet.getSize().x, 0.0f),
                glm::vec2(0.0f, quadSet.getSize().y),
                glm::vec2(quadSet.getSize().x, quadSet.getSize().y),
            };

            for (int i = 0; i < 4; ++i) {
                const glm::vec2 originalWideFovCorner = timewarpedPreviousCornersInWideFov[i];

                bool xInRange = originalWideFovCorner.x >= normalViewRectMin.x && originalWideFovCorner.x <= normalViewRectMax.x;
                bool yInRange = originalWideFovCorner.y >= normalViewRectMin.y && originalWideFovCorner.y <= normalViewRectMax.y;

                if (!xInRange && !yInRange) {
                    continue;
                } else {
                    const glm::vec2 originalNormalViewCorner = glm::vec2(timewarpedPreviousCornersInNormalView[i]);
                    const glm::vec2 candidateCorner = glm::vec2(quadWindowCorners[i]);
                    const glm::vec2 deltaInNormalView = candidateCorner - originalNormalViewCorner;

                    if (xInRange) {
                        timewarpedPreviousCornersInWideFov[i].x = normalViewCornersInWideFoVImage[i].x + deltaInNormalView.x;
                    }
                    if (yInRange) {
                        timewarpedPreviousCornersInWideFov[i].y = normalViewCornersInWideFoVImage[i].y + deltaInNormalView.y;
                    }
                }

                spdlog::info(
                    "Mirrored timewarped corner {}  because wideFov=({:.3f}, {:.3f}) was inside the normal-view rect; "
                    "mirrored=({:.3f}, {:.3f})",
                    i,
                    originalWideFovCorner.x,
                    originalWideFovCorner.y,
                    timewarpedPreviousCornersInWideFov[i].x,
                    timewarpedPreviousCornersInWideFov[i].y);
            }

            CornerUncertaintyInfo cornerUncertainties[4];
            glm::vec2 uncertaintyBaseWideCorners[4] = {
                timewarpedPreviousCornersInWideFov[0],
                timewarpedPreviousCornersInWideFov[1],
                timewarpedPreviousCornersInWideFov[2],
                timewarpedPreviousCornersInWideFov[3],
            };
            double chi2Threshold = 0.0;
            float uncertaintyMinRadiusPx = 0.0f;
            float uncertaintyMaxRadiusPx = 0.0f;
            if (useAnalyticReprojectionUncertainty) {
                chi2Threshold = chi2ForWideFovConfidence(wideFovReprojectionUncertainty->confidence);
                uncertaintyMinRadiusPx = std::max(0.0f, wideFovReprojectionUncertainty->minRadiusPx);
                uncertaintyMaxRadiusPx = std::max(
                    uncertaintyMinRadiusPx,
                    wideFovReprojectionUncertainty->maxRadiusPx);
                const std::array<double, 36> poseCovarianceCameraFrame =
                    transformPoseCovarianceToCameraFrame(
                        wideFovReprojectionUncertainty->poseCovariance,
                        currentViewMatrix);

                for (int i = 0; i < 4; ++i) {
                    const bool depthValid =
                        cornerDepthsReliable
                        && std::isfinite(cornerPlaneDepths[i])
                        && cornerPlaneDepths[i] > 0.0f
                        && cornerPlaneDepths[i] < 1.0f;
                    if (!depthValid) {
                        continue;
                    }

                    cornerUncertainties[i] = computeCornerReprojectionUncertainty(
                        cornerInWorldPoints[i],
                        currentViewMatrix,
                        remoteCameraWideFoVProjectionMatrix,
                        quadSet.getSize(),
                        poseCovarianceCameraFrame,
                        chi2Threshold,
                        uncertaintyMinRadiusPx,
                        uncertaintyMaxRadiusPx);
                }

                expandCornerQuadByUncertainty(timewarpedPreviousCornersInWideFov, cornerUncertainties);
                spdlog::info(
                    "Wide-FOV analytic corner uncertainty frame {}: radii_px=({:.3f}, {:.3f}, {:.3f}, {:.3f}), "
                    "confidence={:.2f}, chi2={:.2f}, clamp=[{:.3f}, {:.3f}]",
                    frameID,
                    cornerUncertainties[0].radiusPx,
                    cornerUncertainties[1].radiusPx,
                    cornerUncertainties[2].radiusPx,
                    cornerUncertainties[3].radiusPx,
                    wideFovReprojectionUncertainty->confidence,
                    chi2Threshold,
                    uncertaintyMinRadiusPx,
                    uncertaintyMaxRadiusPx);

                if (!wideFovCornerUncertaintyCSVFileName.empty()) {
                    wideFovCornerUncertaintyCSVFile.open(wideFovCornerUncertaintyCSVFileName, std::ios::app);
                    for (int i = 0; i < 4; ++i) {
                        const CornerUncertaintyInfo& info = cornerUncertainties[i];
                        wideFovCornerUncertaintyCSVFile
                            << frameID << ","
                            << i << ","
                            << wideFovReprojectionUncertainty->confidence << ","
                            << chi2Threshold << ","
                            << cornerPlaneDepths[i] << ","
                            << (info.valid ? 1 : 0) << ","
                            << info.cameraPoint.x << ","
                            << info.cameraPoint.y << ","
                            << info.cameraPoint.z << ","
                            << timewarpedPreviousCornersInNormalView[i].x << ","
                            << timewarpedPreviousCornersInNormalView[i].y << ","
                            << uncertaintyBaseWideCorners[i].x << ","
                            << uncertaintyBaseWideCorners[i].y << ","
                            << timewarpedPreviousCornersInWideFov[i].x << ","
                            << timewarpedPreviousCornersInWideFov[i].y << ","
                            << info.sigmaU00 << ","
                            << info.sigmaU01 << ","
                            << info.sigmaU11 << ","
                            << info.lambdaMax << ","
                            << info.unclampedRadiusPx << ","
                            << info.radiusPx << ","
                            << uncertaintyMinRadiusPx << ","
                            << uncertaintyMaxRadiusPx
                            << std::endl;
                    }
                    wideFovCornerUncertaintyCSVFile.close();
                }
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

            for (size_t targetViewIndex = 1; targetViewIndex < activeWideFovMaskTargetViews.size(); ++targetViewIndex) {
                glm::vec2 sampledCornersInNormalView[4];
                glm::vec2 sampledCornersInWideFov[4];
                const glm::mat4& sampledTargetViewMatrix = activeWideFovMaskTargetViews[targetViewIndex];

                for (int i = 0; i < 4; ++i) {
                    glm::vec3 corner = normalViewportCorners[i];
                    corner.z = cornerPlaneDepths[i] * 2.0f - 1.0f;

                    const glm::vec4 cornerInWorld =
                        prevViewMatrixInverse * prevProjectionMatrixInverse * glm::vec4(corner, 1.0f);

                    glm::vec4 sampledCornerInNormal =
                        currentProjectionMatrix * sampledTargetViewMatrix * cornerInWorld;
                    sampledCornerInNormal /= sampledCornerInNormal.w;
                    sampledCornerInNormal.x =
                        (sampledCornerInNormal.x + 1.0f) * 0.5f * quadSet.getSize().x;
                    sampledCornerInNormal.y =
                        (sampledCornerInNormal.y + 1.0f) * 0.5f * quadSet.getSize().y;
                    sampledCornersInNormalView[i] = glm::vec2(sampledCornerInNormal);

                    glm::vec4 sampledCornerInWideFov =
                        remoteCameraWideFoVProjectionMatrix * sampledTargetViewMatrix * cornerInWorld;
                    sampledCornerInWideFov /= sampledCornerInWideFov.w;
                    sampledCornerInWideFov.x =
                        (sampledCornerInWideFov.x + 1.0f) * 0.5f * quadSet.getSize().x;
                    sampledCornerInWideFov.y =
                        (sampledCornerInWideFov.y + 1.0f) * 0.5f * quadSet.getSize().y;
                    sampledCornersInWideFov[i] = glm::vec2(sampledCornerInWideFov);
                }

                for (int i = 0; i < 4; ++i) {
                    const glm::vec2 originalWideFovCorner = sampledCornersInWideFov[i];
                    const bool xInRange =
                        originalWideFovCorner.x >= normalViewRectMin.x
                        && originalWideFovCorner.x <= normalViewRectMax.x;
                    const bool yInRange =
                        originalWideFovCorner.y >= normalViewRectMin.y
                        && originalWideFovCorner.y <= normalViewRectMax.y;

                    if (!xInRange && !yInRange) {
                        continue;
                    }

                    const glm::vec2 deltaInNormalView =
                        quadWindowCorners[i] - sampledCornersInNormalView[i];
                    if (xInRange) {
                        sampledCornersInWideFov[i].x =
                            normalViewCornersInWideFoVImage[i].x + deltaInNormalView.x;
                    }
                    if (yInRange) {
                        sampledCornersInWideFov[i].y =
                            normalViewCornersInWideFoVImage[i].y + deltaInNormalView.y;
                    }
                }

                quadMaskShader.setVec2("uCorners[0]", sampledCornersInWideFov[0]);
                quadMaskShader.setVec2("uCorners[1]", sampledCornersInWideFov[1]);
                quadMaskShader.setVec2("uCorners[2]", sampledCornersInWideFov[2]);
                quadMaskShader.setVec2("uCorners[3]", sampledCornersInWideFov[3]);
                quadMaskQuad.draw();
            }

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
                    wideFovGroundTruthDir,
                    wideFovTexelUsageMaskDir,
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

            appendCornerDepthRow(
                "wide_fov",
                true,
                wideFovTargetPoseSource,
                remoteCameraToUse.getViewMatrix(),
                renderTargetToUse);

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
