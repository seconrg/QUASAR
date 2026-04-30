#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <PoseSendRecvSimulator.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace quasar;

namespace {

constexpr float kPredictionCovarianceFloor = 1e-9f;

glm::vec3 rotationDeltaVector(const glm::quat& to, const glm::quat& from) {
    glm::quat delta = glm::normalize(to * glm::inverse(from));
    if (delta.w < 0.0f) {
        delta = -delta;
    }

    const glm::vec3 imaginary(delta.x, delta.y, delta.z);
    const float imaginaryLength = glm::length(imaginary);
    if (imaginaryLength < 1e-6f) {
        return 2.0f * imaginary;
    }

    const float angle = 2.0f * std::atan2(
        imaginaryLength,
        glm::clamp(delta.w, -1.0f, 1.0f));
    return imaginary * (angle / imaginaryLength);
}

void addOuterProduct(glm::mat3& covariance, const glm::vec3& value, double scale = 1.0) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            covariance[col][row] += static_cast<float>(
                static_cast<double>(value[row]) * static_cast<double>(value[col]) * scale);
        }
    }
}

void addDiagonalFloor(glm::mat3& covariance) {
    covariance[0][0] += kPredictionCovarianceFloor;
    covariance[1][1] += kPredictionCovarianceFloor;
    covariance[2][2] += kPredictionCovarianceFloor;
}

void fillPoseCovariance6x6(PoseSendRecvSimulator::PredictionCovariance& covariance) {
    covariance.pose6x6.fill(0.0);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            covariance.pose6x6[row * 6 + col] = covariance.positionCovariance[col][row];
            covariance.pose6x6[(row + 3) * 6 + (col + 3)] = covariance.rotationCovariance[col][row];
        }
    }
}

double confidenceRadius6D(double confidence) {
    const double clampedConfidence = std::clamp(confidence, 1e-6, 1.0 - 1e-6);
    if (std::abs(clampedConfidence - confidence) > 1e-9) {
        spdlog::warn(
            "Clamped requested covariance confidence {} to {}",
            confidence,
            clampedConfidence);
    }

    auto chiSquare6Cdf = [](double value) {
        const double halfValue = value * 0.5;
        return 1.0 - std::exp(-halfValue) * (1.0 + halfValue + 0.5 * halfValue * halfValue);
    };

    double low = 0.0;
    double high = 1.0;
    while (chiSquare6Cdf(high) < clampedConfidence) {
        high *= 2.0;
    }

    for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = 0.5 * (low + high);
        if (chiSquare6Cdf(mid) < clampedConfidence) {
            low = mid;
        }
        else {
            high = mid;
        }
    }

    return std::sqrt(0.5 * (low + high));
}

std::array<double, 36> choleskyLower6x6(std::array<double, 36> covariance) {
    std::array<double, 36> lower{};
    for (int i = 0; i < 6; ++i) {
        covariance[i * 6 + i] = std::max(covariance[i * 6 + i], 1e-12);
    }

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col <= row; ++col) {
            double sum = covariance[row * 6 + col];
            for (int k = 0; k < col; ++k) {
                sum -= lower[row * 6 + k] * lower[col * 6 + k];
            }

            if (row == col) {
                lower[row * 6 + col] = std::sqrt(std::max(sum, 1e-12));
            }
            else {
                const double denominator = lower[col * 6 + col];
                lower[row * 6 + col] = denominator > 1e-12 ? sum / denominator : 0.0;
            }
        }
    }

    return lower;
}

std::array<double, 6> transformUnitDirection6D(
    const std::array<double, 36>& lower,
    const std::array<double, 6>& direction,
    double radius)
{
    std::array<double, 6> offset{};
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col <= row; ++col) {
            offset[row] += lower[row * 6 + col] * direction[col];
        }
        offset[row] *= radius;
    }
    return offset;
}

std::vector<std::array<double, 6>> buildSeparatedUnitDirections6D() {
    std::vector<std::array<double, 6>> directions;
    auto addDirection = [&](std::array<double, 6> direction) {
        double normSquared = 0.0;
        for (double value : direction) {
            normSquared += value * value;
        }
        if (normSquared <= 0.0) {
            return;
        }
        const double invNorm = 1.0 / std::sqrt(normSquared);
        for (double& value : direction) {
            value *= invNorm;
        }
        directions.push_back(direction);
    };

    for (int axis = 0; axis < 6; ++axis) {
        std::array<double, 6> positive{};
        positive[axis] = 1.0;
        addDirection(positive);
        positive[axis] = -1.0;
        addDirection(positive);
    }

    for (int firstAxis = 0; firstAxis < 6; ++firstAxis) {
        for (int secondAxis = firstAxis + 1; secondAxis < 6; ++secondAxis) {
            for (double firstSign : {-1.0, 1.0}) {
                for (double secondSign : {-1.0, 1.0}) {
                    std::array<double, 6> direction{};
                    direction[firstAxis] = firstSign;
                    direction[secondAxis] = secondSign;
                    addDirection(direction);
                }
            }
        }
    }

    return directions;
}

double covarianceOffsetDistance(
    const std::array<double, 6>& lhs,
    const std::array<double, 6>& rhs,
    double minPositionSeparationM,
    double minRotationSeparationRad)
{
    const double positionScale = std::max(minPositionSeparationM, 1e-6);
    const double rotationScale = std::max(minRotationSeparationRad, 1e-6);

    double positionDistanceSquared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double diff = (lhs[axis] - rhs[axis]) / positionScale;
        positionDistanceSquared += diff * diff;
    }

    double rotationDistanceSquared = 0.0;
    for (int axis = 3; axis < 6; ++axis) {
        const double diff = (lhs[axis] - rhs[axis]) / rotationScale;
        rotationDistanceSquared += diff * diff;
    }

    return std::sqrt(positionDistanceSquared + rotationDistanceSquared);
}

Pose poseFromCovarianceOffset(const Pose& meanPose, const std::array<double, 6>& offset) {
    glm::vec3 scale;
    glm::quat meanRotation;
    glm::vec3 meanPosition;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(glm::inverse(meanPose.mono.view), scale, meanRotation, meanPosition, skew, perspective);
    meanRotation = glm::normalize(meanRotation);

    const glm::vec3 positionOffset(
        static_cast<float>(offset[0]),
        static_cast<float>(offset[1]),
        static_cast<float>(offset[2]));
    const glm::vec3 rotationOffsetVector(
        static_cast<float>(offset[3]),
        static_cast<float>(offset[4]),
        static_cast<float>(offset[5]));
    const float rotationOffsetAngle = glm::length(rotationOffsetVector);
    const glm::quat rotationOffset = rotationOffsetAngle > 1e-7f
        ? glm::angleAxis(rotationOffsetAngle, rotationOffsetVector / rotationOffsetAngle)
        : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    const glm::mat4 transform =
        glm::translate(glm::mat4(1.0f), meanPosition + positionOffset)
        * glm::mat4_cast(glm::normalize(rotationOffset * meanRotation));

    Pose sampledPose;
    sampledPose.setViewMatrix(glm::inverse(transform));
    sampledPose.setProjectionMatrix(meanPose.mono.proj);
    sampledPose.send_timestamp = meanPose.send_timestamp;
    return sampledPose;
}

} // namespace

PoseSendRecvSimulator::PoseSendRecvSimulator(PoseSendRecvSimulatorCreateParams params)
    : networkLatencyS(timeutils::millisToSeconds(params.networkLatencyMs))
    , networkJitterS(timeutils::millisToSeconds(params.networkJitterMs))
    , renderTimeS(timeutils::millisToSeconds(params.renderTimeMs))
    , posePrediction(params.posePrediction)
    , poseSmoothing(params.poseSmoothing)
    , generator(params.seed)
    , distribution(-networkJitterS, networkJitterS)
    , actualInJitter(randomJitter())
    , actualOutJitter(randomJitter())
{}

void PoseSendRecvSimulator::setNetworkLatency(double networkLatencyMs) {
    networkLatencyS = timeutils::millisToSeconds(networkLatencyMs);
    clear();
}

void PoseSendRecvSimulator::setNetworkJitter(double networkJitterMs) {
    networkJitterS = timeutils::millisToSeconds(networkJitterMs);
    distribution = std::uniform_real_distribution<double>(-networkJitterS, networkJitterS);
    clear();
}

void PoseSendRecvSimulator::setRenderTime(double renderTimeMs) {
    renderTimeS = timeutils::millisToSeconds(renderTimeMs);
    clear();
}

void PoseSendRecvSimulator::clear() {
    incomingPoses.clear();
    outPoses.clear();
    outOrigTimestamps.clear();
    positionErrors.clear();
    rotationErrors.clear();
    rtts.clear();
    positionHistory.clear();
    rotationHistory.clear();
    lastPredictionDebugInfo = {};
    lastPredictionCovariance = {};
}

void PoseSendRecvSimulator::sendPose(const PerspectiveCamera& camera, double now) {
    glm::mat4 viewMatrix = glm::inverse(camera.getViewMatrix());
    // spdlog::info("Send pose to server: {}, {}, {}", camera.getViewMatrix()[3][0], camera.getViewMatrix()[3][1], camera.getViewMatrix()[3][2]);
    incomingPoses.push_back({
        camera.getViewMatrix(),
        camera.getProjectionMatrix(),
        static_cast<double>(timeutils::secondsToMicros(now))
    });
    update(now);
}

void PoseSendRecvSimulator::update(float now) {
    if (now <= lastUpdateTimeS) return;
    lastUpdateTimeS = now;

    if (!incomingPoses.empty()) {
        double dtFuture = networkLatencyS;
        Pose poseToRecv = incomingPoses.front();
        double timestampS = timeutils::microsToSeconds(poseToRecv.send_timestamp);
        if (networkLatencyS > 0 && now - timestampS < dtFuture + actualInJitter) return;

        poseToRecv.send_timestamp = static_cast<double>(timeutils::secondsToMicros(now));
        actualInJitter = randomJitter();

        outPoses.push_back(poseToRecv);
        outOrigTimestamps.push_back(timestampS);
        incomingPoses.pop_front();
    }
}

bool PoseSendRecvSimulator::recvPoseToRender(Pose& pose, double now) {
    if (outPoses.empty() && outOrigTimestamps.empty()) return false;

    double dtFuture = networkLatencyS + renderTimeS;
    double jitterPredicted = randomJitter();

    Pose poseToSend = (networkLatencyS != 0) ? outPoses.front() : outPoses.back();
    PredictionDebugInfo predictionDebugInfo{};
    PredictionCovariance predictionCovariance{};
    if (posePrediction && outPoses.size() >= 3) {
        auto& lastPose = outPoses[outPoses.size() - 1];
        auto& prevPose = outPoses[outPoses.size() - 2];
        auto& secondPrevPose = outPoses[outPoses.size() - 3];
        predictionDebugInfo.valid = true;
        predictionDebugInfo.usedPrediction = true;
        predictionDebugInfo.latestTimestampUs = static_cast<int64_t>(lastPose.send_timestamp);
        predictionDebugInfo.previousTimestampUs = static_cast<int64_t>(prevPose.send_timestamp);
        predictionDebugInfo.secondPreviousTimestampUs = static_cast<int64_t>(secondPrevPose.send_timestamp);
        predictionDebugInfo.predictedTimestampUs = static_cast<int64_t>(timeutils::secondsToMicros(now + dtFuture + jitterPredicted));

        if (!getPosePredicted(
                poseToSend,
                lastPose,
                prevPose,
                secondPrevPose,
                now + dtFuture + jitterPredicted,
                &predictionCovariance))
        {
            return false;
        }
    }

    double timestampS = timeutils::microsToSeconds(outPoses.front().send_timestamp);
    if (networkLatencyS > 0 && now - timestampS < dtFuture + actualOutJitter) return false;

    actualOutJitter = randomJitter();

    double oldTimestampS = outOrigTimestamps.front();
    rtts.push_back(timeutils::secondsToMillis(now - oldTimestampS));

    pose = poseToSend;
    lastPredictionDebugInfo = predictionDebugInfo;
    lastPredictionCovariance = predictionCovariance;
    if (!posePrediction || outPoses.size() >= 3) {
        outPoses.pop_front();
        outOrigTimestamps.pop_front();
    }

    return true;
}

bool PoseSendRecvSimulator::predictPose(
    Pose& predictedPose,
    const Pose& latest,
    const Pose& previous,
    const Pose& secondPrevious,
    double targetFutureTimeS,
    PredictionCovariance* covariance)
{
    return getPosePredicted(predictedPose, latest, previous, secondPrevious, targetFutureTimeS, covariance);
}

std::vector<Pose> PoseSendRecvSimulator::samplePredictionCovariance(
    const Pose& meanPose,
    const PredictionCovariance& covariance,
    size_t sampleCount,
    double confidence,
    double minPositionSeparationM,
    double minRotationSeparationDeg) const
{
    std::vector<Pose> sampledPoses;
    if (!covariance.valid || sampleCount == 0) {
        return sampledPoses;
    }

    const double radius = confidenceRadius6D(confidence);
    const std::array<double, 36> lower = choleskyLower6x6(covariance.pose6x6);
    const std::vector<std::array<double, 6>> directions = buildSeparatedUnitDirections6D();
    const double minRotationSeparationRad = glm::radians(minRotationSeparationDeg);

    std::vector<std::array<double, 6>> candidateOffsets;
    candidateOffsets.reserve(directions.size());
    for (const std::array<double, 6>& direction : directions) {
        candidateOffsets.push_back(transformUnitDirection6D(lower, direction, radius));
    }

    std::vector<std::array<double, 6>> selectedOffsets;
    selectedOffsets.reserve(sampleCount);
    while (selectedOffsets.size() < sampleCount && !candidateOffsets.empty()) {
        double bestScore = -std::numeric_limits<double>::infinity();
        size_t bestIndex = 0;

        for (size_t candidateIndex = 0; candidateIndex < candidateOffsets.size(); ++candidateIndex) {
            double minDistance = std::numeric_limits<double>::infinity();
            if (selectedOffsets.empty()) {
                std::array<double, 6> zeroOffset{};
                minDistance = covarianceOffsetDistance(
                    candidateOffsets[candidateIndex],
                    zeroOffset,
                    minPositionSeparationM,
                    minRotationSeparationRad);
            }
            else {
                for (const std::array<double, 6>& selectedOffset : selectedOffsets) {
                    minDistance = std::min(
                        minDistance,
                        covarianceOffsetDistance(
                            candidateOffsets[candidateIndex],
                            selectedOffset,
                            minPositionSeparationM,
                            minRotationSeparationRad));
                }
            }

            if (minDistance > bestScore) {
                bestScore = minDistance;
                bestIndex = candidateIndex;
            }
        }

        if (bestScore < 1.0) {
            spdlog::warn(
                "Could only select {} covariance pose samples with the requested separation "
                "({:.4f}m, {:.4f}deg) inside the {:.1f}% confidence region",
                selectedOffsets.size(),
                minPositionSeparationM,
                minRotationSeparationDeg,
                confidence * 100.0);
            break;
        }

        selectedOffsets.push_back(candidateOffsets[bestIndex]);
        candidateOffsets.erase(candidateOffsets.begin() + static_cast<std::ptrdiff_t>(bestIndex));
    }

    sampledPoses.reserve(selectedOffsets.size());
    for (const std::array<double, 6>& offset : selectedOffsets) {
        sampledPoses.push_back(poseFromCovarianceOffset(meanPose, offset));
    }

    return sampledPoses;
}

void PoseSendRecvSimulator::accumulateError(const PerspectiveCamera& camera, const PerspectiveCamera& remoteCamera) {
    float positionDiff = glm::distance(camera.getPosition(), remoteCamera.getPosition());
    glm::quat q1 = glm::normalize(camera.getRotationQuat());
    glm::quat q2 = glm::normalize(remoteCamera.getRotationQuat());
    if (glm::dot(q1, q2) < 0.0f) q2 = -q2;

    float angleDiffRadians = 2.0f * glm::acos(glm::clamp(glm::dot(q1, q2), -1.0f, 1.0f));
    float angleDiffDegrees = glm::degrees(angleDiffRadians);

    positionErrors.push_back(positionDiff);
    rotationErrors.push_back(std::abs(angleDiffDegrees));
}

PoseSendRecvSimulator::ErrorStats PoseSendRecvSimulator::getAvgErrors() {
    ErrorStats stats;
    stats.positionErrMeanStd.x = calculateMean(positionErrors);
    stats.positionErrMeanStd.y = calculateStdDev(positionErrors, stats.positionErrMeanStd.x);
    stats.positionErrMinMax.x = *std::min_element(positionErrors.begin(), positionErrors.end());
    stats.positionErrMinMax.y = *std::max_element(positionErrors.begin(), positionErrors.end());

    stats.rotationErrMeanStd.x = calculateMean(rotationErrors);
    stats.rotationErrMeanStd.y = calculateStdDev(rotationErrors, stats.rotationErrMeanStd.x);
    stats.rotationErrMinMax.x = *std::min_element(rotationErrors.begin(), rotationErrors.end());
    stats.rotationErrMinMax.y = *std::max_element(rotationErrors.begin(), rotationErrors.end());

    stats.rttMeanStd.x = calculateMean(rtts);
    stats.rttMeanStd.y = calculateStdDev(rtts, stats.rttMeanStd.x);

    return stats;
}

void PoseSendRecvSimulator::printErrors() {
    ErrorStats stats = getAvgErrors();
    spdlog::info("Pose Error:");
    spdlog::info("  Pos ({:.2f}±{:.2f},[{:.1f},{:.2f}])m", stats.positionErrMeanStd.x, stats.positionErrMeanStd.y, stats.positionErrMinMax.x, stats.positionErrMinMax.y);
    spdlog::info("  Rot ({:.2f}±{:.2f},[{:.1f},{:.2f}])°", stats.rotationErrMeanStd.x, stats.rotationErrMeanStd.y, stats.rotationErrMinMax.x, stats.rotationErrMinMax.y);
    spdlog::info("  RTT ({:.2f}±{:.2f})ms", stats.rttMeanStd.x, stats.rttMeanStd.y);
}

glm::vec3 PoseSendRecvSimulator::savitzkyGolayFilter(const std::deque<glm::vec3>& buffer) {
    if (buffer.size() < 5) return buffer.back();
    static const std::array<float, 5> coeffs = {
        -3.0f / 35.0f, 12.0f / 35.0f, 17.0f / 35.0f, 12.0f / 35.0f, -3.0f / 35.0f
    };
    glm::vec3 result(0.0f);
    for (int i = 0; i < 5; i++) {
        result += coeffs[i] * buffer[buffer.size() - 5 + i];
    }
    return result;
}

glm::quat PoseSendRecvSimulator::averageQuaternions(const std::deque<glm::quat>& quats) {
    if (quats.empty()) return glm::quat(1, 0, 0, 0);
    glm::quat avg = quats[0];
    for (size_t i = 1; i < quats.size(); i++) {
        if (glm::dot(avg, quats[i]) < 0.0f)
            avg = glm::slerp(avg, -quats[i], 1.0f / (i + 1));
        else
            avg = glm::slerp(avg, quats[i], 1.0f / (i + 1));
    }
    return glm::normalize(avg);
}

double PoseSendRecvSimulator::randomJitter() {
    if (networkJitterS <= 0.0) {
        return 0.0;
    }
    return distribution(generator);
}

double PoseSendRecvSimulator::calculateMean(const std::vector<double>& errors) const {
    if (errors.empty()) return 0.0;
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

double PoseSendRecvSimulator::calculateStdDev(const std::vector<double>& errors, double mean) const {
    if (errors.size() < 2) return 0.0;
    double sumSquaredDiffs = 0.0;
    for (double err : errors) {
        sumSquaredDiffs += (err - mean) * (err - mean);
    }
    return std::sqrt(sumSquaredDiffs / (errors.size() - 1));
}

bool PoseSendRecvSimulator::getPosePredicted(
    Pose& predictedPose,
    const Pose& latest, const Pose& previous, const Pose& secondPrevious,
    double targetFutureTimeS,
    PredictionCovariance* covariance)
{
    if (covariance != nullptr) {
        *covariance = {};
    }

    double t2 = timeutils::microsToSeconds(secondPrevious.send_timestamp);
    double t1 = timeutils::microsToSeconds(previous.send_timestamp);
    double t0 = timeutils::microsToSeconds(latest.send_timestamp);

    float dt1 = t1 - t2;
    float dt2 = t0 - t1;

    if (dt1 <= 0.0f || dt2 <= 0.0f) return false;

    float dtFuture = targetFutureTimeS - t0;
    const float maxPredictTime = 0.1f;
    dtFuture = glm::clamp(dtFuture, 0.0f, maxPredictTime);

    glm::vec3 scale, skew;
    glm::vec4 perspective;

    glm::vec3 p2, p1, p0;
    glm::quat r2, r1, r0;

    glm::decompose(glm::inverse(secondPrevious.mono.view), scale, r2, p2, skew, perspective);
    glm::decompose(glm::inverse(previous.mono.view), scale, r1, p1, skew, perspective);
    glm::decompose(glm::inverse(latest.mono.view), scale, r0, p0, skew, perspective);

    if (glm::dot(r1, r0) < 0.0f) r1 = -r1;
    if (glm::dot(r2, r1) < 0.0f) r2 = -r2;

    glm::vec3 filteredP0 = poseSmoothing ? savitzkyGolayFilter([&] {
        positionHistory.push_back(p0);
        if (positionHistory.size() > maxPositionHistorySize) positionHistory.pop_front();
        return positionHistory;
    }()) : p0;

    glm::vec3 v1 = (p1 - p2) / dt1;
    glm::vec3 v2 = (p0 - p1) / dt2;
    glm::vec3 v = 0.5f * (v1 + v2);
    glm::vec3 velocityDisagreement = v2 - v1;
    glm::vec3 a = velocityDisagreement / dt2;
    a = glm::clamp(a, -3.0f, 3.0f);

    glm::vec3 rawPrediction = filteredP0 + v * dtFuture + 0.5f * a * dtFuture * dtFuture;

    float confidence = 1.0f - glm::smoothstep(0.02f, 0.06f, dtFuture);
    glm::vec3 finalPrediction = poseSmoothing ? glm::mix(filteredP0, rawPrediction, confidence) : rawPrediction;

    glm::quat dq = glm::normalize(r0 * glm::inverse(r1));
    float angle = glm::angle(dq);
    glm::vec3 axis = glm::axis(dq);
    if (glm::length(axis) < 1e-5f || glm::any(glm::isnan(axis))) axis = glm::vec3(0, 1, 0);
    float angularSpeed = angle / dt2;

    angularSpeed = glm::clamp(angularSpeed, 0.0f, glm::radians(200.0f));
    float futureAngle = angularSpeed * dtFuture;
    futureAngle = glm::clamp(futureAngle, 0.0f, glm::radians(45.0f));

    glm::quat deltaFuture = glm::angleAxis(futureAngle, axis);
    glm::quat predictedRotation = glm::normalize(deltaFuture * r0);

    glm::quat finalRotation = poseSmoothing ? averageQuaternions([&] {
        rotationHistory.push_back(predictedRotation);
        if (rotationHistory.size() > maxRotationHistorySize) rotationHistory.pop_front();
        return rotationHistory;
    }()) : predictedRotation;

    if (covariance != nullptr) {
        covariance->valid = true;
        covariance->targetFutureTimeS = targetFutureTimeS;
        covariance->dtFutureS = dtFuture;

        const double jitterVarianceS2 = networkJitterS > 0.0
            ? (networkJitterS * networkJitterS) / 3.0
            : 0.0;

        const glm::vec3 positionModelUncertainty =
            velocityDisagreement * dtFuture + 0.5f * a * dtFuture * dtFuture;
        addOuterProduct(covariance->positionCovariance, positionModelUncertainty);
        if (jitterVarianceS2 > 0.0) {
            addOuterProduct(covariance->positionCovariance, v, jitterVarianceS2);
        }
        addDiagonalFloor(covariance->positionCovariance);

        const glm::vec3 angularVelocity1 = rotationDeltaVector(r1, r2) / dt1;
        const glm::vec3 angularVelocity2 = rotationDeltaVector(r0, r1) / dt2;
        const glm::vec3 angularVelocityDisagreement = angularVelocity2 - angularVelocity1;
        const glm::vec3 angularAcceleration = angularVelocityDisagreement / dt2;
        const glm::vec3 rotationModelUncertainty =
            angularVelocityDisagreement * dtFuture + 0.5f * angularAcceleration * dtFuture * dtFuture;
        addOuterProduct(covariance->rotationCovariance, rotationModelUncertainty);
        if (jitterVarianceS2 > 0.0) {
            addOuterProduct(covariance->rotationCovariance, angularVelocity2, jitterVarianceS2);
        }
        addDiagonalFloor(covariance->rotationCovariance);

        fillPoseCovariance6x6(*covariance);
    }

    glm::mat4 predictedTransform = glm::translate(glm::mat4(1.0f), finalPrediction) * glm::mat4_cast(finalRotation);
    glm::mat4 predictedView = glm::inverse(predictedTransform);

    predictedPose.setViewMatrix(predictedView);
    predictedPose.setProjectionMatrix(latest.mono.proj);
    predictedPose.send_timestamp = static_cast<double>(timeutils::secondsToMicros(targetFutureTimeS));

    spdlog::info("  Latest Position:   ({:.3f}, {:.3f}, {:.3f})", p0.x, p0.y, p0.z);
    spdlog::info("  Previous Position: ({:.3f}, {:.3f}, {:.3f})", p1.x, p1.y, p1.z);
    spdlog::info("  Previous but two position: ({:.3f}, {:.3f}, {:.3f})", p2.x, p2.y, p2.z);

    spdlog::info("  Predicted Position: ({:.3f}, {:.3f}, {:.3f})", finalPrediction.x, finalPrediction.y, finalPrediction.z);

    return true;
}
