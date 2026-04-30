#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <random>
#include <deque>
#include <vector>
#include <numeric>
#include <cmath>
#include <array>

#include <Cameras/PerspectiveCamera.h>
#include <CameraPose.h>

namespace quasar {

struct PoseSendRecvSimulatorCreateParams {
    double networkLatencyMs;
    double networkJitterMs;
    double renderTimeMs;
    bool posePrediction = false;
    bool poseSmoothing = false;
    uint seed = 42;
};

class PoseSendRecvSimulator {
public:
    bool posePrediction;
    bool poseSmoothing;

    struct PredictionDebugInfo {
        bool valid = false;
        bool usedPrediction = false;
        int64_t latestTimestampUs = -1;
        int64_t previousTimestampUs = -1;
        int64_t secondPreviousTimestampUs = -1;
        int64_t predictedTimestampUs = -1;
    };

    struct ErrorStats {
        glm::vec2 positionErrMeanStd;
        glm::vec2 positionErrMinMax;
        glm::vec2 rotationErrMeanStd;
        glm::vec2 rotationErrMinMax;
        glm::vec2 rttMeanStd;
    };

    struct PredictionCovariance {
        bool valid = false;
        double targetFutureTimeS = 0.0;
        double dtFutureS = 0.0;
        // Position covariance in m^2.
        glm::mat3 positionCovariance{0.0f};
        // Rotation covariance in rad^2, represented in local axis-angle tangent space.
        glm::mat3 rotationCovariance{0.0f};
        // Row-major 6x6 covariance: [x, y, z, rot_x, rot_y, rot_z].
        std::array<double, 36> pose6x6{};
    };

    PoseSendRecvSimulator(PoseSendRecvSimulatorCreateParams params);

    void setNetworkLatency(double networkLatencyMs);
    void setNetworkJitter(double networkJitterMs);
    void setRenderTime(double renderTimeMs);
    void clear();

    void sendPose(const PerspectiveCamera& camera, double now);
    void update(float now);
    bool recvPoseToRender(Pose& pose, double now);
    bool predictPose(
        Pose& predictedPose,
        const Pose& latest,
        const Pose& previous,
        const Pose& secondPrevious,
        double targetFutureTimeS,
        PredictionCovariance* covariance = nullptr);
    std::vector<Pose> samplePredictionCovariance(
        const Pose& meanPose,
        const PredictionCovariance& covariance,
        size_t sampleCount = 5,
        double confidence = 0.99,
        double minPositionSeparationM = 0.005,
        double minRotationSeparationDeg = 0.05) const;

    void accumulateError(const PerspectiveCamera& camera, const PerspectiveCamera& remoteCamera);
    ErrorStats getAvgErrors();
    void printErrors();
    const PredictionDebugInfo& getLastPredictionDebugInfo() const { return lastPredictionDebugInfo; }
    const PredictionCovariance& getLastPredictionCovariance() const { return lastPredictionCovariance; }

private:
    double networkLatencyS;
    double networkJitterS;
    double renderTimeS;

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    double lastUpdateTimeS = -1.0;
    std::deque<Pose> incomingPoses;
    std::deque<Pose> outPoses;
    std::deque<double> outOrigTimestamps;

    std::vector<double> positionErrors;
    std::vector<double> rotationErrors;
    std::vector<double> rtts;

    double actualInJitter;
    double actualOutJitter;

    std::deque<glm::vec3> positionHistory;
    static constexpr size_t maxPositionHistorySize = 10;

    std::deque<glm::quat> rotationHistory;
    static constexpr size_t maxRotationHistorySize = 5;

    PredictionDebugInfo lastPredictionDebugInfo;
    PredictionCovariance lastPredictionCovariance;

    glm::vec3 savitzkyGolayFilter(const std::deque<glm::vec3>& buffer);
    glm::quat averageQuaternions(const std::deque<glm::quat>& quats);
    double randomJitter();
    double calculateMean(const std::vector<double>& errors) const;
    double calculateStdDev(const std::vector<double>& errors, double mean) const;

    bool getPosePredicted(
        Pose& predictedPose,
        const Pose& latest, const Pose& previous, const Pose& secondPrevious,
        double targetFutureTimeS,
        PredictionCovariance* covariance = nullptr);
};

} // namespace quasar

#endif // POSE_SIM_H
