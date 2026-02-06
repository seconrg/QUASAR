#ifndef POSE_RECEIVER_H
#define POSE_RECEIVER_H

#include <chrono>
#include <thread>
#include <cstring>
#include <deque>

#include <spdlog/spdlog.h>

#include <Networking/Socket.h>

#include <glm/gtc/type_ptr.hpp>

#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>
#include <Networking/DataReceiverUDP.h>

#include <CameraPose.h>

namespace quasar {

class PoseReceiver : public DataReceiverUDP {
public:
    std::string streamerURL;

    PoseReceiver(Camera* camera, const std::string& streamerURL, float poseDropThresMs = 50.0f, size_t maxSavedPoses = 2)
        : camera(camera)
        , streamerURL(streamerURL)
        , poseDropThresUs(timeutils::millisToMicros(poseDropThresMs))
        , maxSavedPoses(maxSavedPoses)
        , DataReceiverUDP(streamerURL, sizeof(Pose))
    {
        if (!streamerURL.empty()) {
            spdlog::info("Created PoseReceiver that recvs from URL: udp://{}", this->streamerURL);
        }
    }
    ~PoseReceiver() = default;

    void onDataReceived(const std::vector<char>& data) override {
        std::lock_guard<std::mutex> lock(m);

        if (data.size() < sizeof(Pose)) {
            spdlog::warn("Received data size is smaller than expected Pose size");
            return;
        }

        Pose newPose;
        std::memcpy(&newPose, data.data(), sizeof(Pose));

        newPose.recv_timestamp = timeutils::getTimeMicros();

        // Avoid adding outdated poses
        if (!poseQueue.empty() && newPose.send_timestamp - poseQueue.back().send_timestamp <= poseDropThresUs) {
            return;
        }

        poseQueue.push_back(newPose);
        if (poseQueue.size() > maxSavedPoses) {
            poseQueue.pop_front();
        }
    }
    
    // 
    typedef struct PoseInfo {
        double send_timestamp;
        double recv_timestamp;
        pose_id_t pose_id;
    } poseInfo;
    
    poseInfo receivePose(bool setProj = true) {
        std::lock_guard<std::mutex> lock(m);

        if (poseQueue.empty()) {
            return poseInfo{0.0, 0.0, pose_id_t(-1)};
        }

        Pose pose = poseQueue.front();
        poseQueue.pop_front();

        if (camera->isVR()) {
            auto* vrCamera = static_cast<VRCamera*>(camera);
            if (setProj) {
                vrCamera->setProjectionMatrices({pose.stereo.projL, pose.stereo.projR});
            }
            vrCamera->setViewMatrices({pose.stereo.viewL, pose.stereo.viewR});
        }
        else {
            auto* perspectiveCamera = static_cast<PerspectiveCamera*>(camera);
            if (setProj) {
                perspectiveCamera->setProjectionMatrix(pose.mono.proj);
            }
            perspectiveCamera->setViewMatrix(pose.mono.view);
        }

        return poseInfo{pose.send_timestamp, pose.recv_timestamp, pose.id};
    }

private:
    Camera* camera;
    float poseDropThresUs;
    size_t maxSavedPoses;

    std::mutex m;
    std::deque<Pose> poseQueue;
};

} // namespace quasar

#endif // POSE_RECEIVER_H
