#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>

#include <Receivers/HybridReceiver.h>
#include <Networking/DataReceiverTCP.h>
#include <Receivers/VideoTexture.h>
#include <Codecs/AlphaCodec.h>

namespace quasar {

class HybridReceiver : public DataReceiverTCP {
public: 
    struct Params {
        uint32_t numLayers;
        float viewSphereDiameter;
        float wideFOV;
    };

    struct Header {
        pose_id_t poseID;
        Params params;

        // Size for visible layers
        uint32_t visibleLayerSize;
        uint32_t visibleLayerWideFovSize;
        
        // Size for depth peeling hidden layers
        uint32_t cameraSize;
        uint32_t alphaSize;
        uint32_t geometrySize;

        size_t getSize() const { return sizeof(Header) + cameraSize + alphaSize + geometrySize + visibleLayerSize + visibleLayerWideFovSize; }

    };

    std::string videoURL;
    std::string depthAndProxiesURL;

    uint hiddenLayers;
    float viewSphereDiameter;

    VideoTexture videoAtlasTexture;
    Texture alphaAtlasTexture;

    HybridReceiver(QuadSet& quadSet, uint hiddenLayers, 
                   const std::string& videoURL = "", 
                   const std::string& depthAndProxiesURL = "");
    
    HybridReceiver(QuadSet& quadSet, uint maxLayers, 
                   float remoteFOV, float remoteFOVWide, 
                   const std::string& videoURL = "", 
                   const std::string& proxiesURL = "");
    ~HybridReceiver() = default;

    QuadMesh& getMesh(int layer) { return meshesHidLayer[layer]; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }

    void loadFromMemory(const std::vector<char>& inputData);


    void recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFOV;

};


}



#endif // QUASAR_RECEIVER_H