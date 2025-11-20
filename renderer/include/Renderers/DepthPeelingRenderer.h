#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/GBuffer.h>
#include <RenderTargets/FrameRenderTarget.h>

namespace quasar {

class DepthPeelingRenderer : public DeferredRenderer {
public:
    uint maxLayers;
    std::vector<int> layerIndices;
    float viewSphereDiameter = 0.5f;
    float edpDelta = 0.0005f;

    std::vector<FrameRenderTarget> peelingLayers;

    DepthPeelingRenderer(const Config& config, uint maxLayers = 4, const std::vector<int>& layerIndices = {}, bool edp = false);
    ~DepthPeelingRenderer() = default;

    void setViewSphereDiameter(float viewSphereDiameter) { this->viewSphereDiameter = viewSphereDiameter; }
    virtual void setScreenShaderUniforms(const Shader& screenShader) override;

    virtual void resize(uint width, uint height) override;

    virtual RenderStats drawSceneByLayer(Scene& scene, const Camera& camera, uint32_t layerIndex, uint32_t clearMask);
    virtual RenderStats drawScene(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, bool renderFrontEnd = true, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    virtual RenderStats drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    RenderStats compositeLayers();

private:
    bool edp;
    Shader compositeLayersShader;
};

} // namespace quasar

#endif // DEPTH_PEELING_H
