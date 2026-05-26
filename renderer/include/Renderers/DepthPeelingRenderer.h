#ifndef DEPTH_PEELING_H
#define DEPTH_PEELING_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/GBuffer.h>
#include <RenderTargets/FrameRenderTarget.h>
#include <glm/glm.hpp>

namespace quasar {

class DepthPeelingRenderer : public DeferredRenderer {
public:
    uint maxLayers;
    float viewSphereDiameter = 0.5f;
    float edpDelta = 0.0005f;
    float eOverride = -1.0f;
    glm::vec2 eScreenDirection = glm::vec2(1.0f, 0.0f);
    float ePerpendicularScale = 1.0f;
    bool eScreenSpaceFootprint = false;
    float eScreenMajorRadiusPx = 0.0f;
    float eScreenMinorRadiusPx = 0.0f;
    bool eViewOffsetFootprint = false;
    glm::vec3 eViewOffsetUncertaintyM{0.0f};

    std::vector<FrameRenderTarget> peelingLayers;

    DepthPeelingRenderer(const Config& config, uint maxLayers = 4, bool edp = false);
    ~DepthPeelingRenderer() = default;

    void setViewSphereDiameter(float viewSphereDiameter) { this->viewSphereDiameter = viewSphereDiameter; }
    void setEOverride(float eRadius) { eOverride = eRadius; }
    void clearEOverride() { eOverride = -1.0f; }
    void setEAnisotropy(const glm::vec2& screenDirection, float perpendicularScale);
    void clearEAnisotropy();
    void setEScreenSpaceFootprint(float majorRadiusPx, float minorRadiusPx);
    void clearEScreenSpaceFootprint();
    void setEViewOffsetFootprint(const glm::vec3& viewOffsetUncertaintyM);
    void clearEViewOffsetFootprint();
    float getEffectiveE() const { return eOverride >= 0.0f ? eOverride : viewSphereDiameter / 2.0f; }
    virtual void setScreenShaderUniforms(const Shader& screenShader) override;

    virtual void resize(uint width, uint height) override;

    virtual RenderStats drawScene(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;
    virtual RenderStats drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT) override;

    RenderStats compositeLayers();

private:
    bool edp;
    Shader compositeLayersShader;
};

} // namespace quasar

#endif // DEPTH_PEELING_H
