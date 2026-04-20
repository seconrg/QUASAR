#ifndef HOLE_FILLER_H
#define HOLE_FILLER_H

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class HoleFiller : public PostProcessingEffect {
public:
    HoleFiller()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_COMMON_HOLE_FILLER_FRAG,
            .fragmentCodeSize = SHADER_COMMON_HOLE_FILLER_FRAG_len,
        })
    {}

    void setDepthThreshold(float depthThreshold) {
        shader.bind();
        shader.setFloat("depthThreshold", depthThreshold);
    }

    void enableTonemapping(bool enable) {
        shader.bind();
        shader.setBool("tonemap", enable);
    }

    void setWideFovFillTextures(const Texture* colorTexture, const Texture* alphaTexture) {
        wideFovFillColorTexture = colorTexture;
        wideFovFillAlphaTexture = alphaTexture;
        shader.bind();
        shader.setBool("useWideFovFill", wideFovFillColorTexture != nullptr && wideFovFillAlphaTexture != nullptr);
    }

    void setWideFovFillMapping(const glm::vec3 cornersInWideFovImage[4]) {
        for (int i = 0; i < 4; ++i) {
            wideFovNormalViewCorners[i] = cornersInWideFovImage[i];
        }
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        renderer.setScreenShaderUniforms(shader);
        bindWideFovFillTextures();
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        renderer.setScreenShaderUniforms(shader);
        bindWideFovFillTextures();
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    void bindWideFovFillTextures() {
        shader.bind();
        if (wideFovFillColorTexture != nullptr && wideFovFillAlphaTexture != nullptr) {
            shader.setTexture("wideFovFillColor", *wideFovFillColorTexture, 5);
            shader.setTexture("wideFovFillAlpha", *wideFovFillAlphaTexture, 6);
            for (int i = 0; i < 4; ++i) {
                shader.setVec2(
                    "wideFovNormalViewCorners[" + std::to_string(i) + "]",
                    glm::vec2(wideFovNormalViewCorners[i]));
            }
        }
    }

    Shader shader;
    const Texture* wideFovFillColorTexture = nullptr;
    const Texture* wideFovFillAlphaTexture = nullptr;
    glm::vec3 wideFovNormalViewCorners[4] = {
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f),
    };
};

} // namespace quasar

#endif // HOLE_FILLER_H
