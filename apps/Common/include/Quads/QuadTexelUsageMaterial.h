#ifndef QUAD_TEXEL_USAGE_MATERIAL_H
#define QUAD_TEXEL_USAGE_MATERIAL_H

#include <Materials/Material.h>
#include <Quads/QuadMaterial.h>
#include <Texture.h>
#include <memory>

namespace quasar {

/**
 * QuadMesh / material_quad pipeline: same shading as QuadMaterial, plus an RGBA8 image matching the
 * base-color map resolution where each surviving fragment that sampled the map marks that texel blue
 * (same idea as TexelUsageMaterial, which uses the no-rubber-sheet vertex shader).
 */
class QuadTexelUsageMaterial : public Material {
public:
    glm::vec4 baseColor;
    glm::vec4 baseColorFactor;
    Material::AlphaMode alphaMode;

    explicit QuadTexelUsageMaterial(const QuadMaterialCreateParams& params);

    void bind() const override;
    /// Not a virtual override; hides Material::unbind to also release image binding.
    void unbind() const;

    std::shared_ptr<Shader> getShader() const override { return texelUsageShader; }

    uint getTextureCount() const override { return 1; }

    Texture* getTexelUsageTexture() { return usageWriteTexture.get(); }
    const Texture* getTexelUsageTexture() const { return usageWriteTexture.get(); }

    void clearTexelUsageMap();

private:
    std::shared_ptr<Shader> texelUsageShader;
    std::unique_ptr<Texture> usageWriteTexture;
};

} // namespace quasar

#endif
