#ifndef TEXEL_USAGE_MATERIAL_H
#define TEXEL_USAGE_MATERIAL_H

#include <Materials/UnlitMaterial.h>
#include <Texture.h>
#include <memory>

namespace quasar {

/**
 * Same shading as NoRubberSheetMaterial (closest-Z rejection), plus an RGBA8 image matching the
 * base-color map resolution: every surviving fragment that sampled the map sets that texel to blue
 * in `getTexelUsageTexture()` via `imageStore`.
 *
 * QuadMesh proxies (material_quad) use QuadTexelUsageMaterial — same texel-usage image idea, quad vertex shader.
 */
class TexelUsageMaterial : public UnlitMaterial {
public:
    explicit TexelUsageMaterial(const UnlitMaterialCreateParams& params);

    void bind() const override;
    /// Not a virtual override; hides Material::unbind to also release image binding.
    void unbind() const;

    std::shared_ptr<Shader> getShader() const override { return texelUsageShader; }

    /// Writable usage map; same width/height as base color texture when present, else 1×1.
    Texture* getTexelUsageTexture() { return usageWriteTexture.get(); }
    const Texture* getTexelUsageTexture() const { return usageWriteTexture.get(); }

    /// Clear usage map to transparent black (call before a pass where you want a fresh mask).
    void clearTexelUsageMap();

private:
    std::shared_ptr<Shader> texelUsageShader;
    std::unique_ptr<Texture> usageWriteTexture;
};

} // namespace quasar

#endif
