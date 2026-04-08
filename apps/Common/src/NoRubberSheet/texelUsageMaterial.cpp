#include <NoRubberSheet/texelUsageMaterial.h>
#include <shaders_common.h>

#include <vector>

using namespace quasar;

TexelUsageMaterial::TexelUsageMaterial(const UnlitMaterialCreateParams& params)
    : UnlitMaterial(params) {
    std::vector<std::string> defines = {
        "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::OPAQUE)),
        "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::MASKED)),
        "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::TRANSPARENT)),
    };
    for (const auto& define : UnlitMaterial::extraShaderDefines) {
        defines.push_back(define);
    }

    ShaderDataCreateParams shaderParams{
        .version = "420 core",
        .vertexCodeData = SHADER_COMMON_NORUBBERSHEETMESH_VERT,
        .vertexCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_VERT_len,
        .fragmentCodeData = SHADER_COMMON_NORUBBERSHEETMESH_TEXEL_USAGE_FRAG,
        .fragmentCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_TEXEL_USAGE_FRAG_len,
        .defines = defines,
    };
    texelUsageShader = std::make_shared<Shader>(shaderParams);

    uint32_t w = 1;
    uint32_t h = 1;
    if (textures[0] != nullptr) {
        w = textures[0]->width;
        h = textures[0]->height;
    }
    usageWriteTexture = std::make_unique<Texture>(TextureDataCreateParams{
        .width = w,
        .height = h,
        .internalFormat = GL_RGBA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
        .data = nullptr,
    });
    clearTexelUsageMap();
}

void TexelUsageMaterial::bind() const {
    texelUsageShader->bind();
    texelUsageShader->setVec4("material.baseColor", baseColor);
    texelUsageShader->setVec4("material.baseColorFactor", baseColorFactor);
    texelUsageShader->setInt("material.alphaMode", static_cast<int>(alphaMode));
    texelUsageShader->setFloat("material.maskThreshold", maskThreshold);

    const std::string name = "material.baseColorMap";
    glActiveTexture(GL_TEXTURE0);
    texelUsageShader->setBool("material.hasBaseColorMap", textures[0] != nullptr);

    if (textures[0] != nullptr) {
        texelUsageShader->setTexture(name, *textures[0], 0);
    }
    else {
        texelUsageShader->setInt(name, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (usageWriteTexture != nullptr) {
        glBindImageTexture(
            0,
            usageWriteTexture->ID,
            0,
            GL_FALSE,
            0,
            GL_WRITE_ONLY,
            GL_RGBA8);
    }
}

void TexelUsageMaterial::unbind() const {
    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    UnlitMaterial::unbind();
}

void TexelUsageMaterial::clearTexelUsageMap() {
    if (usageWriteTexture == nullptr) {
        return;
    }
    const size_t n = static_cast<size_t>(usageWriteTexture->width) * static_cast<size_t>(usageWriteTexture->height) * 4u;
    std::vector<unsigned char> zeros(n, 0);
    usageWriteTexture->loadFromData(zeros.data(), false);
}
