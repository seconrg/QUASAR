#include <Quads/QuadTexelUsageMaterial.h>
#include <shaders_common.h>

#include <vector>

using namespace quasar;

QuadTexelUsageMaterial::QuadTexelUsageMaterial(const QuadMaterialCreateParams& params)
    : baseColor(params.baseColor)
    , baseColorFactor(params.baseColorFactor)
    , alphaMode(params.alphaMode) {
    TextureFileCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR,
    };

    if (params.baseColorTexturePath != "") {
        textureParams.path = params.baseColorTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.baseColorTexture);
    }

    if (params.alphaTexturePath != "") {
        textureParams.path = params.alphaTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.alphaTexture);
    }

    std::vector<std::string> defines = {
        "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::OPAQUE)),
        "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::MASKED)),
        "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::TRANSPARENT)),
    };

    ShaderDataCreateParams shaderParams{
        .version = "420 core",
        .vertexCodeData = SHADER_COMMON_MATERIAL_QUAD_VERT,
        .vertexCodeSize = SHADER_COMMON_MATERIAL_QUAD_VERT_len,
        .fragmentCodeData = SHADER_COMMON_MATERIAL_QUAD_TEXEL_USAGE_FRAG,
        .fragmentCodeSize = SHADER_COMMON_MATERIAL_QUAD_TEXEL_USAGE_FRAG_len,
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

void QuadTexelUsageMaterial::bind() const {
    texelUsageShader->bind();
    texelUsageShader->setVec4("material.baseColor", baseColor);
    texelUsageShader->setVec4("material.baseColorFactor", baseColorFactor);
    texelUsageShader->setInt("material.alphaMode", static_cast<int>(alphaMode));

    std::string name = "material.baseColorMap";
    glActiveTexture(GL_TEXTURE0);
    texelUsageShader->setBool("material.hasBaseColorMap", textures[0] != nullptr);

    if (textures[0] != nullptr) {
        texelUsageShader->setTexture(name, *textures[0], 0);
    }
    else {
        texelUsageShader->setInt(name, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    name = "material.alphaMap";
    glActiveTexture(GL_TEXTURE1);
    texelUsageShader->setBool("material.hasAlphaMap", textures.size() > 1 && textures[1] != nullptr);
    if (textures.size() > 1 && textures[1] != nullptr) {
        texelUsageShader->setTexture(name, *textures[1], 1);
    }
    else {
        texelUsageShader->setInt(name, 1);
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

void QuadTexelUsageMaterial::unbind() const {
    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    Material::unbind();
}

void QuadTexelUsageMaterial::clearTexelUsageMap() {
    if (usageWriteTexture == nullptr) {
        return;
    }
    const size_t n = static_cast<size_t>(usageWriteTexture->width) * static_cast<size_t>(usageWriteTexture->height) * 4u;
    std::vector<unsigned char> zeros(n, 0);
    usageWriteTexture->loadFromData(zeros.data(), false);
}
