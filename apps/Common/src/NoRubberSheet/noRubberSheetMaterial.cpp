#include <NoRubberSheet/noRubberSheetMaterial.h>
#include <shaders_common.h>

using namespace quasar;

NoRubberSheetMaterial::NoRubberSheetMaterial(const UnlitMaterialCreateParams& params)
    : UnlitMaterial(params)
{
    // delete the previous shader (Override the previous shader)
    shader = nullptr;
    std::vector<std::string> defines = {
        "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::OPAQUE)),
        "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::MASKED)),
        "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::TRANSPARENT))
    };
    for (const auto& define : extraShaderDefines) {
        defines.push_back(define);
    }

    ShaderDataCreateParams unlitShaderParams{
        .vertexCodeData = SHADER_COMMON_NORUBBERSHEETMESH_VERT,
        .vertexCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_VERT_len,
        .fragmentCodeData = SHADER_COMMON_NORUBBERSHEETMESH_FRAG,
        .fragmentCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_FRAG_len,
        .defines = defines,
    };
    shader = std::make_shared<Shader>(unlitShaderParams);
}
