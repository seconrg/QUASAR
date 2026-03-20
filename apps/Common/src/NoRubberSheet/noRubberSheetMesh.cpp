#include "NoRubberSheet/noRubberSheetMaterial.h"
#include "Primitives/Mesh.h"
#include <Materials/Material.h>
#include <NoRubberSheet/noRubberSheetMesh.h>
#include <Utils/TimeUtils.h>
#include <shaders_common.h>

using namespace quasar;

NoRubberSheetMesh::NoRubberSheetMesh(const MeshSizeCreateParams& params)
    : closestZBuffer({
        .target = GL_ARRAY_BUFFER,
        .dataSize = sizeof(float),
    }),
    zBuffer({
        .target = GL_ARRAY_BUFFER,
        .dataSize = sizeof(float),
    }),
    Mesh(params)
{

    setZBuffers(params.maxVertices);

    // std::vector<std::string> defines = {
    //     "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::OPAQUE)),
    //     "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::MASKED)),
    //     "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::TRANSPARENT))
    // };
    // noRubberSheetShader = Shader({
    //     .vertexCodeData = SHADER_COMMON_NORUBBERSHEETMESH_VERT,
    //     .vertexCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_VERT_len,
    //     .fragmentCodeData = SHADER_COMMON_NORUBBERSHEETMESH_FRAG,
    //     .fragmentCodeSize = SHADER_COMMON_NORUBBERSHEETMESH_FRAG_len,
    //     .defines = defines,
    // });
}

void NoRubberSheetMesh::setZBuffers(size_t verticesSize) {

    if (verticesSize == 0) {
        return;
    }

    zBuffer.bind();
    zBuffer.resize(verticesSize);
    zBuffer.unbind();

    closestZBuffer.bind();
    closestZBuffer.resize(verticesSize);
    closestZBuffer.unbind();

    zBuffer.bind();
    zBuffer.resize(verticesSize);
    zBuffer.unbind();
}

RenderStats NoRubberSheetMesh::draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;

    // return Mesh::draw(primitiveType, camera, model, frustumCull, overrideMaterial);

    // If the camera is a VR camera, check if the AABB is visible in both frustums
    if (camera.isVR()) {
        auto vrcamera = static_cast<const VRCamera*>(&camera);
        auto& frustumLeft = vrcamera->left.getFrustum();
        auto& frustumRight = vrcamera->right.getFrustum();
        if (frustumCull && !frustumLeft.aabbIsVisible(aabb, model) && !frustumRight.aabbIsVisible(aabb, model)) {
            return stats;
        }
    }
    else {
        auto monocamera = static_cast<const PerspectiveCamera*>(&camera);
        auto& frustum = monocamera->getFrustum();
        if (frustumCull && !frustum.aabbIsVisible(aabb, model)) {
            return stats;
        }
    }

    if (overrideMaterial != nullptr) {
        spdlog::info("NoRubberSheetMesh::draw: Using override material");
    }else {
        spdlog::info("NoRubberSheetMesh::draw: Using default material");
    }

    auto materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();
    
    // bind the z buffer
    glBindVertexArray(vertexArrayBuffer);
    zBuffer.bind();
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void *)0);

    glBindVertexArray(0);

    // bind the closest z buffer
    glBindVertexArray(vertexArrayBuffer);
    closestZBuffer.bind();
    glEnableVertexAttribArray(7);
    glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void *)0);
    glBindVertexArray(0);
    // Set camera params
    setMaterialCameraParams(camera, materialToUse);

    // Set model and normal matrix
    // materialToUse->getShader()->setUint("vertStride", vertexStride);
    materialToUse->getShader()->setMat4("model", model);
    materialToUse->getShader()->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

    stats = Mesh::draw(primitiveType);

    materialToUse->unbind();

    return stats;
}