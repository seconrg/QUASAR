#ifndef NO_RUBBER_SHEET_MESH_H
#define NO_RUBBER_SHEET_MESH_H

#include <Primitives/Mesh.h>

namespace quasar {

class NoRubberSheetMesh : public Mesh {
public:
    NoRubberSheetMesh(const MeshSizeCreateParams& params);
    ~NoRubberSheetMesh() = default;

    void setZBuffers(size_t verticesSize);

    RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model, bool frustumCull, const Material* overrideMaterial) override;

    // void setVertexStride(uint vertexStride) {this->vertexStride = vertexStride;};
    // uint getVertexStride() const { return vertexStride; }

    // Buffer for hold the z value in the view space
    Buffer zBuffer;
    // Buffer for hold the closest z value for each vertex
    Buffer closestZBuffer;

// private:
// uint vertexStride;

    // Shader noRubberSheetShader;
};

} // namespace quasar

#endif // NO_RUBBER_SHEET_MESH_H