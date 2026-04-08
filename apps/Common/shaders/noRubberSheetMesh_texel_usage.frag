// Like noRubberSheetMesh.frag; additionally marks texels in a parallel RGBA8 image (same resolution as baseColorMap) blue for each shading sample that used that map.
#include "constants.glsl"
#include "camera.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec3 FragNormal;
layout(location = 3) out uvec4 FragIDs;

layout(rgba8, binding = 0) uniform writeonly image2D baseColorTexelUsageOut;

in VertexData {
    flat uint DrawID;
    vec2 TexCoord;
    vec3 PositionView;
    vec3 PositionWorld;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 PositionLightSpace;
    float ClosestZ;
    float InterpolatedZ;
} fsIn;

uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap;

    sampler2D baseColorMap;
} material;

void main() {
    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoord) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColor * material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold) {
        discard;
    }

    float dist_thresh = 0.02 * abs(fsIn.ClosestZ);
    float slope_thresh = min(fwidth(fsIn.InterpolatedZ), 0.15);
    float threshold = dist_thresh + slope_thresh + 0.08;
    if (abs(fsIn.InterpolatedZ - fsIn.ClosestZ) >= threshold) {
        discard;
    }

    if (material.hasBaseColorMap) {
        ivec2 dims = textureSize(material.baseColorMap, 0);
        vec2 uv = fract(fsIn.TexCoord);
        ivec2 tc = clamp(ivec2(floor(uv * vec2(dims))), ivec2(0), dims - ivec2(1));
        imageStore(baseColorTexelUsageOut, tc, vec4(0.0, 0.0, 1.0, 1.0));
    }

    FragColor = vec4(baseColor.rgb, alpha);
    FragAlpha = alpha;
    FragNormal = normalize(fsIn.Normal);
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, material.alphaMode);
    FragIDs.z = floatBitsToUint((-fsIn.PositionView.z - camera.near) / (camera.far - camera.near));
}
