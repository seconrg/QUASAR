// basically the same as the material_unlit.frag, but with the addition of the closest z value and the interpolated z value
#include "constants.glsl"
#include "camera.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec3 FragNormal;
layout(location = 3) out uvec4 FragIDs;

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

// Material
uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap; // use diffuse map

    // Material textures
    sampler2D baseColorMap; // 0
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
    if (alpha < material.maskThreshold)
        discard;
     
    // float threshold = fwidth(fsIn.InterpolatedZ - fsIn.ClosestZ) * 2.0;
    // 2. Distance tolerance (e.g., 2% of the object's distance)
    float dist_thresh = 0.02 * abs(fsIn.ClosestZ);

    // 3. Slope tolerance (allow natural angles, but cap it so rubber sheets fail)
    float slope_thresh = min(fwidth(fsIn.InterpolatedZ), 0.15);
    float threshold = dist_thresh + slope_thresh + 0.08;
    // Check whether interpolated z value is close to the closest z value
    if (abs(fsIn.InterpolatedZ - fsIn.ClosestZ) >= threshold) {
        // discard the fragment if it is too far from the closest z value
        // otherwise, we still output the fragment
        discard;
    }
    FragColor = vec4(baseColor.rgb, alpha);
    FragAlpha = alpha;
    FragNormal = normalize(fsIn.Normal);
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, material.alphaMode);
    FragIDs.z = floatBitsToUint((-fsIn.PositionView.z - camera.near) / (camera.far - camera.near));
}