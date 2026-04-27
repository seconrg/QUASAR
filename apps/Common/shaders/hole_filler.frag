#include "constants.glsl"
#include "tonemap.glsl"

layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;
uniform sampler2D wideFovFillColor;
uniform sampler2D wideFovFillAlpha;
uniform vec2 wideFovNormalViewCorners[4];

uniform float depthThreshold;
uniform int searchRadius = 3;

uniform bool tonemap = true;
uniform bool useWideFovFill = false;
uniform float exposure = 1.0;

bool isNearlyBlack(vec3 color) {
    return max(max(abs(color.r), abs(color.g)), abs(color.b)) <= 1e-4;
}

vec2 mapNormalViewPixelToWideFovTexCoord() {
    vec2 screenSizePx = vec2(textureSize(screenColor, 0));
    vec2 wideFovSizePx = vec2(textureSize(wideFovFillColor, 0));
    // The stencil trim path uses GL viewport pixel coordinates, so use fragment coordinates
    // here as well instead of the post-process UV convention.
    vec2 normalPixel = gl_FragCoord.xy;
    vec2 basisX = (wideFovNormalViewCorners[1] - wideFovNormalViewCorners[0]) / screenSizePx.x;
    vec2 basisY = (wideFovNormalViewCorners[2] - wideFovNormalViewCorners[0]) / screenSizePx.y;
    vec2 wideFovPixel = wideFovNormalViewCorners[0] + basisX * normalPixel.x + basisY * normalPixel.y;
    return clamp(wideFovPixel / wideFovSizePx, vec2(0.0), vec2(1.0));
}

void main() {
    vec3 color = texture(screenColor, TexCoord).rgb;
    float centerDepth = texture(screenDepth, TexCoord).r;
    bool usedWideFovFill = false;

    if (useWideFovFill && isNearlyBlack(color)) {
        vec2 wideFovTexCoord = mapNormalViewPixelToWideFovTexCoord();
        float wideFovAlpha = texture(wideFovFillAlpha, wideFovTexCoord).r;
        vec3 wideFovColor = texture(wideFovFillColor, wideFovTexCoord).rgb;
        if ((wideFovAlpha > 0.0 || !isNearlyBlack(wideFovColor)) && !isNearlyBlack(wideFovColor)) {
            color = wideFovColor;
            usedWideFovFill = true;
        }
    }

    if (!usedWideFovFill && centerDepth >= MAX_DEPTH) {
        vec2 textureSize = vec2(textureSize(screenColor, 0));

        bool isSkyBox = true;
        for (int i = 1; i <= searchRadius; i++) {
            float topDepth = texture(screenDepth, TexCoord + vec2(0.0, i / textureSize.y)).r;
            float bottomDepth = texture(screenDepth, TexCoord - vec2(0.0, i / textureSize.y)).r;
            float leftDepth = texture(screenDepth, TexCoord - vec2(i / textureSize.x, 0.0)).r;
            float rightDepth = texture(screenDepth, TexCoord + vec2(i / textureSize.x, 0.0)).r;

            bool bothSidesUnder =
                ((abs(topDepth - bottomDepth) <= depthThreshold) &&
                 (abs(topDepth - centerDepth) > depthThreshold) &&
                 (abs(bottomDepth - centerDepth) > depthThreshold)) ||
                ((abs(leftDepth - rightDepth) <= depthThreshold) &&
                 (abs(leftDepth - centerDepth) > depthThreshold) &&
                 (abs(rightDepth - centerDepth) > depthThreshold));
            bothSidesUnder = bothSidesUnder ||
                             ((topDepth < MAX_DEPTH && bottomDepth < MAX_DEPTH) ||
                              (leftDepth < MAX_DEPTH && rightDepth < MAX_DEPTH));
            if (bothSidesUnder) {
                isSkyBox = false;
                break;
            }
        }

        // Fill hole
        if (!isSkyBox) {
            vec3 sumColor = vec3(0.0);
            float sumWeight = 0.0;

            for (int x = -searchRadius; x <= searchRadius; x++) {
                for (int y = -searchRadius; y <= searchRadius; y++) {
                    vec2 texCoord = TexCoord + vec2(x, y) / textureSize;
                    float sampleDepth = texture(screenDepth, texCoord).r;
                    if (sampleDepth < MAX_DEPTH) {
                        float weight = 1.0 / (1.0 + abs(centerDepth - sampleDepth));
                        sumColor += texture(screenColor, texCoord).rgb * weight;
                        sumWeight += weight;
                    }
                }
            }

            if (sumWeight > 0.0) {
                color = sumColor / sumWeight;
            }
        }
    }

    if (tonemap) {
        color = tonemapExponential(color, exposure);
    }

    FragColor = vec4(color, 1.0);
}
