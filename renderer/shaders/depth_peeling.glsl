#ifdef DO_DEPTH_PEELING
uniform bool peelDepth;
uniform usampler2D prevIDMap;

uniform float E;
uniform float edpDelta;
uniform bool edpViewOffsetFootprint;
// Camera-position uncertainty/error in the remote/source view space.
// Prediction mode sends per-axis uncertainty; ground-truth mode sends signed
// remote-to-recorded pose error. Positive z is expanded to four x/y corner
// samples below; negative z is covered by the scalar DynamicE/LCOC fallback.
uniform vec3 edpViewOffsetUncertaintyM;
uniform int layerIndex;

// Adapted from https://github.com/cgskku/pvhv/blob/main/shaders/edp.frag
#define DP_EPSILON 0.0005
#define EDP_SAMPLES 16

bool cullUmbraWithE(float fragmentDepth, float zf, int height, float eRadius) {
    float d = fragmentDepth; // fragment depth
    float df = mix(camera.near, camera.far, zf); // blocker depth
    float s  = tan(camera.fovy * 0.5) * 2.0 * df / height; // pixel geometry size
    if (eRadius < s) return true; // no more peeling, because the pixel geometry size > lens size
    float x  = df * s / (eRadius - s);
    return d < df + x;
}

bool cullUmbra(float fragmentDepth, float zf, int height) {
    return cullUmbraWithE(fragmentDepth, zf, height, E);
}

float LCOCWithE(float d, float df, int height, float eRadius) {
    float K = float(height)*0.5 / df / tan(camera.fovy*0.5); // screen-space LCOC scale
    return K * eRadius * abs(df-d) / d; // relative radius of COC against df (blocker depth)
}

float LCOC(float d, float df, int height) {
    return LCOCWithE(d, df, height, E);
}

bool projectViewOffsetSampleToPixel(vec3 sourceViewPos, vec3 viewOffset, vec2 viewportSize, out vec2 pixel) {
    vec3 sampledViewPos = sourceViewPos - viewOffset;
    vec4 clip = camera.projection * vec4(sampledViewPos, 1.0);
    if (clip.w <= 1e-5)
        return false;

    vec2 ndc = clip.xy / clip.w;
    if (any(isnan(ndc)) || any(isinf(ndc)))
        return false;

    pixel = (ndc + vec2(1.0)) * 0.5 * viewportSize;
    return true;
}

vec3 sourceViewPosFromPixelAndDepth(ivec2 pixelCoords, float depth, vec2 viewportSize) {
    vec2 ndc = ((vec2(pixelCoords) + vec2(0.5)) / viewportSize) * 2.0 - vec2(1.0);
    return vec3(
        ndc.x * depth / camera.projection[0][0],
        ndc.y * depth / camera.projection[1][1],
        -depth);
}

float scalarEForViewOffset(vec3 blockerViewPos) {
    // if (!edpViewOffsetFootprint || edpViewOffsetUncertaintyM.z >= 0.0)
    //     return E;

    // Moving forward is handled by resizing scalar DynamicE per blocker depth.
    // The positive z side still uses the explicit view-offset footprint sampling.
    // If the forward uncertainty would cross past the blocker, this projection
    // model is no longer valid, so keep the dynamic E computed on the CPU.
    if (edpViewOffsetUncertaintyM.z < blockerViewPos.z)
        return E;

    float z2 = abs(blockerViewPos.z);
    float z1 = abs(edpViewOffsetUncertaintyM.z);
    float denominator = max(abs(blockerViewPos.z - edpViewOffsetUncertaintyM.z), 1e-5);
    float maxXYUncertainty = max((abs(edpViewOffsetUncertaintyM.x) + abs(blockerViewPos.x)),
                                  (abs(edpViewOffsetUncertaintyM.y) + abs(blockerViewPos.y)));
    float maxHorizontalShift = max(abs(edpViewOffsetUncertaintyM.x), abs(edpViewOffsetUncertaintyM.y));
    // float frontMotionE =
    //     abs(edpViewOffsetUncertaintyM.x) + z1
    //     * (abs(edpViewOffsetUncertaintyM.x) + abs(blockerViewPos.x))
    //     / denominator;
    float frontMotionE =
        maxHorizontalShift + z1 * maxXYUncertainty / denominator;

    if (isnan(frontMotionE) || isinf(frontMotionE) || frontMotionE <= 0.0)
        return E;

    return frontMotionE;
}

bool prevLayerSampleRevealsHidden(
    vec2 sampleCoord,
    int width,
    int height,
    uint qItem,
    float blockerDepthNormalized)
{
    if (sampleCoord.x < 0.0 || sampleCoord.x >= width || sampleCoord.y < 0.0 || sampleCoord.y >= height)
        return false;

    ivec2 samplePixel = clamp(
        ivec2(round(sampleCoord)),
        ivec2(0),
        ivec2(width - 1, height - 1));
    uvec4 w = texelFetch(prevIDMap, samplePixel, 0);

    float sampleDepthNormalized = uintBitsToFloat(w.z);
    if (sampleDepthNormalized == 0)
        return true;
    if (sampleDepthNormalized >= MAX_DEPTH)
        return false;

    int prevAlphaMode = int(w.w);
    if (prevAlphaMode != ALPHA_OPAQUE)
        return true;
    if (qItem != w.w)
        return true;
    if (sampleDepthNormalized >= blockerDepthNormalized + edpDelta)
        return true;
    if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta)
        return true;

    return false;
}

bool sampleFootprintRevealsHidden(
    vec2 fragCoord,
    int width,
    int height,
    uint qItem,
    float blockerDepthNormalized,
    float radius)
{
    if (radius <= 1e-4)
        return false;

    // Match the original PVHV sampling from 10975312: use one circular LCOC
    // footprint centered at the current pixel instead of a directional ellipse.
    bool footprintInside =
        fragCoord.x - radius >= 0.0
        && fragCoord.x + radius < width
        && fragCoord.y - radius >= 0.0
        && fragCoord.y + radius < height;

    bool sampleVisible = false;
    for (int i = 0; i < EDP_SAMPLES; i++) {
        float angle = float(i) * 2.0 * PI / EDP_SAMPLES;
        vec2 offset = vec2(radius * cos(angle), radius * sin(angle));
        vec2 sampleCoord = fragCoord + offset;

        if (prevLayerSampleRevealsHidden(sampleCoord, width, height, qItem, blockerDepthNormalized)) {
            if (footprintInside) return true;
            sampleVisible = true;
        }
    }

    return footprintInside && sampleVisible;
}

bool inViewOffsetPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q, int width, int height) {
    float blockerDepthNormalized = uintBitsToFloat(q.z);
    float blockerDepth = mix(camera.near, camera.far, blockerDepthNormalized);
    vec2 viewportSize = vec2(width, height);
    vec2 fragCoord = vec2(pixelCoords);
    vec3 blockerViewPos = sourceViewPosFromPixelAndDepth(pixelCoords, blockerDepth, viewportSize);
    uint qItem = q.w;

    if (edpViewOffsetUncertaintyM.z <= 0.0)
        return false;

    vec2 xyExtent = abs(edpViewOffsetUncertaintyM.xy);
    float zOffset = edpViewOffsetUncertaintyM.z;
    if (max(max(xyExtent.x, xyExtent.y), zOffset) <= 1e-5)
        return false;

    for (int i = 0; i < 4; i++) {
        vec2 xySign = vec2((i & 1) == 0 ? 1.0 : -1.0, (i & 2) == 0 ? 1.0 : -1.0);
        vec3 viewOffset = vec3(xyExtent * xySign, zOffset);

        vec2 blockerSamplePixel;
        vec2 fragSamplePixel;
        if (!projectViewOffsetSampleToPixel(blockerViewPos, viewOffset, viewportSize, blockerSamplePixel))
            continue;
        if (!projectViewOffsetSampleToPixel(fragViewPos, viewOffset, viewportSize, fragSamplePixel))
            continue;

        vec2 screenSpaceParallax = fragSamplePixel - blockerSamplePixel;
        float screenSpaceParallaxRadius = length(screenSpaceParallax);
        if (screenSpaceParallaxRadius <= 1e-4)
            continue;

        if (sampleFootprintRevealsHidden(
                fragCoord,
                width,
                height,
                qItem,
                blockerDepthNormalized,
                screenSpaceParallaxRadius))
        {
            return true;
        }
    }

    return false;
}

bool inScalarPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q, int width, int height) {
    float fragmentDepth = -fragViewPos.z;
    float blockerDepthNormalized = uintBitsToFloat(q.z);
    float df = mix(camera.near, camera.far, blockerDepthNormalized);
    vec3 blockerViewPos = sourceViewPosFromPixelAndDepth(pixelCoords, df, vec2(width, height));
    float effectiveE = scalarEForViewOffset(blockerViewPos);

    if (layerIndex > 2) return cullUmbraWithE(fragmentDepth, blockerDepthNormalized, height, effectiveE);
    
    // effectiveE = 0.25;
    float R = LCOCWithE(fragmentDepth, df, height, effectiveE);

    uint q_item	= q.w;

    return sampleFootprintRevealsHidden(
        vec2(pixelCoords),
        width,
        height,
        q_item,
        blockerDepthNormalized,
        R);
}

bool inPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q) {
    int width = textureSize(prevIDMap, 0).x;
    int height = textureSize(prevIDMap, 0).y;

    // if (edpViewOffsetFootprint) {
    //     if (inViewOffsetPVHV(pixelCoords, fragViewPos, q, width, height))
    //         return true;
    // }

    return inScalarPVHV(pixelCoords, fragViewPos, q, width, height);
}

void applyDepthPeeling(vec3 fragViewPos) {
    if (peelDepth) {
        ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
        uvec4 q = texelFetch(prevIDMap, pixelCoords, 0);

        float currDepth = -fragViewPos.z;
        float prevDepthNormalized = uintBitsToFloat(q.z);
        if (prevDepthNormalized == 0 || prevDepthNormalized >= MAX_DEPTH)
            discard;
        if (currDepth <= mix(camera.near, camera.far, prevDepthNormalized) + DP_EPSILON)
            discard;
#ifdef EDP
        int prevAlphaMode = int(q.w);
        if ((prevAlphaMode == ALPHA_OPAQUE) && !inPVHV(pixelCoords, fragViewPos, q))
            discard;
#endif
    }
}
#endif
