#ifdef DO_DEPTH_PEELING
uniform bool peelDepth;
uniform usampler2D prevIDMap;

uniform float E;
uniform float edpDelta;
uniform vec2 edpDirection;
uniform float edpPerpendicularScale;
uniform int layerIndex;

// Adapted from https://github.com/cgskku/pvhv/blob/main/shaders/edp.frag
#define DP_EPSILON 0.0005
#define EDP_SAMPLES 16

bool cullUmbra(float fragmentDepth, float zf, int height) {
    float d = fragmentDepth; // fragment depth
    float df = mix(camera.near, camera.far, zf); // blocker depth
    float s  = tan(camera.fovy * 0.5) * 2.0 * df / height; // pixel geometry size
    if (E < s) return true; // no more peeling, because the pixel geometry size > lens size
    float x  = df * s / (E - s);
    return d < df + x;
}

float LCOC(float d, float df, int height) {
    float K = float(height)*0.5 / df / tan(camera.fovy*0.5); // screen-space LCOC scale
    return K * E * abs(df-d) / d; // relative radius of COC against df (blocker depth)
}

bool inPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q) {
    int width = textureSize(prevIDMap, 0).x;
    int height = textureSize(prevIDMap, 0).y;

    float fragmentDepth = -fragViewPos.z;

    if (layerIndex > 2) return cullUmbra(fragmentDepth, uintBitsToFloat(q.z), height);

    float blockerDepthNormalized = uintBitsToFloat(q.z);
    float df = mix(camera.near, camera.far, blockerDepthNormalized);
    float R = LCOC(fragmentDepth, df, height);

    uint q_item	= q.w;

    vec2 majorDirection = length(edpDirection) > 1e-5 ? normalize(edpDirection) : vec2(1.0, 0.0);
    vec2 minorDirection = vec2(-majorDirection.y, majorDirection.x);
    float minorScale = clamp(edpPerpendicularScale, 0.0, 1.0);
    vec2 ellipseHalfExtent = abs(majorDirection) * R + abs(minorDirection) * (R * minorScale);

    // If the sampling footprint is fully inside the image, return true as soon as a visible sample is found.
    // If the footprint reaches outside the image, require at least one visible sample inside the image to consider visible.
    vec2 fragCoord = vec2(pixelCoords);
    bool lcocInside = (fragCoord.x - ellipseHalfExtent.x >= 0.0 && fragCoord.x + ellipseHalfExtent.x < width &&
                       fragCoord.y - ellipseHalfExtent.y >= 0.0 && fragCoord.y + ellipseHalfExtent.y < height);

    bool sampleVisible = false;
    for (int i = 0; i < EDP_SAMPLES; i++) {
        float angle = float(i) * 2.0 * PI / EDP_SAMPLES;
        vec2 offset = majorDirection * (R * cos(angle)) + minorDirection * (R * minorScale * sin(angle));
        vec2 sampleCoord = fragCoord + offset;

        // Skip samples that fall outside the image
        if (sampleCoord.x < 0.0 || sampleCoord.x >= width || sampleCoord.y < 0.0 || sampleCoord.y >= height)
            continue;

        ivec2 samplePixel = clamp(
            ivec2(round(sampleCoord)),
            ivec2(0),
            ivec2(width - 1, height - 1));
        uvec4 w = texelFetch(prevIDMap, samplePixel, 0);
		uint w_item = w.w;

        float sampleDepthNormalized = uintBitsToFloat(w.z);
        if (sampleDepthNormalized == 0) {
            if (lcocInside) return true;
            sampleVisible = true; // hole in prev map -> visible at edge
            continue;
        }
        if (sampleDepthNormalized >= MAX_DEPTH) continue;

        // If opaque in previous layer, consider visible
        int prevAlphaMode = int(w.w);
        if (prevAlphaMode != ALPHA_OPAQUE) {
            if (lcocInside) return true;
            sampleVisible = true;
            continue;
        }

        // If drawIDs differ, consider visible
        if (q_item != w_item) {
            if (lcocInside) return true;
            sampleVisible = true;
            continue;
        }

        if (sampleDepthNormalized >= blockerDepthNormalized + edpDelta) {
            if (lcocInside) return true;
            sampleVisible = true;
            continue;
        }
        else if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta) {
            if (lcocInside) return true;
            sampleVisible = true;
            continue;
        }
    }

    // If circle reached outside image, require at least one visible sample inside the image
    if (lcocInside) return sampleVisible;

    return false;
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
