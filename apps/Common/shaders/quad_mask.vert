uniform vec2 uCorners[4];
uniform vec2 uViewport;

vec2 toNdc(vec2 p) {
    return vec2((p.x / uViewport.x) * 2.0 - 1.0,
                (p.y / uViewport.y) * 2.0 - 1.0);
}

void main() {
    // Indices for two triangles: (0,1,3) and (0,3,2)
    int ids[6] = int[6](0, 1, 3, 0, 3, 2);
    vec2 posPx = uCorners[ids[gl_VertexID]];
    gl_Position = vec4(toNdc(posPx), 0.0, 1.0);

    // gl_Position = vec4(toNdc(pos), 0.0, 1.0);

    // use hardcoded corners for debug
    // vec2 pos;
    // if (gl_VertexID == 0) pos = vec2(-0.5, -0.5);
    // else if (gl_VertexID == 1) pos = vec2(0.5, -0.5);
    // else if (gl_VertexID == 2) pos = vec2(0.5, 0.5);
    // else if (gl_VertexID == 3) pos = vec2(-0.5, -0.5);
    // else if (gl_VertexID == 4) pos = vec2(0.5, 0.5);
    // else pos = vec2(-0.5, 0.5);
    // gl_Position = vec4(pos, 0.0, 1.0);
}
