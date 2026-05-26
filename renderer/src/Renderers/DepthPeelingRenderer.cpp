#include <Cameras/VRCamera.h>
#include <Renderers/DepthPeelingRenderer.h>

#include <algorithm>

using namespace quasar;

namespace {

void setEDPUniforms(
    float eRadius,
    float edpDelta,
    const glm::vec2& eScreenDirection,
    float ePerpendicularScale,
    bool eScreenSpaceFootprint,
    float eScreenMajorRadiusPx,
    float eScreenMinorRadiusPx,
    bool eViewOffsetFootprint,
    const glm::vec3& eViewOffsetUncertaintyM) {
    if (LitMaterial::deferredShader != nullptr) {
        LitMaterial::deferredShader->bind();
        LitMaterial::deferredShader->setFloat("E", eRadius);
        LitMaterial::deferredShader->setFloat("edpDelta", edpDelta);
        LitMaterial::deferredShader->setVec2("edpDirection", eScreenDirection);
        LitMaterial::deferredShader->setFloat("edpPerpendicularScale", ePerpendicularScale);
        LitMaterial::deferredShader->setBool("edpScreenSpaceFootprint", eScreenSpaceFootprint);
        LitMaterial::deferredShader->setFloat("edpScreenMajorRadiusPx", eScreenMajorRadiusPx);
        LitMaterial::deferredShader->setFloat("edpScreenMinorRadiusPx", eScreenMinorRadiusPx);
        LitMaterial::deferredShader->setBool("edpViewOffsetFootprint", eViewOffsetFootprint);
        LitMaterial::deferredShader->setVec3("edpViewOffsetUncertaintyM", eViewOffsetUncertaintyM);
    }
    if (LitMaterial::forwardShader != nullptr) {
        LitMaterial::forwardShader->bind();
        LitMaterial::forwardShader->setFloat("E", eRadius);
        LitMaterial::forwardShader->setFloat("edpDelta", edpDelta);
        LitMaterial::forwardShader->setVec2("edpDirection", eScreenDirection);
        LitMaterial::forwardShader->setFloat("edpPerpendicularScale", ePerpendicularScale);
        LitMaterial::forwardShader->setBool("edpScreenSpaceFootprint", eScreenSpaceFootprint);
        LitMaterial::forwardShader->setFloat("edpScreenMajorRadiusPx", eScreenMajorRadiusPx);
        LitMaterial::forwardShader->setFloat("edpScreenMinorRadiusPx", eScreenMinorRadiusPx);
        LitMaterial::forwardShader->setBool("edpViewOffsetFootprint", eViewOffsetFootprint);
        LitMaterial::forwardShader->setVec3("edpViewOffsetUncertaintyM", eViewOffsetUncertaintyM);
    }
    if (UnlitMaterial::shader != nullptr) {
        UnlitMaterial::shader->bind();
        UnlitMaterial::shader->setFloat("E", eRadius);
        UnlitMaterial::shader->setFloat("edpDelta", edpDelta);
        UnlitMaterial::shader->setVec2("edpDirection", eScreenDirection);
        UnlitMaterial::shader->setFloat("edpPerpendicularScale", ePerpendicularScale);
        UnlitMaterial::shader->setBool("edpScreenSpaceFootprint", eScreenSpaceFootprint);
        UnlitMaterial::shader->setFloat("edpScreenMajorRadiusPx", eScreenMajorRadiusPx);
        UnlitMaterial::shader->setFloat("edpScreenMinorRadiusPx", eScreenMinorRadiusPx);
        UnlitMaterial::shader->setBool("edpViewOffsetFootprint", eViewOffsetFootprint);
        UnlitMaterial::shader->setVec3("edpViewOffsetUncertaintyM", eViewOffsetUncertaintyM);
    }
}

} // namespace

DepthPeelingRenderer::DepthPeelingRenderer(const Config& config, uint maxLayers, bool edp)
    : maxLayers(maxLayers)
    , edp(edp)
    , DeferredRenderer(config)
    , compositeLayersShader({
        .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
        .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_COMPOSITE_LAYERS_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_COMPOSITE_LAYERS_FRAG_len,
        .defines = {
            "#define MAX_LAYERS " + std::to_string(maxLayers)
        }
    })
{
    sortTransparent = false;

    // Enable depth peeling in shaders
    LitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    UnlitMaterial::extraShaderDefines.push_back("#define DO_DEPTH_PEELING");
    if (edp) {
        LitMaterial::extraShaderDefines.push_back("#define EDP");
        UnlitMaterial::extraShaderDefines.push_back("#define EDP");
    }

    RenderTargetCreateParams params {
        .width = config.width,
        .height = config.height,
        .internalFormat = outputRT.colorTexture.internalFormat,
        .format = outputRT.colorTexture.format,
        .type = outputRT.colorTexture.type,
        .wrapS = outputRT.colorTexture.wrapS,
        .wrapT = outputRT.colorTexture.wrapT,
        .minFilter = outputRT.colorTexture.minFilter,
        .magFilter = outputRT.colorTexture.magFilter,
        .multiSampled = outputRT.colorTexture.multiSampled,
    };
    peelingLayers.reserve(maxLayers);
    for (int i = 0; i < maxLayers; i++) {
        peelingLayers.emplace_back(params);
    }
}

void DepthPeelingRenderer::setEAnisotropy(const glm::vec2& screenDirection, float perpendicularScale) {
    eScreenDirection = glm::length(screenDirection) > 1e-5f
        ? glm::normalize(screenDirection)
        : glm::vec2(1.0f, 0.0f);
    ePerpendicularScale = glm::clamp(perpendicularScale, 0.0f, 1.0f);
}

void DepthPeelingRenderer::clearEAnisotropy() {
    eScreenDirection = glm::vec2(1.0f, 0.0f);
    ePerpendicularScale = 1.0f;
}

void DepthPeelingRenderer::setEScreenSpaceFootprint(float majorRadiusPx, float minorRadiusPx) {
    eScreenSpaceFootprint = true;
    eScreenMajorRadiusPx = std::max(0.0f, majorRadiusPx);
    eScreenMinorRadiusPx = std::max(0.0f, minorRadiusPx);
}

void DepthPeelingRenderer::clearEScreenSpaceFootprint() {
    eScreenSpaceFootprint = false;
    eScreenMajorRadiusPx = 0.0f;
    eScreenMinorRadiusPx = 0.0f;
}

void DepthPeelingRenderer::setEViewOffsetFootprint(const glm::vec3& viewOffsetUncertaintyM) {
    eViewOffsetFootprint = true;
    eViewOffsetUncertaintyM = viewOffsetUncertaintyM;
}

void DepthPeelingRenderer::clearEViewOffsetFootprint() {
    eViewOffsetFootprint = false;
    eViewOffsetUncertaintyM = glm::vec3(0.0f);
}

void DepthPeelingRenderer::resize(uint width, uint height) {
    DeferredRenderer::resize(width, height);
    for (auto layer : peelingLayers) {
        layer.resize(width, height);
    }
}

void DepthPeelingRenderer::setScreenShaderUniforms(const Shader& screenShader) {
    // Set texture uniforms
    screenShader.bind();
    screenShader.setTexture("screenColor", outputRT.colorTexture, 0);
    screenShader.setTexture("screenDepth", peelingLayers[0].depthStencilTexture, 1);
    screenShader.setTexture("screenNormals", gBuffer.normalsTexture, 2);
    screenShader.setTexture("screenPositions", gBuffer.positionTexture, 3);
    screenShader.setTexture("idTexture", gBuffer.idTexture, 4);
}

RenderStats DepthPeelingRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    for (int layer = 0; layer < maxLayers; layer++) {
        beginRendering();
        if (clearMask != 0) {
            gBuffer.clear(clearMask);
        }

        // Disable blending
        pipeline.blendState.blendEnabled = false; pipeline.apply();

        const Texture* prevIDMap = (layer >= 1) ? &peelingLayers[layer-1].idTexture : nullptr;

        // Set layer index in shaders
        if (LitMaterial::deferredShader != nullptr) {
            LitMaterial::deferredShader->bind();
            LitMaterial::deferredShader->setInt("layerIndex", layer);
        }
        if (LitMaterial::forwardShader != nullptr) {
            LitMaterial::forwardShader->bind();
            LitMaterial::forwardShader->setInt("layerIndex", layer);
        }
        if (UnlitMaterial::shader != nullptr) {
            UnlitMaterial::shader->bind();
            UnlitMaterial::shader->setInt("layerIndex", layer);
        }

        // Render scene
        for (auto* child : scene.children) {
            stats += drawNodeImmediate(scene, camera, child, glm::mat4(1.0f), true, nullptr, prevIDMap);
        }

        // Re-enable blending
        pipeline.blendState.blendEnabled = true; pipeline.apply();

        endRendering();

        // Draw lighting pass
        stats += lightingPass(scene, camera);

        // Draw skybox (only in last layer)
        if (layer == maxLayers - 1) {
            // stats += drawSkyBox(scene, camera);
        }

        copyToFrameRT(peelingLayers[layer]);
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;
    if (camera.isVR()) {
        auto* vrCamera = static_cast<const VRCamera*>(&camera);

        pipeline.rasterState.scissorTestEnabled = true;

        // Left eye
        gBuffer.setScissor({ 0, 0, width / 2, height });
        gBuffer.setViewport({ 0, 0, width / 2, height });
        outputRT.setScissor({ 0, 0, width / 2, height });
        outputRT.setViewport({ 0, 0, width / 2, height });
        stats += drawObjects(scene, vrCamera->left, clearMask);

        // Right eye
        gBuffer.setScissor({ width / 2, 0, width / 2, height });
        gBuffer.setViewport({ width / 2, 0, width / 2, height });
        outputRT.setScissor({ width / 2, 0, width / 2, height });
        outputRT.setViewport({ width / 2, 0, width / 2, height });
        stats += drawObjects(scene, vrCamera->right, clearMask);
    }
    else {
        pipeline.apply();

        if (edp) {
            setEDPUniforms(
                getEffectiveE(),
                edpDelta,
                eScreenDirection,
                ePerpendicularScale,
                eScreenSpaceFootprint,
                eScreenMajorRadiusPx,
                eScreenMinorRadiusPx,
                eViewOffsetFootprint,
                eViewOffsetUncertaintyM);
        }

        RenderStats stats;

        // Update shadows
        updateDirLightShadow(scene, camera);
        updatePointLightShadows(scene, camera);

        // Draw all objects in the scene
        stats += drawScene(scene, camera, clearMask);

        // Draw lights for debugging
        stats += drawLights(scene, camera);

        // Dont draw skybox here, it's drawn in drawScene

        // Composite layers
        stats += compositeLayers();
    }

    return stats;
}

RenderStats DepthPeelingRenderer::drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    if (edp) {
        setEDPUniforms(
            getEffectiveE(),
            edpDelta,
            eScreenDirection,
            ePerpendicularScale,
            eScreenSpaceFootprint,
            eScreenMajorRadiusPx,
            eScreenMinorRadiusPx,
            eViewOffsetFootprint,
            eViewOffsetUncertaintyM);
    }

    RenderStats stats;

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Dont draw skybox here, it's drawn in drawScene

    // Composite layers
    stats += compositeLayers();

    return stats;
}

RenderStats DepthPeelingRenderer::compositeLayers() {
    RenderStats stats;

    compositeLayersShader.bind();
    for (int i = 0; i < maxLayers; i++) {
        compositeLayersShader.setTexture(
            "peelingLayersColor[" + std::to_string(i) + "]", peelingLayers[i].colorTexture, i);
        compositeLayersShader.setTexture(
            "peelingLayersAlpha[" + std::to_string(i) + "]", peelingLayers[i].alphaTexture, i + maxLayers);
    }

    outputRT.bind();
    outputRT.clear(GL_COLOR_BUFFER_BIT);
    stats += outputFsQuad->draw();
    outputRT.unbind();

    return stats;
}
