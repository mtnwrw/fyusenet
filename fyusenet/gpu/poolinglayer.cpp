//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pooling Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/miscdefs.h"
#include "poolinglayer.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
PoolingLayer::PoolingLayer(const PoolLayerBuilder & builder, int layerNumber):GPULayerBase((const GPULayerBuilder &)builder, layerNumber) {
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"This layer does not support residual input");
    poolSize_[0] = builder.poolsize_[0];
    poolSize_[1] = builder.poolsize_[1];
    downsample_[0] = builder.downsample_[0];
    downsample_[1] = builder.downsample_[1];
    viewport_[0] = (width_ / downsample_[0]) + 2 * outputPadding_;
    viewport_[1] = (height_ / downsample_[1]) + 2 * outputPadding_;
    maxRenderTargets_ = GLInfo::getMaximumDrawBuffers();
    if (maxRenderTargets_ > FBO::MAX_DRAWBUFFERS) maxRenderTargets_ = FBO::MAX_DRAWBUFFERS;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    currentShader_ = nullptr;
}


/**
 * @copydoc LayerBase::cleanup
 */
void PoolingLayer::cleanup() {
    currentShader_ = nullptr;
    FNET_DEL_AND_CLEAR(vertexBuffer_);
    FNET_DEL_AND_CLEAR(indexBuffer_);
    FNET_DEL_AND_CLEAR(vertexArray_);
    // reset shaders here because the GL context is bound here (in case no cache is used)
    for (auto & shader : shaders_) shader.reset();
    GPULayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void PoolingLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupVBO(vertexArray_);
    setupIBO(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_=true;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> PoolingLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    if (inputChannels_ < PIXEL_PACKING) {
        // for input textures, we support textures with less than 4 channels (might be from upload)
        auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
        result.emplace_back(channel++, 0, width_+2*inputPadding_, height_+2*inputPadding_,
                            format.first, format.second, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_SOURCE, inputChannels_);
    } else {
        int rem = inputChannels_;
        while (rem > 0) {
            result.emplace_back(channel++, 0,
                                width_ + 2 * inputPadding_, height_ + 2 * inputPadding_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE, std::min(rem, PIXEL_PACKING));
            rem -= PIXEL_PACKING;
        }
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> PoolingLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    int rem = outputChannels_;
    while (rem > 0) {
        result.emplace_back(channel++, 0,
                            viewport_[0], viewport_[1],
                            TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_DEST, std::min(rem, PIXEL_PACKING));
        rem -= PIXEL_PACKING;
    }
    return result;
}


/**
 * @copydoc LayerBase::forward
 */
void PoolingLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
#ifdef DEBUG
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]", err, __FILE__, __LINE__, getName().c_str());
#endif
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    int totaltex = (inputChannels_ / PIXEL_PACKING) + (((inputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    int outputpasses = (totaltex / maxRenderTargets_) + (((totaltex % maxRenderTargets_) > 0) ? 1 : 0);
    int texoffset = 0;
    vertexArray_->bind();
#ifdef DEBUG
    if ((int)framebuffers_.size() < outputpasses) {
        THROW_EXCEPTION_ARGS(FynException, "Found %d framebuffers but have %d output passes in layer %s",
                             framebuffers_.size(), outputpasses, getName().c_str());
    }
#endif
    currentShader_ = nullptr;
    beforeRender();
    for (int opass = 0; opass < outputpasses; opass++) {
        framebuffers_.at(opass)->bind();
        framebuffers_.at(opass)->setWriteMask();
        if (totaltex >= maxRenderTargets_) {
            glClear(GL_COLOR_BUFFER_BIT);
            renderChannelBatch(opass, maxRenderTargets_, texoffset);
            texoffset += maxRenderTargets_;
            totaltex -= maxRenderTargets_;
        } else if (totaltex > 0) {
            glClear(GL_COLOR_BUFFER_BIT);
            renderChannelBatch(opass, totaltex, texoffset);
            texoffset += totaltex;
            totaltex = 0;
        }
        framebuffers_.at(opass)->unbind();
    }
    afterRender();
    if (currentShader_) currentShader_->unbind();
    vertexArray_->unbind();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them. It utilizes a private interface consisting of
 * #compileShader and #initShader in order for the derived classes to determine which type of pooling
 * has to be done.
 *
 * The following (shader) preprocessor macros are defined:
 *   - \c NUM_LANES
 *   - \c POOL_SIZE (currently this code assumes that the pool size is isotropic)
 *   - \c DOWNSAMPLE (currently this code assumes that downsampling is isotropic)
 *
 * @see compileShader, initShader
 */
void PoolingLayer::setupShaders() {
#if defined(WIN32) || defined(WIN64)
        using ssize_t = int64_t;
#endif
    char preproc[1024] = {0}, extra[128];
    for (int i = 1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n", i);
        ssize_t mc = (ssize_t)preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
        assert(mc > 0);
        // NOTE (mw) this assumes that the poolsize is isotropic !
        snprintf(extra, sizeof(extra),"#define POOL_SIZE %d\n", poolSize_[0]);
        strncat(preproc, extra, mc);
        mc -= (ssize_t)strlen(extra);
        assert(mc > 0);
        // NOTE (mw) this assumes that the downsampling is isotropic !
        snprintf(extra, sizeof(extra), "#define DOWNSAMPLE %d\n", downsample_[0]);
        strncat(preproc, extra, mc);
        mc -= (ssize_t)strlen(extra);
        assert(mc > 0);
        shaders_[i - 1] = compileShader(preproc);
        shaderStates_[i - 1] = initShader(shaders_[i - 1], i);
    }
}


/**
 * @brief Setup vertices / geometry for the proxy polygon
 *
 * @param vao Pointer to vertex array object that will reference the VBO
 *
 * @pre Supplied \p vao is already bound
 *
 * This function creates a proxy polygon that is used to drive the fragment shader that performs
 * the actual computation.
 *
 * @see setupIBO
 */
void PoolingLayer::setupVBO(VAO *vao) {
    const int vertsize = 4;
    float tmp[vertsize * 4];
    float posleft  = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[0]);
    float posright =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[0]);
    float postop   = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[1]);
    float posbottom = 1.0f - ((float)(2*outputPadding_) / (float)viewport_[1]);
    float tleft = ((float) inputPadding_ / (float) (width_ + 2 * inputPadding_));
    float ttop = ((float) inputPadding_ / (float) (height_ + 2 * inputPadding_));
    float thspan = ((float) width_) / (float) (width_ + 2 * inputPadding_);
    float tvspan = ((float) height_) / (float) (height_ + 2 * inputPadding_);
    if (downsample_[0] > 1) {
        tleft -= 0.5f*(float)(downsample_[0]-1) / (float)(width_ + 2*inputPadding_);
    }
    if (downsample_[1] > 1) {
        ttop -= 0.5f*(float)(downsample_[1]-1) / (float)(height_ + 2*inputPadding_);
    }
    tmp[0 * vertsize + 0] = posleft;        // position
    tmp[0 * vertsize + 1] = postop;
    tmp[0 * vertsize + 2] = tleft;          // texture
    tmp[0 * vertsize + 3] = ttop;
    tmp[1 * vertsize + 0] = posleft;        // position
    tmp[1 * vertsize + 1] = posbottom;
    tmp[1 * vertsize + 2] = tleft;          // texture
    tmp[1 * vertsize + 3] = ttop + tvspan;
    tmp[2 * vertsize + 0] = posright;       // position
    tmp[2 * vertsize + 1] = posbottom;
    tmp[2 * vertsize + 2] = tleft + thspan; // texture
    tmp[2 * vertsize + 3] = ttop + tvspan;
    tmp[3 * vertsize + 0] = posright;       // position
    tmp[3 * vertsize + 1] = postop;
    tmp[3 * vertsize + 2] = tleft + thspan; // texture
    tmp[3 * vertsize + 3] = ttop;
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(tmp, (int)(vertsize * 4 * sizeof(float)), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0, vertsize, GL_FLOAT, GL_FALSE, 0, 0);
}


/**
 * @brief Setup index buffer object that defines polygon connectivity
 *
 * @param vao Pointer to vertex array object that keeps track of the buffers for the polygon
 *
 * This function initializes an index buffer object with the connectivity for a simple quadrilateral.
 *
 * @see setupVBO
 */
void PoolingLayer::setupIBO(VAO *vao) {
    // TODO (mw) this is a very often occuring pattern, create a global instance and reuse it
    // instead, reduces the number of handles floating around
    GLshort indices[6]={0,1,2,0,2,3};
    indexBuffer_ = new IBO(context_);
    indexBuffer_->setBufferData(indices, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void PoolingLayer::setupFBOs() {
    int totaltex = (outputChannels_ / PIXEL_PACKING) + (((outputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    if ((int)outputTextures_.size() < totaltex) {
        THROW_EXCEPTION_ARGS(FynException, "Mismatch in output textures (%d) and textures required by render passes (%d)",
                             outputTextures_.size(), totaltex);
    }
    int outputpasses = (totaltex / maxRenderTargets_) + (((totaltex % maxRenderTargets_) > 0) ? 1 : 0);
    int texoffset = 0;
    for (int p = 0; p < outputpasses; p++) {
        int pack = 1;
        FBO *fbo = new FBO(context_, viewport_[0], viewport_[1], outputTextures_.at(texoffset++));
        while ((pack < maxRenderTargets_) && (texoffset < totaltex)) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0 + pack, outputTextures_.at(texoffset++));
            pack++;
        }
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void PoolingLayer::updateFBOs() {
    int totaltex = (outputChannels_ / PIXEL_PACKING) + (((outputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    if ((int)outputTextures_.size() < totaltex) {
        THROW_EXCEPTION_ARGS(FynException, "Mismatch in output textures (%d) and textures required by render passes (%d)",
                             outputTextures_.size(), totaltex);
    }
    int outputpasses = (totaltex / maxRenderTargets_) + (((totaltex % maxRenderTargets_) > 0) ? 1 : 0);
    int texoffset = 0;
    for (int p = 0; p < outputpasses; p++) {
        int pack = 0;
        FBO *fbo = framebuffers_.at(p);
        fbo->bind();
        while ((pack < maxRenderTargets_) && (texoffset < totaltex)) {
            fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0 + pack, outputTextures_.at(texoffset++));
            pack++;
        }
        fbo->unbind();
    }
    outputChanged_ = false;
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
