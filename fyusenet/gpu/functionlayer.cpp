//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Function Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/miscdefs.h"
#include "functionlayer.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
FunctionLayer::FunctionLayer(const GPULayerBuilder &builder, int layerNumber) : GPULayerBase(builder, layerNumber) {
    maxRenderTargets_ = GLInfo::getMaximumRecommendedDrawBuffers();
    if (builder.maxSequenceLen_ > 0) {
        // layer is to be used on sequence data
        isSequence_ = true;
        width_ = (builder.in() + PIXEL_PACKING-1) / PIXEL_PACKING;
        height_ = builder.maxSequenceLen_;
        inputPadding_ = 0;
        outputPadding_ = 0;
        viewport_[0] = width_;
        viewport_[1] = height_;
        maxRenderTargets_ = 1;
    }
    if (flags_ &  LayerFlags::POST_BATCHNORM) {
        THROW_EXCEPTION_ARGS(FynException, "This layer type does not support batchnorm (yet)");
    }
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void FunctionLayer::cleanup() {
    FNET_DEL_AND_CLEAR(vertexBuffer_);
    FNET_DEL_AND_CLEAR(indexBuffer_);
    FNET_DEL_AND_CLEAR(vertexArray_);
    GPULayerBase::cleanup();
}

/**
 * @brief Setup layer by allocating and initializing required GL resources
 *
 * @pre OpenGL context that is to be used for rendering must be current to the calling thread
 *
 * This function performs the required setup for layer operation. It allocates GL resources like
 * FBOs and VBOs, pre-computes the proxy-polygons used for rendering and also compiles all required
 * GL shaders to perform the computations on the proxy polygons.
 */
void FunctionLayer::setup() {
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
std::vector<BufferSpec> FunctionLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    if (isSequence_) {
        result.push_back(BufferSpec(0, 0, width_, height_,
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    } else {
        int channel = 0;
        if (inputChannels_ < PIXEL_PACKING) {
            // for input textures, we support textures with less than 4 channels (might be from upload)
            auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
            result.emplace_back(channel++, 0, width_ + 2 * inputPadding_, height_ + 2 * inputPadding_,
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
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> FunctionLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    if (isSequence_) {
        result.push_back(BufferSpec(0, 0,
                                    width_, height_,
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    } else {
        int channel = 0;
        int rem = outputChannels_;
        while (rem > 0) {
            result.emplace_back(channel++, 0,
                                viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST, std::min(PIXEL_PACKING, rem));
            rem -= PIXEL_PACKING;
        }
    }
    return result;
}


/**
 * @brief Execute layer
 *
 * @param sequence Sequence number (\b must be strictly increasing)
 * @param state State token that contains information for stateful layers and dynamic buffer sizes
 *
 * This function performs the actual computation that maps the input data to the output data
 * for this layer. The supplied \p sequence number must be strictly increasing per inference run
 * and may also be used for debugging purposes, in case errors only manifests themselves after a
 * certain number of computation cycles. It can also be used to keep track of the total number of
 * inference runs. Internally, it is used to make sure that asynchronously transmitted data is
 * up-to-date (on PBO reads for example).
 *
 * This particular implementation performs a multi-pass render sequence, based on the number
 * of input and output channels and calls an implementation specific interface consisting
 * of:
 *   - #beforeRender() which should implement pre-render inits
 *   - #renderChannelBatch() which should implement rendering of all input channels to the selected
 *     output channel(s)
 *   - #afterRender() which should implement post-render cleanups
 *
 * All classes deriving from FunctionLayer must implement the above interface.
 */
void FunctionLayer::forward(uint64_t sequence, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x",err);
#endif
    prepareRender(false, false);
    beforeRender();
    int totaltex = (inputChannels_ / PIXEL_PACKING) + (((inputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    if (isSequence_) {
        glEnable(GL_SCISSOR_TEST);
        totaltex = 1;
        glScissor(0, 0, viewport_[0], state->seqLength);
        glViewport(0, 0, viewport_[0], state->seqLength);
    } else {
        glViewport(0, 0, viewport_[0], viewport_[1]);
    }
    int texoffset = 0;
    vertexArray_->bind();
    for (int opass=0; opass < (int)framebuffers_.size(); opass++) {
        framebuffers_.at(opass)->bind();
        framebuffers_.at(opass)->setWriteMask();
        if (totaltex >= maxRenderTargets_) {
            glClear(GL_COLOR_BUFFER_BIT); // this is to instruct the tile-engine that we don't need the old tile-content
            renderChannelBatch(opass, maxRenderTargets_, texoffset);
            texoffset += maxRenderTargets_;
            totaltex -= maxRenderTargets_;
        } else if (totaltex > 0) {
            glClear(GL_COLOR_BUFFER_BIT); // this is to instruct the tile-engine that we don't need the old tile-content
            renderChannelBatch(opass, totaltex, texoffset);
            texoffset += totaltex;
            totaltex = 0;
        }
        framebuffers_.at(opass)->unbind();
    }
    afterRender();
    vertexArray_->unbind();
    if (isSequence_) glDisable(GL_SCISSOR_TEST);
}




/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Setup %VBO that contains the vertex/texture information of the proxy polygon
 *
 * @param vao Pointer to vertex array object that keeps track of the buffers for the polygon
 *
 * This function initializes a simple quadrilateral which is used as proxy polygon to run the
 * fragment shader on, which performs the actual computation.
 *
 * @see setupIBO, setup
 */
void FunctionLayer::setupVBO(VAO *vao) {
    const int vertsize = 4;
    float tmp[vertsize * 4];
    float posleft  = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[0]);
    float posright =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[0]);
    float postop   = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[1]);
    float posbottom = 1.0f - ((float)(2*outputPadding_) / (float)viewport_[1]);
    float thspan = (float) (width_) / (float) (width_ + 2 * inputPadding_);
    float tvspan = (float) (height_) / (float) (height_ + 2 * inputPadding_);
    float tleft = (float) inputPadding_ / (float) (width_ + 2 * inputPadding_);
    float ttop = (float) inputPadding_ / (float) (height_ + 2 * inputPadding_);
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
void FunctionLayer::setupIBO(VAO *vao) {
    GLshort indices[6] = {0, 1, 2, 0, 2, 3};
    indexBuffer_ = new IBO(context_);
    indexBuffer_->setBufferData(indices, 6 * sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void FunctionLayer::setupFBOs() {
    int totaltex = (inputChannels_ / PIXEL_PACKING) + (((inputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    if ((int)outputTextures_.size() < totaltex) {
        THROW_EXCEPTION_ARGS(FynException, "Mismatch in output textures (%d) and textures required by render passes (%d)", outputTextures_.size(), totaltex);
    }
    int outputpasses = (totaltex / maxRenderTargets_) + (((totaltex % maxRenderTargets_) > 0) ? 1 : 0);
    int texoffset = 0;
    for (int p = 0; p < outputpasses; p++) {
        int pack = 1;
        FBO *fbo = new FBO(context_, viewport_[0], viewport_[1], outputTextures_.at(texoffset++));
        while ((pack < maxRenderTargets_) && (texoffset < totaltex)) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0 + pack, outputTextures_.at(texoffset++), GL_TEXTURE_2D);
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
void FunctionLayer::updateFBOs() {
    int totaltex = (inputChannels_ / PIXEL_PACKING) + (((inputChannels_ % PIXEL_PACKING) > 0) ? 1 : 0);
    if ((int)outputTextures_.size() < totaltex) {
        THROW_EXCEPTION_ARGS(FynException, "Mismatch in output textures (%d) and textures required by render passes (%d)", outputTextures_.size(), totaltex);
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
