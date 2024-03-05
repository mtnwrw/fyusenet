//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Base Class (generic GPU)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glinfo.h"
#include "../../common/logging.h"
#include "../../common/miscdefs.h"
#include "convlayerbase_vanilla.h"
#include "../convweightarrayKxKxNxM.h"

namespace fyusion::fyusenet::gpu::vanilla {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param builder Builder object that contains parameterization for the layer
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * @pre The constructor must be called with the GL context supplied in \p builder as the active
 *      context
 *
 * This constructor parses basic information from the supplied \p builder and initializes the
 * layer with the parsed data.
 */
ConvLayerBase::ConvLayerBase(const ConvLayerBuilder & builder) : ConvLayerBase(builder, builder.number_) {
}


/**
 * @brief Constructor
 *
 * @param builder convolution-specific layer builder that contains parameterization for the layer
 *
 * @param layerNumber Layer number that defines sequence position in execution
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * @pre The constructor must be called with the GL context supplied in \p builder as the active
 *      context
 *
 * This constructor parses basic information from the supplied \p builder and initializes the
 * layer with the parsed data.
 */
ConvLayerBase::ConvLayerBase(const ConvLayerBuilder & builder,int layerNumber) : gpu::ConvLayerBase(builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    // -------------------------------------------------------------------
    // Determine maximum number of render targets based on GPU capability
    // on the drawing side and capacity on the number of uniforms for a
    // fragment shader...
    // -------------------------------------------------------------------
    maxRenderTargets_ = GLInfo::getMaximumRecommendedDrawBuffers();
    int maxvecs = GLInfo::getMaxUniformVectors(GLInfo::FRAGMENT);
    maxvecs -= (flags_ & LayerFlags::POST_BATCHNORM) ? maxRenderTargets_ : 0;  // very conservative estimate here
    int biasvec = (outputPadding_) ? 1 : 0;
    int maxrt = std::max(1, (maxvecs - VEC_OVERHEAD) / (4*kernel_ + biasvec));
    maxRenderTargets_ = std::min(maxRenderTargets_, maxrt);
    // -------------------------------------------------------------------
    // Initialize default/fallback parameters
    // -------------------------------------------------------------------
    zeroBias_ = new float[maxRenderTargets_ * PIXEL_PACKING + PIXEL_PACKING];
    memset(zeroBias_, 0, (maxRenderTargets_ * PIXEL_PACKING + PIXEL_PACKING) * sizeof(float));
    // -------------------------------------------------------------------
    // Check for GPU types that might require special treatment...
    // -------------------------------------------------------------------
    if (GLInfo::getGPUType() == GLInfo::ARM_MALI) {
        mali_ = true;
        std::string renderer = GLInfo::getRendererString();
        if (!renderer.empty()) {            
            if (strstr(renderer.c_str(),"-T")) preG71_ = true;
            // (code removed) if preG71_ is set, we should better use the Mali specific convolution layer
        }
    }
}


/**
 * @brief Constructor
 *
 * @param builder convolution-specific layer builder that contains parameterization for the layer
 *
 * @param layerNumber Layer number that defines sequence position in execution
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * @pre The constructor must be called with the GL context supplied in \p builder as the active
 *      context
 *
 * This constructor parses basic information from the supplied \p builder and initializes the
 * layer with the parsed data.
 */
ConvLayerBase::ConvLayerBase(const GPULayerBuilder & builder,int layerNumber) : gpu::ConvLayerBase(builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    // -------------------------------------------------------------------
    // Determine maximum number of render targets based on GPU capability
    // on the drawing side and capacity on the number of uniforms for a
    // fragment shader...
    // -------------------------------------------------------------------
    maxRenderTargets_ = GLInfo::getMaximumRecommendedDrawBuffers();
    int maxvecs = GLInfo::getMaxUniformVectors(GLInfo::FRAGMENT);
    maxvecs -= (flags_ & LayerFlags::POST_BATCHNORM) ? maxRenderTargets_ : 0;  // very conservative estimate here
    int biasvec = (outputPadding_) ? 1 : 0;
    int maxrt = std::max(1, (maxvecs - VEC_OVERHEAD) / (4*kernel_ + biasvec));
    maxRenderTargets_ = std::min(maxRenderTargets_, maxrt);
    // -------------------------------------------------------------------
    // Initialize default/fallback parameters
    // -------------------------------------------------------------------
    zeroBias_ = new float[maxRenderTargets_ * PIXEL_PACKING + PIXEL_PACKING];
    memset(zeroBias_, 0, (maxRenderTargets_ * PIXEL_PACKING + PIXEL_PACKING) * sizeof(float));
    // -------------------------------------------------------------------
    // Check for GPU types that might require special treatment...
    // -------------------------------------------------------------------
    if (GLInfo::getGPUType() == GLInfo::ARM_MALI) {
        mali_ = true;
        std::string renderer = GLInfo::getRendererString();
        if (!renderer.empty()) {
            if (strstr(renderer.c_str(),"-T")) preG71_ = true;
            // (code removed) if preG71_ is set, we should better use the Mali specific convolution layer
        }
    }
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
ConvLayerBase::~ConvLayerBase() {
    FNET_DEL_AND_CLEAR(weights_);
    FNET_DEL_AND_CLEAR(zeroBias_);
    if ((vertexBuffer_) || (indexBuffer_) || (vertexArray_)) {
        FNLOGW("Cleanup was not called prior to destruction");
#ifdef DEBUG
        assert(false);
#endif
    }
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void ConvLayerBase::cleanup() {
    FNET_DEL_AND_CLEAR(vertexBuffer_);
    FNET_DEL_AND_CLEAR(indexBuffer_);
    FNET_DEL_AND_CLEAR(vertexArray_);
    FNET_DEL_AND_CLEAR(residualBuffer_);
    gpu::ConvLayerBase::cleanup();
}




/**
 * @brief Perform setup of layer code
 *
 * @pre The GL context that is to be used for running the inference must be current to the calling
 *      thread and loadParameters() has been called prior to this function.
 *
 * @post Layer is marked as valid
 *
 * This function sets up the required shaders, framebuffers and proxy polygon data.
 *
 * @see setupShaders, setupFBOs
 */
void ConvLayerBase::setup() {
#ifdef DEBUG
    glGetError();
#endif
    setupShaders();
    setupFBOs();
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_,kernel_);
    vertexArray_->unbind();
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        THROW_EXCEPTION_ARGS(FynException,"Failed to setup network layer (glerr=0x%x)",err);
    }
#endif
    valid_ = true;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> ConvLayerBase::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    int rem = inputChannels_;
    if (rem < PIXEL_PACKING) {
        // for input textures, we support textures with less than 4 channels (might be from upload)
        auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
        result.emplace_back(channel, 0, width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                            format.first, format.second, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_SOURCE, rem);
    } else {
        while (rem > 0) {
            result.emplace_back(channel++, 0, width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE, std::min(PIXEL_PACKING, rem));
            rem -= PIXEL_PACKING;
        }
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        channel = 0;
        rem = outputChannels_;
        while (rem > 0) {
            result.emplace_back(channel++, 1, residualViewport_[0], residualViewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::RESIDUAL_SOURCE, std::min(PIXEL_PACKING, rem));
            rem -= PIXEL_PACKING;
        }
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> ConvLayerBase::getRequiredOutputBuffers() const {
    int channel = 0;
    std::vector<BufferSpec> result;
    int rem = outputChannels_;
    while (rem > 0) {
        result.emplace_back(channel++, 0, viewport_[0], viewport_[1],
                            TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_DEST, std::min(PIXEL_PACKING, rem));
        rem -= PIXEL_PACKING;
    }
    return result;
}


/**
 * @copydoc gpu::ConvLayerBase::loadParameters
 */
void ConvLayerBase::loadParameters(const ParameterProvider *weights) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    weights_ = new ConvWeightArrayKxKxNxM(kernel_, inputChannels_, outputChannels_, maxRenderTargets_);
    weights->map(getName()+std::string(".bias"), getNumber(), 1).with([&](const std::any & data) {
        weights_->extractBiasData(std::any_cast<const float *>(data));
    });
    weights->map(getName()+std::string(".weights"), getNumber(), 0).with([&](const std::any & data) {
        weights_->extractWeightData(std::any_cast<const float *>(data));
    });
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        weights->map(getName()+std::string(".bn"), getNumber(), 2).with([&](const std::any & data) {
            weights_->extractBatchnormData(std::any_cast<const float *>(data));
        });
    }
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Set bias value (for unpadded outputs)
 *
 * @param outPass Output rendering pass
 * @param bias Pointer to weight array that stored biases and weights
 *
 * Depending on whether an output padding is selected, this preloads the target fraembuffers
 * with the bias values in case the output padding was zero. For non-zero paddings, the bias is
 * handled within the shader itself.
 */
void ConvLayerBase::setBias(int outPass, const UniformWeightArray *bias) {
    if (outputPadding_ > 0) {
        // if we have padding, the shader takes care, just clear the target FB here
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        // clear the target FB to the bias value
        const float * data = bias->getPackageBias(outPass);
        for (int i=0; i < bias->numRenderTargets(outPass); i++) {
            glClearBufferfv(GL_COLOR, i, data + i * PIXEL_PACKING);
        }
    }
}


/**
 * @brief Convolution-specific shader preprocessing on source level
 *
 * @param[inout] preproc Pointer to target pre-processor string which will be used as preprocessor
 *                       definitions with GPULayerBase::compileShaderPair
 *
 * @param maxChars Maximum available characters in the \p preproc array
 *
 * @return Remaining capacity in \p preproc buffer
 *
 * This function constructs (parts of) a preprocessor string for use in the vertex and fragment
 * shaders. It currently takes care of the following things:
 *  - kernel size
 *  - shader-controller bias
 *  - dilation for <i>a trous</i> convolution
 */
size_t ConvLayerBase::shaderPreprocessing(char *preproc, size_t maxChars) {
#if defined(WIN32) || defined(WIN64)
    using ssize_t = int64_t;
#endif
    char extra[80];
    ssize_t mc = (ssize_t)preprocessor_.generatePreprocessorPreamble(flags_, preproc, maxChars);
    if (outputPadding_ > 0) {
        strncat(preproc, "#define USE_BIAS\n", mc);
        mc = (ssize_t)maxChars - (ssize_t)strlen(preproc);
    }
    snprintf(extra, sizeof(extra), "#define CONVSIZE %d\n",kernel_);
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    snprintf(extra, sizeof(extra), "#define CONVMID %d\n",(kernel_-1)/2);
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    snprintf(extra, sizeof(extra), "#define DILATION %d\n",dilation_[0]);       // TODO (mw) support anisotropic dilation
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    return (size_t)std::max((ssize_t)0, mc);
}


/**
 * @brief Setup set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to VAO object that tracks the geometry definition
 * @param kernel Kernel size of the convolution
 *
 * @pre The vertex array object (\p vao ) to be used with the buffers created here is already bound
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up a tensor. Currently, the shaders
 * perform the horizontal convolution with multiple texture lookups and the polygons are used for
 * the vertical convolution by drawing multiple polygons that are shifted in their (input)
 * texture coordinates.
 */
void ConvLayerBase::setupNetworkPolygons(VAO *vao, int kernel) {
    int vertsize = 4, rvertsize = 2;
    int offset0 = 0, offset1 = 0;
    float * attrs0 = new float[vertsize*4*kernel];
    float * attrs1 = new float[rvertsize*4*kernel];
    float posleft  = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[0]);
    float posright =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[0]);
    float postop  =  -1.0f + ((float)(2*outputPadding_) / (float)viewport_[1]);
    float posbottom = 1.0f - ((float)(2*outputPadding_) / (float)viewport_[1]);
    float restop = (float)outputPadding_ / (float)viewport_[1];
    float resbottom = (float)(viewport_[1] - outputPadding_) / (float)viewport_[1];
    float resleft = (float)outputPadding_ / (float)viewport_[0];
    float resright = (float)(viewport_[0] - outputPadding_) / (float)viewport_[0];
    for (int conv = 0; conv < kernel; conv++) {
        float tleft,ttop;
        float thspan = (float)(width_) / (float)(width_ + 2*inputPadding_);
        float tvspan = (float)(height_) / (float)(height_ + 2*inputPadding_);
        tleft = (float)inputPadding_ / (float)(width_ + 2*inputPadding_);
        ttop = ((float)inputPadding_ + sourceStep_*(float)(conv-((kernel-1)/2)))/(float)(height_ + 2*inputPadding_);
        if (downsample_[0] > 1) {
            tleft -= sourceStep_ * 0.5f*(float)(downsample_[0]-1) / (float)(width_ + 2*inputPadding_);
        }
        if (downsample_[1] > 1) {
            ttop -= sourceStep_ * 0.5f*(float)(downsample_[1]-1) / (float)(height_ + 2*inputPadding_);
        }
        //-----------------------------------------------------
        // Positions first (output layers)
        //-----------------------------------------------------
        attrs0[offset0+0*vertsize+0] = posleft;
        attrs0[offset0+0*vertsize+1] = postop;
        attrs0[offset0+1*vertsize+0] = posleft;
        attrs0[offset0+1*vertsize+1] = posbottom;
        attrs0[offset0+2*vertsize+0] = posright;
        attrs0[offset0+2*vertsize+1] = posbottom;
        attrs0[offset0+3*vertsize+0] = posright;
        attrs0[offset0+3*vertsize+1] = postop;
        //-----------------------------------------------------
        // Texture offsets (input layers)
        //-----------------------------------------------------
        attrs0[offset0+0*vertsize+2] = tleft;
        attrs0[offset0+0*vertsize+3] = ttop;
        attrs0[offset0+1*vertsize+2] = tleft;
        attrs0[offset0+1*vertsize+3] = ttop+tvspan;
        attrs0[offset0+2*vertsize+2] = tleft+thspan;
        attrs0[offset0+2*vertsize+3] = ttop+tvspan;
        attrs0[offset0+3*vertsize+2] = tleft+thspan;
        attrs0[offset0+3*vertsize+3] = ttop;
        offset0 += 4*vertsize;
        //-----------------------------------------------------
        // Residual coordinates and flags...
        //-----------------------------------------------------
        attrs1[offset1+0*rvertsize+0] = resleft;
        attrs1[offset1+0*rvertsize+1] = restop;
        attrs1[offset1+1*rvertsize+0] = resleft;
        attrs1[offset1+1*rvertsize+1] = resbottom;
        attrs1[offset1+2*rvertsize+0] = resright;
        attrs1[offset1+2*rvertsize+1] = resbottom;
        attrs1[offset1+3*rvertsize+0] = resright;
        attrs1[offset1+3*rvertsize+1] = restop;
        offset1+=4*rvertsize;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, (int)(vertsize * 4 * kernel * sizeof(float)), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0, vertsize, GL_FLOAT, GL_FALSE, 0, 0);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        vao->enableArray(1);
        residualBuffer_ = new VBO(context_);
        residualBuffer_->setBufferData(attrs1, (int)(rvertsize * 4 * kernel * sizeof(float)), GL_STATIC_DRAW);
        residualBuffer_->bind();
        vao->setVertexAttributeBuffer(1, rvertsize, GL_FLOAT, GL_FALSE, 0, 0);
    }
    delete [] attrs0;
    delete [] attrs1;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[6*kernel];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i < kernel; i++) {
        int offset = i*4;
        indices[i*6+0] = (GLshort)(offset+0);
        indices[i*6+1] = (GLshort)(offset+1);
        indices[i*6+2] = (GLshort)(offset+2);
        indices[i*6+3] = (GLshort)(offset+0);
        indices[i*6+4] = (GLshort)(offset+2);
        indices[i*6+5] = (GLshort)(offset+3);
    }
    indexBuffer_->setBufferData(indices, (int)(6 * kernel * sizeof(GLshort)), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}



/**
 * @brief Prepare/initialize set of FBOs for writing the layer reults
 *
 * OpenGL requires a target framebuffer to render the data into. In order to use textures as a
 * buffer mechanism, instead of the default framebuffer (which is for example a surface that is
 * displayed on the screen) we use framebuffer objects (%FBO) that are backed by the output textures
 * to do the rendering. For each output texture there is one %FBO is used.
 *
 * @see updateFBO
 */
void ConvLayerBase::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    if (!weights_) THROW_EXCEPTION_ARGS(FynException,"No weights loaded");
    int texoffset=0;
    for (int pass=0; pass < weights_->numOutputRenderPasses(); pass++) {
        FBO * fbo = new FBO(context_, viewport_[0], viewport_[1], outputTextures_.at(texoffset++));
        fbo->bind();
        for (int i=1; i < weights_->numRenderTargets(pass); i++) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0+i,outputTextures_.at(texoffset++), GL_TEXTURE_2D);
        }
        fbo->setWriteMask();
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_=false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void ConvLayerBase::updateFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    if (!weights_) THROW_EXCEPTION_ARGS(FynException,"No weights loaded");
    int texoffset=0;
    for (int pass=0; pass < weights_->numOutputRenderPasses(); pass++) {
        FBO * fbo = framebuffers_.at(pass);
        fbo->bind();
        for (int i=0; i < weights_->numRenderTargets(pass); i++) {
            fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0+i, outputTextures_.at(texoffset++));
        }
        fbo->unbind();
    }
    outputChanged_ = false;
}

} // fyusion::fyusenet::gpu::vanilla namespace

// vim: set expandtab ts=4 sw=4:
