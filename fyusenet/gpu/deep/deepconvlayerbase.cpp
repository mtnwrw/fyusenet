//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// DeepConvolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glinfo.h"
#include "../gfxcontextlink.h"
#include "deepconvlayerbase.h"
#include "deeplayerbase.h"
#include "../floatconversion.h"

namespace fyusion::fyusenet::gpu::deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

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
DeepConvLayerBase::DeepConvLayerBase(const ConvLayerBuilder & builder, int layerNumber) : ConvLayerBase(builder, layerNumber),
    tiler_(new DeepTiler(builder.type_,builder.width(),builder.height(),builder.in(),builder.out(),(float)builder.upsample_[0]/(float)(builder.downsample_[0]),(float)builder.upsample_[1]/(float)(builder.downsample_[1]),builder.inputPadding_,builder.outputPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1],builder.kernel_)) {
    viewport_[0] = tiler_->getViewportWidth();
    viewport_[1] = tiler_->getViewportHeight();
    if (GLInfo::getGPUType() == GLInfo::ARM_MALI) {
        mali_ = true;
        std::string renderer = GLInfo::getRendererString();
        if (!renderer.empty()) {
            if (strstr(renderer.c_str(),"-T")) preG71_=true;
        }
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        // we use a temporary instance of the tiler to get the residual texture size right for the connector
        DeepTiler restiler(LayerType::RESIDUAL,builder.width(),builder.height(),builder.out(),builder.out(),(float)builder.upsample_[0]/(float)builder.downsample_[0],(float)builder.upsample_[1]/(float)builder.downsample_[1],0,builder.residualPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1]);
        residualViewport_[0] = restiler.getViewportWidth();
        residualViewport_[1] = restiler.getViewportHeight();
        // the actual tiler to be used for generating the polygons
        residualTiler_ = new DeepTiler(LayerType::RESIDUAL,builder.width(),builder.height(),builder.out(),builder.out(),(float)builder.upsample_[0]/(float)builder.downsample_[0],(float)builder.upsample_[1]/(float)builder.downsample_[1],builder.residualPadding_,builder.outputPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1]);
    }
    assert(dilation_[0] == dilation_[1]);
    largeDilation_ = (std::max(dilation_[0], dilation_[1]) * (kernel_ - 1)/2) > 7;
#ifdef HIGH_PRECISION
    halfSupport_ = false;
#else
    halfSupport_ = GLInfo::supportsHalf();
#endif
    hasParameters_ = true;
}


/**
 * @brief Constructor
 *
 * @param builder General layer builder that contains parameterization for the layer
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
DeepConvLayerBase::DeepConvLayerBase(const GPULayerBuilder & builder, int layerNumber) : ConvLayerBase(builder, layerNumber),
    tiler_(new DeepTiler(builder.type_,builder.width(),builder.height(),builder.in(),builder.out(),(float)builder.upsample_[0]/(float)(builder.downsample_[0]),(float)builder.upsample_[1]/(float)(builder.downsample_[1]),builder.inputPadding_,builder.outputPadding_,1,1,1,1,1)) {
    viewport_[0] = tiler_->getViewportWidth();
    viewport_[1] = tiler_->getViewportHeight();
    if (GLInfo::getGPUType() == GLInfo::ARM_MALI) {
        mali_ = true;
        std::string renderer = GLInfo::getRendererString();
        if (!renderer.empty()) {
            if (strstr(renderer.c_str(),"-T")) preG71_=true;
        }
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        // we use a temporary instance of the tiler to get the residual texture size right for the connector
        DeepTiler restiler(LayerType::RESIDUAL,builder.width(),builder.height(),builder.out(),builder.out(),(float)builder.upsample_[0]/(float)builder.downsample_[0],(float)builder.upsample_[1]/(float)builder.downsample_[1],0,builder.residualPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1]);
        residualViewport_[0] = restiler.getViewportWidth();
        residualViewport_[1] = restiler.getViewportHeight();
        // the actual tiler to be used for generating the polygons
        residualTiler_ = new DeepTiler(LayerType::RESIDUAL,builder.width(),builder.height(),builder.out(),builder.out(),(float)builder.upsample_[0]/(float)builder.downsample_[0],(float)builder.upsample_[1]/(float)builder.downsample_[1],builder.residualPadding_,builder.outputPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1]);
    }
#ifdef HIGH_PRECISION
    halfSupport_ = false;
#else
    halfSupport_ = GLInfo::supportsHalf();
#endif
    hasParameters_ = true;
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
DeepConvLayerBase::~DeepConvLayerBase()  {
    delete tiler_;
    delete residualTiler_;
    tiler_ = nullptr;
    residualTiler_ = nullptr;
    if ((vertexBuffer_) || (indexBuffer_) || (vertexArray_) || (textureOffsets_)) {
        FNLOGE("Cleanup not called");
        assert(false);
    }
    delete [] postBNBias_;
    delete [] postBNScales_;
    postBNBias_ = nullptr;
    postBNScales_ = nullptr;
}



/**
 * @brief Perform setup of layer code
 *
 * @pre The GL context that is to be used for running the inference must be current to the calling
 *      thread and loadParameters() has been called prior to this function.
 *
 * @post Layer is marked as valid
 *
 * This function sets up the required proxy polygons, FBOs and shaders. Check derived classes for
 * shader-specific initializations.
 *
 * @see setupNetworkPolygons, setupShaders, setupFBOs
 */
void DeepConvLayerBase::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
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
 * @copydoc GPULayerBase::cleanup
 */
void DeepConvLayerBase::cleanup() {
    delete residualBuffer_;
    delete vertexBuffer_;
    delete indexBuffer_;
    delete vertexArray_;
    delete textureOffsets_;
    if (weightTexture_) glDeleteTextures(1, &weightTexture_);
    if (biasTexture_) glDeleteTextures(1, &biasTexture_);
    if (inputCoordTexture_) glDeleteTextures(1, &inputCoordTexture_);
    textureOffsets_ = nullptr;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    residualBuffer_ = nullptr;
    inputCoordTexture_ = 0;
    weightTexture_ = 0;
    biasTexture_ = 0;
    ConvLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepConvLayerBase::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0 ,0, tiler_->getInputTextureWidth(), tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        result.push_back(BufferSpec(0, 1, residualViewport_[0], residualViewport_[1],
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::RESIDUAL_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepConvLayerBase::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}



/**
 * @copydoc LayerBase::isApplicable
 */
bool DeepConvLayerBase::isApplicable() const {
    if ((( GLInfo::isGLES() && GLInfo::getVersion() < GLInfo::GLES_3_0)) ||
        ((!GLInfo::isGLES()) && (GLInfo::getVersion() < GLInfo::GL_3_0))) {
        if (!GLInfo::hasExtension("GL_EXT_texture_integer")) {
            return false;
        }
    }
    if (GLInfo::getMaxVaryingVectors() < 8) {
        return false;
    }
    return true;
}



/**
 * @brief Read weights and biases from raw data and store them into a texture
 *
 * @param weightSource Pointer to ParameterProvider object that supplies all weight data
 *
 * This function extracts the weights and biases stored in the \p weightSource parameter for
 * usage with the GPU. For convolutions, the convolution weights are supposed to be
 * stored in the following nested order:
 *
 * @code
 * [outchannel][kernely][kernelx][inchannel]
 * @endcode
 *
 * Thus for \e m input channels and a kernel of size \e k (i.e. a \f$ k \times k \f$ kernel),
 * we expect a 4D array of size \f$ n \times k \times k \times m \f$. The bias data as well
 * is supposed to be stored as simple 1D vectors and the batch-norm parameters are expected
 * to be stored in the following order:
 *  1. all scales (single value per output channel for a total of \c output values)
 *  2. all offsets (single value per output channel for a total of \c output values)
 *
 * As opposed to the shallow tensor handling, it is not efficient to use multiple render passes
 * with changing uniforms for the convolution (at least not in my tests on mobile GPUs). Instead,
 * a different path is chosen, which packs the convolution coefficients into textures and uses a
 * few tricks - when available.
 *
 * The texture format for the convolution coefficients is as follows:
 *   - Pixel format is \c RGBA
 *   - Texture \e height corresponds to the number of output channels multiplied by the convolution
 *     kernel size
 *   - Texture \e width corresponds to the number of input channels multiplied by the convolution
 *     kernel size, padded for even size (there is an additional tweak, see below for details)
 *   - Each pixel in the texture corresponds to 4 (or 8) convolution coefficients that are laid
 *     out as parts of 4x4 matrices
 *   - Four (4) consecutive pixels in a row represent a 4x4 matrix with the input channels as their
 *     column space and the output channels as their row space
 *   - Depending on the convolution kernel size, \e k neighboring 4x4 matrices horizontally
 *     represent the horizontal part of the kernel and \e k neighboring 4x4 matrices vertically
 *     represent the vertical part of the kernel
 *   - I should really put a picture here
 *
 * An additional tweak to the setup described above is the capability to contract the VRAM
 * requirements by half. In order to do that, we do not use a floating-point texture, but a
 * 32-bit integer (per channel) texture. We then fit two 16-bit floating-point numbers in a
 * single channel and can reduce the texture width by 50% . This has to be decoded by the shader
 * later.
 *
 * The parameter provider is called with the following \c name parameters on loading data:
 *   - \c layername.weights for the weight data, \c subIndex set to 0
 *   - \c layername.bias for the bias data, \c subIndex set to 1
 *   - \c layername.bn for the batch-norm data, \c subIndex set to 2
 *
 *  Where \c layername is the name assigned to this layer by the builder
 */
void DeepConvLayerBase::loadParameters(const ParameterProvider *weightSource) {
    assert(weightSource);
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    // as we store matrices here, we have 4 items, therefore do not divide by PIXEL_PACKING
    int texwidth = ((inputChannels_ % PIXEL_PACKING)==0) ? inputChannels_ : inputChannels_ + (PIXEL_PACKING - (inputChannels_ % PIXEL_PACKING));
    texwidth *= kernel_;
    if (texwidth & 1) texwidth++;
    int texheight = ((outputChannels_ + (PIXEL_PACKING-1)) / PIXEL_PACKING) * kernel_;  // 4 pixels per matrix
#ifdef HIGH_PRECISION
    int checkwidth = texwidth;
#else
    int checkwidth = (halfSupport_) ? texwidth/2 : texwidth;
#endif
    if ((checkwidth > GLInfo::getMaximumTextureSize()) || (texheight > GLInfo::getMaximumTextureSize())) {
        THROW_EXCEPTION_ARGS(FynException, "Weights do not fit into GL texture");
    }
    if (auto wgtsrc = weightSource->get(getName()+std::string(".weights"), getNumber(), 0) ; !wgtsrc.empty()) {
        float * weights = new float[texwidth * texheight * PIXEL_PACKING];
        memset(weights, 0, texwidth * texheight * PIXEL_PACKING * sizeof(float));
        const float *srcweights = std::any_cast<const float *>(wgtsrc.get());
        for (int outlayer = 0; outlayer < outputChannels_; outlayer += PIXEL_PACKING) {
            int orem = ((outputChannels_ - outlayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (outputChannels_ - outlayer);
            for (int fy = 0; fy < kernel_; fy++) {
                float *wptr = weights + ((outlayer / PIXEL_PACKING) * kernel_ + fy) * (texwidth * PIXEL_PACKING);
                // below defines one row in the target texture
                for (int inlayer = 0; inlayer < inputChannels_; inlayer += PIXEL_PACKING) {
                    int irem = ((inputChannels_ - inlayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_ - inlayer);
                    for (int fx = 0; fx < kernel_; fx++) {
                        for (int ol = outlayer; ol < outlayer + orem; ol++) {
                            for (int il = inlayer; il < inlayer + irem; il++) {
                                int srcoffset = ol * (kernel_ * kernel_ * inputChannels_) + ((fy * kernel_ + fx) * inputChannels_) + il;
                                *wptr = srcweights[srcoffset];
                                wptr++;
                            }
                            wptr += PIXEL_PACKING - irem;
                        }
                        wptr += (PIXEL_PACKING - orem) * PIXEL_PACKING;
                    }
                }
            }
        }
        if (!weightTexture_) glGenTextures(1, &weightTexture_);
        glBindTexture(GL_TEXTURE_2D, weightTexture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifndef HIGH_PRECISION
        if (halfSupport_) {
            unsigned int *fp16 = FloatConversion::getInstance()->toFP16UI(weights, texwidth * texheight * PIXEL_PACKING);
#ifdef GL_RGBA32UI
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, texwidth / 2, texheight, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, fp16);
#else
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32UI_EXT,texwidth/2,texheight,0,GL_RGBA_INTEGER_EXT,GL_UNSIGNED_INT,fp16);
#endif
            delete[] fp16;
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, texwidth, texheight, 0, GL_RGBA, GL_FLOAT, weights);
        }
#else
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,texwidth,texheight,0,GL_RGBA,GL_FLOAT,weights);
#endif
        delete[] weights;
    }
    //------------------------------------------------------
    // If we have the post-BN flag set, store the batchnorm
    // stuff...
    //------------------------------------------------------
    if (flags_ & fyusenet::LayerFlags::POST_BATCHNORM) {
        auto bnsrc = weightSource->get(getName()+".bn", getNumber(), 2);
        if (bnsrc.empty()) {
            THROW_EXCEPTION_ARGS(FynException, "Cannot find batchnorm data for layer %s", getName().c_str());
        }
        int padout = PIXEL_PACKING * ((outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING);
        const float * srcbn = std::any_cast<const float *>(bnsrc.get());
        postBNScales_ = new float[padout];
        postBNBias_ = new float[padout];
        memset(postBNScales_, 0, padout*sizeof(float));
        memset(postBNBias_, 0, padout*sizeof(float));
        memcpy(postBNScales_, srcbn, outputChannels_*sizeof(float));
        memcpy(postBNBias_, srcbn+outputChannels_, outputChannels_*sizeof(float));
    }
    //------------------------------------------------------
    // Now for the bias part (and also batchnorm)...
    //------------------------------------------------------
    int bs = PIXEL_PACKING*(1+(outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING);
    if (flags_ & LayerFlags::POST_BATCHNORM) bs *= 2;
    float * bias = new float[bs];
    memset(bias, 0, bs*sizeof(float));
    weightSource->map(getName()+std::string(".bias"), getNumber(), 1).with([&](const std::any &data) {
        memcpy(bias + PIXEL_PACKING, std::any_cast<const float *>(data), outputChannels_ * sizeof(float));
    });
    // load batchnorm scale and bias if necessary
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        for (int i=0; i < outputChannels_; i++) {
            bias[PIXEL_PACKING+i] = bias[PIXEL_PACKING+i] * postBNScales_[i] + postBNBias_[i];
            bias[PIXEL_PACKING+(bs/2)+i] = postBNScales_[i];
        }
    }
    if (!biasTexture_) glGenTextures(1,&biasTexture_);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifdef HIGH_PRECISION
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,1+(outputChannels_+PIXEL_PACKING-1)/PIXEL_PACKING,(flags_ & LayerFlags::POST_BATCHNORM) ? 2 : 1,0,GL_RGBA,GL_FLOAT,bias);
#else
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,1+(outputChannels_+PIXEL_PACKING-1)/PIXEL_PACKING,(flags_ & LayerFlags::POST_BATCHNORM) ? 2 : 1,0,GL_RGBA,GL_FLOAT,bias);
#endif
    delete [] bias;
}


/**
 * @copydoc LayerBase::writeResult
 */
void DeepConvLayerBase::writeResult(const char *fileName,bool includePadding) {
    // FIXME (mw) this is a copy of the same method in DeepLayerBase, fix the inheritance
#ifdef DEBUG
    int owidth = tiler_->getViewportWidth();
    int oheight = tiler_->getViewportHeight();
    float * data = new float[oheight*owidth*PIXEL_PACKING];
    int lwidth = tiler_->getOutputWidth();
    int lheight = tiler_->getOutputHeight();
    if (includePadding) {
        lwidth += 2*inputPadding_;
        lheight += 2*inputPadding_;
    }
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(fileName,"wb");
    if (out) {
#else
    uint8_t * download = new uint8_t[lwidth * lheight * outputChannels_];
    uint8_t * downptr = download;
    if (true) {
#endif
        float * layer = new float[lwidth*lheight];
        memset(layer,0,lwidth*lheight*sizeof(float));
        int layernum=0;
        for (int fb = 0 ; fb < numFBOs(); fb++ ) {
            memset(data, 0, owidth*oheight*PIXEL_PACKING*sizeof(float));
            FBO *fbo = getFBO(fb);
            fbo->writeToMemory<float,GL_FLOAT>(data,PIXEL_PACKING, (GLsizei)(owidth * oheight * PIXEL_PACKING * sizeof(float)));
            for (int ty=0; ty < tiler_->numOutputTiles(DeepTiler::VERTICAL); ty++) {
                for (int tx=0; tx < tiler_->numOutputTiles(DeepTiler::HORIZONTAL); tx++) {
                    int rem = ((outputChannels_ - layernum) > PIXEL_PACKING) ? PIXEL_PACKING : outputChannels_-layernum;
                    float * in = data + ((outputPadding_ + ty*(lheight+outputPadding_))*owidth + outputPadding_+tx*(lwidth+outputPadding_))*PIXEL_PACKING;
                    float * outptr = (includePadding) ? layer + (outputPadding_*lwidth) + outputPadding_ : layer;
                    for (int l=0; l < rem;l++) {
                        for (int y=0; y < lheight;y++) {
                            for (int x=0; x < lwidth;x++) {
                                outptr[x+y*lwidth] = in[(y*owidth+x)*PIXEL_PACKING+l];
                            }
                        }
#ifndef FYUSENET_USE_WEBGL
                        fwrite(layer,1,lwidth*lheight*sizeof(float),out);
#else
                        memcpy(downptr, layer, owidth * oheight * sizeof(float));
                        downptr += lwidth * lheight;
#endif
                    }
                    layernum += PIXEL_PACKING;
                }
            }
        }
        delete [] data;
        delete [] layer;
#ifndef FYUSENET_USE_WEBGL
        fclose(out);
#else
        EM_ASM({window.download($0, $1, $2);}, download, owidth * oheight * outputChannels_ * sizeof(float), fileName);
        delete [] download;
#endif
    } else FNLOGE("Cannot open file %s for writing",fileName);
#endif
}


/**
 * @copydoc GPULayerBase::copyResult
 */
void DeepConvLayerBase::copyResult(float *memory, bool includePadding) {
    // FIXME (mw) this is a copy of the same method in DeepLayerBase, fix the inheritance
#ifdef DEBUG
    if (memory) {
        int owidth = tiler_->getViewportWidth();
        int oheight = tiler_->getViewportHeight();
        float * data = new float[oheight*owidth*PIXEL_PACKING];
        int lwidth = tiler_->getOutputWidth();
        int lheight = tiler_->getOutputHeight();
        if (includePadding) {
            lwidth += 2*inputPadding_;
            lheight += 2*inputPadding_;
        }
        int layernum=0;
        float * layer = memory;
        for (int fb = 0 ; fb < numFBOs(); fb++ ) {
            memset(data, 0, owidth*oheight*PIXEL_PACKING*sizeof(float));
            FBO *fbo = getFBO(fb);
            assert(fbo->numAttachments() == 1);
            fbo->writeToMemory<float,GL_FLOAT>(data,PIXEL_PACKING,(GLsizei)(owidth * oheight * PIXEL_PACKING * sizeof(float)));
            for (int ty=0; ty < tiler_->numOutputTiles(DeepTiler::VERTICAL); ty++) {
                for (int tx=0; tx < tiler_->numOutputTiles(DeepTiler::HORIZONTAL); tx++) {
                    int rem = ((outputChannels_ - layernum) > PIXEL_PACKING) ? PIXEL_PACKING : outputChannels_ - layernum;
                    const float * in = data + ((outputPadding_ + ty*(lheight + outputPadding_))*owidth + outputPadding_ + tx*(lwidth + outputPadding_))*PIXEL_PACKING;
                    float * outptr = (includePadding) ? layer + (outputPadding_ * lwidth) + outputPadding_ : layer;
                    for (int l=0; l < rem; l++) {
                        for (int y=0; y < lheight; y++) {
                            for (int x=0; x < lwidth; x++) {
                                outptr[x+y*lwidth] = in[(y*owidth+x)*PIXEL_PACKING+l];
                            }
                        }
                        layer += lwidth*lheight;
                        outptr += lwidth*lheight;
                    }
                    layernum += PIXEL_PACKING;
                }
            }
        }
        delete [] data;
    }
#endif
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



/**
 * @copydoc ConvLayerBase::setupShaders
 */
void DeepConvLayerBase::setupShaders() {
    char preproc[1024] = {0};
    snprintf(preproc, sizeof(preproc), "#define DISP_UNIT %d\n#define WEIGHT_UNIT %d\n#define BIAS_UNIT %d\n",DISP_TEXTURE,WEIGHT_TEXTURE,BIAS_TEXTURE);
    shaderPreprocessing(preproc, sizeof(preproc) - strlen(preproc)-1);
    compileConvolutionShaders(preproc);
}


/**
 * @brief Convolution-specific shader preprocessing on source level
 *
 * @param[inout] preproc Pointer to target pre-processor string which will be used as preprocessor
 *                       definitions with GPULayerBase::compileShaderPair
 *
 * @param maxChars Maximum available characters in the \p preproc array
 *
 * @return Remaining capacity in preprocessor string
 *
 * This function constructs (parts of) a preprocessor string for use in the vertex and fragment
 * shaders. It currently takes care of the following things:
 *  - kernel size
 *  - shader-controller bias
 *  - dilation for <i>a trous</i> convolution
 */
size_t DeepConvLayerBase::shaderPreprocessing(char *preproc, size_t maxChars) {
#if defined(WIN32) || defined(WIN64)
        using ssize_t = int64_t;
#endif
    char extra[128];
    ssize_t mc = (ssize_t)preprocessor_.generatePreprocessorPreamble(flags_ & (~LayerFlags::RESIDUAL_INPUT), preproc, maxChars);
    assert(mc > 0);
    if ((mali_) && (kernel_ > 1)) {
        strncat(preproc, "#define MALI\n", mc);
        mc = (ssize_t)maxChars - (ssize_t)strlen(preproc);  // ouch
    }
    if (preG71_) {
        strncat(preproc, "#define PRE_G71\n", mc);
        mc = (ssize_t)maxChars - (ssize_t)strlen(preproc);  // ouch
    }
    snprintf(extra, sizeof(extra), "#define KERNEL %d\n",kernel_);
    strncat(preproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    if (largeDilation_) {
        snprintf(extra, sizeof(extra),"#define LARGE_DILATION\n");
    } else {
        // NOTE (mw) for now we can only handle isotropic dilation
        assert(dilation_[0] == dilation_[1]);
        snprintf(extra, sizeof(extra),"#define DILATION %d\n", dilation_[0]);
    }
    strncat(preproc, extra, mc);
    return mc - (ssize_t)strlen(extra);
}


/**
 * @brief Process shader before linking and perform the actual linking
 *
 * @param shader Shared pointer to shader program
 *
 * This function binds the locations of the vertex shader attributes to the correct index and
 * then finally links the shader.
 */
void DeepConvLayerBase::shaderPostprocessing(programptr & shader) {
    try {
        shader->bindAttributeLocation("attributes0", 0);  // NOTE (mw) quite specialized for a base class, maybe rethink
        shader->bindAttributeLocation("attributes1", 1);  // NOTE (mw) quite specialized for a base class, maybe rethink
        shader->bindAttributeLocation("attributes2", 2);  // NOTE (mw) quite specialized for a base class, maybe rethink
        shader->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
}


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to vertex array object that the resulting VBO and IBO are tied to
 *
 * @pre The supplied \p vao vertex array object to be used with this VBO is already bound
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up the output tensor.
 */
void DeepConvLayerBase::setupNetworkPolygons(VAO *vao) {
    int offset0=0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    std::vector<DeepTiler::Tile> tiles = tiler_->createOutputTiles();
    DeepTiler::Tile deftex = tiler_->getDefaultTextureExtents();
    //---------------------------------------------
    // VBO parts, first the default output tiling
    //---------------------------------------------
    for (DeepTiler::Tile & tile : tiles) {
        tile.toFloatVec(attrs0,offset0,4);
        deftex.toFloatVec(attrs0,offset0+2,4);
        offset0 += 4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, (int)(tiler_->numOutputTiles() * 4 * 4 *sizeof(float)),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //---------------------------------------------
    // Now indices for the bias texture and the row
    // indices for the convolution coeffs (y-part
    // of the convolution)...
    //---------------------------------------------
    int * attrs1 = new int[tiler_->numOutputTiles()*2*4];
    memset(attrs1, 0, tiler_->numOutputTiles()*2*4*sizeof(int));
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        for (int j=0; j < 4; j++) {
            attrs1[(i*4+j) * 2 + 0] = i*kernel_;
            attrs1[(i*4+j) * 2 + 1] = i;          // to be used for indexing bias texture
        }
    }
    textureOffsets_ = new VBO(context_);
    vao->enableArray(1);
    textureOffsets_->setBufferData(attrs1, (int)(tiler_->numOutputTiles() * 2 * 4 *sizeof(int)), GL_STATIC_DRAW);
    textureOffsets_->bind();
    vao->setVertexAttributeBuffer(1, 2, GL_INT, 0, 0);
    delete [] attrs1;
    //---------------------------------------------
    // VBO for optional residual input (to be added
    // to the output after BN/ReLU)
    //---------------------------------------------
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        assert(residualTiler_->numOutputTiles() == residualTiler_->numInputTiles());
        float * attrs2 = new float[residualTiler_->numInputTiles()*2*4];
        std::vector<DeepTiler::Tile> rtiles = residualTiler_->createInputTiles(0,0,0);
        int offset2=0;
        for (DeepTiler::Tile tile : rtiles) {
            tile.toFloatVec(attrs2,offset2,2);
            offset2 += 2*4;
        }
        residualBuffer_ = new VBO(context_);
        vao->enableArray(2);
        residualBuffer_->setBufferData(attrs2, (int)(residualTiler_->numInputTiles() * 2 * 4 *sizeof(float)),GL_STATIC_DRAW);
        residualBuffer_->bind();
        vao->setVertexAttributeBuffer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
        delete [] attrs2;
    }
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        int offset = i*4;
        indices[i*6+0] = (GLshort)(offset + 0);
        indices[i*6+1] = (GLshort)(offset + 1);
        indices[i*6+2] = (GLshort)(offset + 2);
        indices[i*6+3] = (GLshort)(offset + 0);
        indices[i*6+4] = (GLshort)(offset + 2);
        indices[i*6+5] = (GLshort)(offset + 3);
    }
    indexBuffer_->setBufferData(indices, (int)(6 * tiler_->numOutputTiles() * sizeof(GLshort)),GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
    //---------------------------------------------------------------------------
    // Dependent texture to perform input lookup in the vertex shader. Takes care
    // of accumulating all input channels to a set of output channels and also
    // shifts the conv-window along the y direction. For each input tile one column
    // in the texture is generated with height equivalent to the kernel size.
    // Each entry in that texture contains a 2D displacement w.r.t. the input
    // texture coordinate system which takes care of the vertical convolution
    // direction...
    //---------------------------------------------------------------------------
    glGenTextures(1,&inputCoordTexture_);
    glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    float * texdata = new float[tiler_->numInputTiles()*4*kernel_];
    DeepTiler::Tile defex = tiler_->getDefaultTextureExtents();
    if ((kernel_ & 1) == 0) {
        THROW_EXCEPTION_ARGS(FynException,"Unsupported window size");
    } else {
        for (int w = -((kernel_ - 1) / 2) ; w <= ((kernel_- 1 ) / 2); w++) {            // currently only odd kernel sizes are supported
            std::vector<DeepTiler::Tile> tiles = tiler_->createInputTiles(0, w * dilation_[1]);
            int offset = (w + ((kernel_ - 1) / 2)) * tiler_->numInputTiles() * 4;
            for (DeepTiler::Tile & tile : tiles) {
                tile.toDisplacement(defex,texdata,offset);
                tile.lowClamp(texdata, offset+2);
                offset += 4;
            }
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tiler_->numInputTiles(), kernel_, 0, GL_RGBA, GL_FLOAT, texdata);
    delete [] texdata;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void DeepConvLayerBase::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    FBO * fbo = new FBO(context_,viewport_[0],viewport_[1],outputTextures_.at(0));
    fbo->bind();
    fbo->setWriteMask();
    fbo->unbind();
    framebuffers_.push_back(fbo);
    outputChanged_ = false;
}



/**
 * @copydoc GPULayerBase::updateFBOs
 */
void DeepConvLayerBase::updateFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    FBO * fbo = framebuffers_.at(0);
    fbo->bind();
    fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0,outputTextures_.at(0));
    fbo->unbind();
    outputChanged_ = false;
}

/**
 * @copydoc GPULayerBase::getInputOrder
 */
BufferSpec::order DeepConvLayerBase::getInputOrder(int port) const {
    return BufferSpec::order::GPU_DEEP;
}

/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order DeepConvLayerBase::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_DEEP;
}



} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
