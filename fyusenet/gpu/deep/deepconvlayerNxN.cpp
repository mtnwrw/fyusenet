//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ odd NxN mask and N > 3
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/glinfo.h"
#include "../../gl/glexception.h"
#include "../uniformweightarray.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "deepconvlayerNxN.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion::fyusenet::gpu::deep {

//-------------------------------------- Local Definitions -----------------------------------------

static const char * OFFSET_DEFS_3 = "#define OFFSET0 2\n"
                                    "#define OFFSET3a 0\n"
                                    "#define OFFSET3b 4\n";

static const char * OFFSET_DEFS_5 = "#define OFFSET0 4\n"
                                    "#define OFFSET3a 2\n"
                                    "#define OFFSET3b 6\n"
                                    "#define OFFSET5a 0\n"
                                    "#define OFFSET5b 8\n";

static const char * OFFSET_DEFS_7 = "#define OFFSET0 6\n"
                                    "#define OFFSET3a 4\n"
                                    "#define OFFSET3b 8\n"
                                    "#define OFFSET5a 2\n"
                                    "#define OFFSET5b 10\n"
                                    "#define OFFSET7a 0\n"
                                    "#define OFFSET7b 12\n";

static const char * OFFSET_DEFS_2 = "#define OFFSET0 2\n"
                                    "#define OFFSET2a 0\n";

static const char * OFFSET_DEFS_4 = "#define OFFSET0 4\n"
                                    "#define OFFSET2a 2\n"
                                    "#define OFFSET2b 6\n"
                                    "#define OFFSET4a 0\n";

static const char * OFFSET_DEFS_6 = "#define OFFSET0 6\n"
                                    "#define OFFSET2a 4\n"
                                    "#define OFFSET2b 8\n"
                                    "#define OFFSET4a 2\n"
                                    "#define OFFSET4b 10\n"
                                    "#define OFFSET6a 0\n";

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc DeepConvLayerBase::DeepConvLayerBase(const GPULayerBuilder&, int)
 */
DeepConvLayerNxN::DeepConvLayerNxN(const ConvLayerBuilder& builder,int layerNumber) : DeepConvLayerBase(builder, layerNumber) {
    assert((builder.kernel_ % 2) == 1);
    assert(builder.kernel_ >= 3);
    assert(builder.groupSize_ == 1);
    maxVectors_ = GLInfo::getMaxVaryingVectors();
    maxKernelWidth_ = (halfSupport_) ? (maxVectors_ - BASE_VECTORS) / 2 : (maxVectors_ - BASE_VECTORS) / 4;
    partialConv_ = (maxKernelWidth_ < kernel_);
    int maxdilstep = std::max(dilation_[0], dilation_[1]);
    if (partialConv_) {
        int partialclip = std::min(7, maxKernelWidth_);
        int kernel = builder.kernel_;
        while (kernel > 0) {
            if (kernel > partialclip) {
                horizSplits_.push_back(partialclip);
                numSplits_++;
                kernel -= partialclip;
            } else {
                if (kernel == 1) {
                    horizSplits_[numSplits_-1]--;
                    kernel++;
                }
                horizSplits_.push_back(kernel);
                kernel = 0;
            }
        }
        int maxpartial = 0;
        std::for_each(horizSplits_.begin(), horizSplits_.end(), [&](int val) { maxpartial = std::max(maxpartial, val); });
        largeDilation_ = (maxdilstep * (maxpartial - 1)/2) > 7;
    }
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepConvLayerNxN::cleanup() {
    shaderStates_.clear();
    noBiasShaderStates_.clear();
    shaders_.clear();
    noBiasShaders_.clear();
    DeepConvLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::forward
 */
void DeepConvLayerNxN::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif    
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE0+DISP_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, inputCoordTexture_);
    glActiveTexture(GL_TEXTURE0+WEIGHT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, weightTexture_);
    glActiveTexture(GL_TEXTURE0+BIAS_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, biasTexture_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Residual flag configured, but no such texture found.");
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, residualTextures_.at(0));
    }
    if (!partialConv_) nonPartialRender();
    else partialRender();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}




/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Execute rendering steps for larger kernel sizes using multiple passes
 */
void DeepConvLayerNxN::partialRender() {
    int tris = tiler_->numOutputTiles();
    int instances = tiler_->numInputTiles() * kernel_;
    for (int part=0; part <= numSplits_; part++) {
        shaders_[part]->bind(shaderStates_.at(part).get());
        glDrawElements(GL_TRIANGLES, tris*6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
        shaders_[part]->unbind((instances > 1) || (part < numSplits_));
    }
    if (instances > 1) {
        for (int part = 0; part <= numSplits_; part++) {
            noBiasShaders_[part]->bind(noBiasShaderStates_.at(part).get());
            glDrawElementsInstanced(GL_TRIANGLES, tris * 6, GL_UNSIGNED_SHORT, (const GLvoid *) 0, instances - 1);
            noBiasShaders_[part]->unbind((part != numSplits_));
        }
    }
}



/**
 * @brief Execute rendering steps for small kernel sizes using two passes
 */
void DeepConvLayerNxN::nonPartialRender() {
    assert(shaders_.size() == 1);
    int instances = tiler_->numInputTiles() * kernel_;
    int tris = tiler_->numOutputTiles();
    shaders_[0]->bind(shaderStates_.at(0).get());
    glDrawElements(GL_TRIANGLES, tris*6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
    shaders_[0]->unbind((instances > 1));
    if (instances > 1)  {
        noBiasShaders_.at(0)->bind(noBiasShaderStates_.at(0).get());
        glDrawElementsInstanced(GL_TRIANGLES, tris*6, GL_UNSIGNED_SHORT, (const GLvoid *)0, instances-1);
        noBiasShaders_[0]->unbind();
    }
}



/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepConvLayerNxN::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1536] = {0};
    if (!partialConv_) {
        char vtxshader[80], frgshader[80];
        snprintf(vtxshader, sizeof(vtxshader), "shaders/deep/deepconv%dx%d_tiled.vert", kernel_, kernel_);
        snprintf(frgshader, sizeof(frgshader), "shaders/deep/deepconv%dx%d_tiled.frag", kernel_, kernel_);
        strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
        // NOTE (mw) only add residual on the first pass (the shader preprocessing masks out the residual flag for the deepconv layers)
        if (flags_ & LayerFlags::RESIDUAL_INPUT) strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
        shaders_.push_back(compileShaderPair(vtxshader, frgshader, finalpreproc,typeid(this)));
        assert(shaders_[0]);
        shaderPostprocessing(shaders_[0]);
        shaderStates_.push_back(initShader(shaders_[0], 0, 0, kernel_));
        strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
        strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
        noBiasShaders_.push_back(compileShaderPair(vtxshader, frgshader, finalpreproc,typeid(this)));
        assert(noBiasShaders_[0]);
        shaderPostprocessing(noBiasShaders_[0]);
        noBiasShaderStates_.push_back(initShader(noBiasShaders_[0], 0, 0, kernel_));
    } else {
        static const char * oddfrag = "shaders/deep/deepconvNxN_partial_odd.frag";
        static const char * evenfrag = "shaders/deep/deepconvNxN_partial_even.frag";
        char extra[256];
        int kerneloffset = 0;
        // NOTE (mw) we assume odd kernels for now
        for (int part=0, offset=-(kernel_-1)/2; part <= numSplits_; part++) {
            int subkern = horizSplits_.at(part);
            bool odd = ((subkern & 1) == 1);
            int subext = (odd) ? (subkern-1)/2 : subkern/2;
            int horizoffset = offset+subext;
            int varyings = subkern * ((halfSupport_) ? 2 : 4);
            snprintf(extra, sizeof(extra), "#define COEFF_VARYINGS %d\n#define NET_KERNEL %d\n", varyings, subkern);
            if (part > 0) strncat(extra, "#define NO_BIAS\n", sizeof(extra) - strlen(extra) - 1);   // NOTE (mw) we disable bias if not in part 0
            appendOffsetDefs(extra, subkern, sizeof(extra) - strlen(extra) -1);
            strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
            // NOTE (mw) only add residual on the first pass and part 0 (the shader preprocessing masks out the residual flag for the deepconv layers)
            if ((flags_ & LayerFlags::RESIDUAL_INPUT) && (part == 0)) {
                strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
            }
            strncat(finalpreproc, extra, sizeof(finalpreproc) - strlen(finalpreproc) - 1);
            programptr prog = compileShaderPair("shaders/deep/deepconvNxN_partial.vert",
                                                (odd) ? oddfrag : evenfrag, finalpreproc, typeid(this));
            shaderPostprocessing(prog);
            shaders_.push_back(prog);
            shaderStates_.push_back(initShader(prog, horizoffset, kerneloffset, kernel_));

            strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
            strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
            strncat(finalpreproc, extra, sizeof(finalpreproc) - strlen(finalpreproc) - 1);
            prog = compileShaderPair("shaders/deep/deepconvNxN_partial.vert",
                                     (odd) ? oddfrag : evenfrag, finalpreproc, typeid(this));
            shaderPostprocessing(prog);
            noBiasShaders_.push_back(prog);
            noBiasShaderStates_.push_back(initShader(prog, horizoffset, kerneloffset, kernel_));
            kerneloffset += subkern;
            offset += subkern;
        }
    }
}


/**
 * @brief Append preprocessor definitions to GLSL code that define offsets into the interface arrays
 *
 * @param[out] string Pointer to string to write text to
 *
 * @param kernel Partial kernel size
 *
 * @param maxChars Capacity of the supplied \p string
 */
void DeepConvLayerNxN::appendOffsetDefs(char *string, int kernel, size_t maxChars) {
    static const char * oddPointers[3] = {OFFSET_DEFS_3, OFFSET_DEFS_5, OFFSET_DEFS_7};
    static const char * evenPointers[3] = {OFFSET_DEFS_2, OFFSET_DEFS_4, OFFSET_DEFS_6};
    if (kernel & 1) {
        assert(kernel >= 3);
        strncat(string, oddPointers[(kernel-3)/2], maxChars);
    } else {
        assert(kernel >= 2);
        strncat(string, evenPointers[(kernel-2)/2], maxChars);
    }
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 * @param horizOffset Horizontal offset within the input tensor
 * @param kernelOffset Horizontal offset within the convolution kernel
 * @param kernelY Window size of convolution kernel (vertical part)
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shader
 */
unistateptr DeepConvLayerNxN::initShader(programptr shader, int horizOffset, int kernelOffset, int kernelY) {
    unistateptr state = UniformState::makeShared(shader);
    // TODO (mw) check partial
    if (!GLInfo::hasBinding()) {
        state->setUniformValue("inputLayer0",0);
        state->setUniformValue("residualLayer0", 1, true);
        state->setUniformValue("inputDisplacements", DISP_TEXTURE);
        state->setUniformValue("inputCoeffs", WEIGHT_TEXTURE);
        state->setUniformValue("biasTexture", BIAS_TEXTURE, true);
    }
    if (largeDilation_) {
        state->setUniformValue("dilationStep", tiler_->getTextureStepX() * (float)dilation_[0]);
    }
    state->setUniformValue("horizOffset", horizOffset, true);
    state->setUniformValue("textureStep", tiler_->getTextureStepX(), true);
    state->setUniformValue("instancesPerTile", kernelY);
    state->setUniformValue("numParts", numSplits_+1, true);
    state->setUniformValue("kernelOffset", kernelOffset, true);
    return state;
}



} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
