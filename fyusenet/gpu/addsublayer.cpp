//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Explicit Add/Sub Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "addsublayer.h"

namespace fyusion::fyusenet::gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
AddSubLayer::AddSubLayer(const GPULayerBuilder & builder, int layerNumber) :
    FunctionLayer(builder, layerNumber) {
    if (builder.getFlags() & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual handling is not supported by this layer");
    if (builder.type_ == LayerType::SUB) negative_ = true;
    //---------------------------------------------
    // We might need to update the maximum number of render
    // targets here, because we require double the input on
    // the texture side...
    //---------------------------------------------
    if (GLInfo::getMaximumTextureUnits() < maxRenderTargets_ * 2) {
        maxRenderTargets_ = std::min(maxRenderTargets_, GLInfo::getMaximumTextureUnits() / 2);
    }
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void AddSubLayer::cleanup() {
    for (int i=0; i < maxRenderTargets_; i++) {
        shaders_[i].reset();
        shaderStates_[i].reset();
    }
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> AddSubLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    texturesPerPort_ = 0;
    for (int port=0; port < 2; port++) {
        int channel=0;
        int rem = inputChannels_;
        if (rem < PIXEL_PACKING) {
            // for input textures, we support textures with less than 4 channels (might be from upload)
            auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
            result.emplace_back(channel++, port, width_+2*inputPadding_, height_ + 2*inputPadding_,
                                format.first, format.second, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE, rem);
            if (port == 0) texturesPerPort_++;
        } else {
            while (rem > 0) {
                result.emplace_back(channel++, port,
                                    width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                                    TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::FUNCTION_SOURCE, std::min(rem, PIXEL_PACKING));
                rem -= PIXEL_PACKING;
                if (port == 0) texturesPerPort_++;
            }
        }
    }
    return result;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void AddSubLayer::beforeRender() {
    currentShader_ = nullptr;
}


/**
 * @brief FunctionLayer::afterRender
 */
void AddSubLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void AddSubLayer::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    for (int tex=0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+2*tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    for (int tex=0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+2*tex+1);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset+texturesPerPort_));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
}


//0 1


/**
 * @copydoc FunctionLayer::setupShaders
 */
void AddSubLayer::setupShaders() {
    char preproc[1024] = {0};
    for (int rt=1; rt <= maxRenderTargets_; rt++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n#define SIGNED %d\n",rt, (negative_) ? 1 : 0);
        preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
        shaders_[rt-1] = compileShader(preproc);
        unistateptr state = UniformState::makeShared(shaders_[rt-1]);
        for (int j=0; j < rt ; j++) {
            char var[128];
            snprintf(var, sizeof(var), "op1Layer%d", j);
            state->setUniformValue(var, 2*j);
            snprintf(var, sizeof(var), "op2Layer%d", j);
            state->setUniformValue(var, 2*j+1);
        }
        shaderStates_[rt-1] = state;
    }
}


/**
 * @brief Compile tensor addition shader using supplied preprocessor definitions
 *
 * @param preproc Preprocessor definitions to use
 *
 * @return Shared pointer to compiled and linked shader program
 */
programptr AddSubLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert","shaders/add.frag",preproc,typeid(this));
    try {
        shader->bindAttributeLocation("attributes0",0);
        shader->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    return shader;
}



} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
