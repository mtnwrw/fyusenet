//--------------------------------------------------------------------------------------------------
// FyuseNet                                                                    (c) Fyusion Inc. 2016
//--------------------------------------------------------------------------------------------------
// Shallow to Deep Layer Converter
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "shallow2deep.h"
#include "../common/miscdefs.h"
#include "deep/deeptiler.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------



/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
Shallow2DeepLayer::Shallow2DeepLayer(const GPULayerBuilder& builder, int layerNumber) :
    DeepLayerBase(builder, layerNumber) {
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException, "This layer does not support residual input");
    maxInputTextures_ = std::min(maxInputTextures_, GLInfo::getMaximumTextureUnits());
    int numintex = (builder.in()+PIXEL_PACKING-1) / PIXEL_PACKING;
    numRenderPasses_ = (numintex+maxInputTextures_-1) / maxInputTextures_;
}


/**
 * @copydoc LayerBase::setup
 */
void Shallow2DeepLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void Shallow2DeepLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (outputChanged_) updateFBOs();
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f ,0.0f, 0.0f);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    framebuffers_.at(0)->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    vertexArray_->bind();
    shader_->bind();
    int intexoffset = 0;
    int quadoffset = 0;
    for (int pass = 0 ; pass < numRenderPasses_; pass++) {
        int quads=0;
        for (int it = 0; it < maxInputTextures_; it++) {
            if (intexoffset >= (int)inputTextures_.size()) break;
            glActiveTexture(GL_TEXTURE0 + it);
            glBindTexture(GL_TEXTURE_2D, inputTextures_.at(intexoffset++));
            quads++;
        }
        glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)(quadoffset*6*sizeof(short)));
        quadoffset += quads;
    }
    shader_->unbind();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void Shallow2DeepLayer::cleanup() {
    FNET_DEL_AND_CLEAR(vertexBuffer_);
    FNET_DEL_AND_CLEAR(indexBuffer_);
    FNET_DEL_AND_CLEAR(vertexArray_);
    FNET_DEL_AND_CLEAR(texUnitBuffer_);
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    DeepLayerBase::cleanup();
}



/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> Shallow2DeepLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    if (inputChannels_ <= PIXEL_PACKING) {
        // NOTE (mw) technically this type of layer is not needed for this case, as deep format
        // and shallow format are the same here, exception would be if padding has to be added
        // for input textures, we support textures with less than 4 channels (might be from upload)
        auto format = BufferSpec::formatByChannels(inputChannels_, TEXTURE_TYPE_DEFAULT);
        result.emplace_back(channel++, 0, width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                            format.first, format.second, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_SOURCE, inputChannels_);
    } else {
        int rem = inputChannels_;
        while (rem > 0) {
            result.emplace_back(channel++, 0,
                                width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                                TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE, std::min(rem, PIXEL_PACKING));
            rem -= PIXEL_PACKING;
        }
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> Shallow2DeepLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    // NOTE (mw) function type is not really correct here
    result.emplace_back(0, 0, viewport_[0], viewport_[1],
                        TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                        BufferSpec::FUNCTION_DEST);
    return result;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to vertex array object that encompasses the polygons
 *
 * @pre The supplied \p vao must be bound already
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up a tensor. This function sets up
 * those proxy polygons.
 */
void Shallow2DeepLayer::setupNetworkPolygons(VAO *vao) {
    float inputquad[4*2];
    inputquad[0*2+0] = (float)inputPadding_/(float)(width_+2*inputPadding_);
    inputquad[0*2+1] = (float)inputPadding_/(float)(height_+2*inputPadding_);
    inputquad[1*2+0] = inputquad[0*2+0];
    inputquad[1*2+1] = inputquad[0*2+1]+(float)height_/(float)(height_+2*inputPadding_);
    inputquad[2*2+0] = inputquad[0*2+0]+(float)width_/(float)(width_+2*inputPadding_);
    inputquad[2*2+1] = inputquad[1*2+1];
    inputquad[3*2+0] = inputquad[2*2+0];
    inputquad[3*2+1] = inputquad[0*2+1];
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    std::vector<deep::DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    int offset0=0;
    for (deep::DeepTiler::Tile tile : otiles) {
        tile.toFloatVec(attrs0,offset0,4);
        for (int i=0;i<4;i++) {
            attrs0[offset0+i*4+2] = inputquad[i*2];
            attrs0[offset0+i*4+3] = inputquad[i*2+1];
        }
        offset0+=4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0,tiler_->numOutputTiles()*4*4*sizeof(float),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //
    int *attrs1 = new int[tiler_->numOutputTiles()*4];
    int offset1=0;
    int texunit=0;
    for (int i=0;i<tiler_->numOutputTiles();i++) {
        for (int j=0;j<4;j++) attrs1[offset1++]=texunit;
        if (++texunit >= maxInputTextures_) texunit=0;
    }
    texUnitBuffer_ = new VBO(context_);
    vao->enableArray(1);
    texUnitBuffer_->setBufferData(attrs1, (int)(tiler_->numOutputTiles() * 4 * sizeof(int)), GL_STATIC_DRAW);
    texUnitBuffer_->bind();
    vao->setVertexAttributeBuffer(1,1,GL_INT,0,0);
    delete [] attrs1;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0;i<tiler_->numOutputTiles();i++) {
        int offset=i*4;
        indices[i*6+0]= (GLshort)(offset+0);
        indices[i*6+1]= (GLshort)(offset+1);
        indices[i*6+2]= (GLshort)(offset+2);
        indices[i*6+3]= (GLshort)(offset+0);
        indices[i*6+4]= (GLshort)(offset+2);
        indices[i*6+5]= (GLshort)(offset+3);
    }
    indexBuffer_->setBufferData(indices, (int)(6 * tiler_->numOutputTiles() * sizeof(GLshort)), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void Shallow2DeepLayer::setupShaders() {
    char preproc[1024] = {0};
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/shallow2deep.vert","shaders/shallow2deep.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->bindAttributeLocation("attributes1",1);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    for (int i=0;i<maxInputTextures_;i++) {
        char lname[40];
        snprintf(lname,sizeof(lname),"inputLayer%d",i);
        shaderState_->setUniformValue(lname,i);
    }
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
