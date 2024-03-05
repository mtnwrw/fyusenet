//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep to Shallow Layer Converter
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/miscdefs.h"
#include "deep2shallow.h"
#include "../gl/vbo.h"
#include "../gl/ibo.h"
#include "deep/deeptiler.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&,int)
 */
Deep2ShallowLayer::Deep2ShallowLayer(const GPULayerBuilder & builder, int layerNumber) :
    deep::DeepLayerBase(builder, layerNumber) {
    maxRenderTargets_ = GLInfo::getMaximumDrawBuffers();
    if (maxRenderTargets_ > FBO::MAX_DRAWBUFFERS) maxRenderTargets_=FBO::MAX_DRAWBUFFERS;
    viewport_[0] = width_ + 2 * outputPadding_;
    viewport_[1] = height_ + 2 * outputPadding_;
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
void Deep2ShallowLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void Deep2ShallowLayer::cleanup() {
    FNET_DEL_AND_CLEAR(posBuffer_);
    FNET_DEL_AND_CLEAR(attr0Buffer_);
    FNET_DEL_AND_CLEAR(attr1Buffer_);
    FNET_DEL_AND_CLEAR(attr2Buffer_);
    FNET_DEL_AND_CLEAR(attr3Buffer_);
    FNET_DEL_AND_CLEAR(indexBuffer_);
    FNET_DEL_AND_CLEAR(vertexArray_);
    shader_.reset();
    DeepLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::forward
 */
void Deep2ShallowLayer::forward(uint64_t sequenceNo, StateToken * state) {
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glViewport(0,0,viewport_[0],viewport_[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    shader_->bind();
    vertexArray_->bind();
    for (int pass=0; pass < (int)MRT_.size(); pass++) {
        framebuffers_.at(pass)->bind();
        framebuffers_.at(pass)->setWriteMask();
        glClear(GL_COLOR_BUFFER_BIT);
        shader_->setMappedUniformValue(UNIFORM_MRT,MRT_.at(pass));
        glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)(pass*6*sizeof(short)));
        framebuffers_.at(pass)->unbind();
    }
    shader_->unbind();
    vertexArray_->unbind();
}



/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> Deep2ShallowLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    int channel = 0;
    for (int i=0; i < outputChannels_; i += PIXEL_PACKING) {
        result.emplace_back(channel++, 0, viewport_[0], viewport_[1],
                            TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::FUNCTION_DEST, std::min(outputChannels_-i, PIXEL_PACKING));
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> Deep2ShallowLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.emplace_back(0, 0, tiler_->getInputTextureWidth(), tiler_->getInputTextureHeight(),
                        TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                        BufferSpec::FUNCTION_SOURCE);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        result.emplace_back(0, 1, residualViewport_[0], residualViewport_[1],
                            TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                            BufferSpec::RESIDUAL_SOURCE);
    }
    return result;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


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
void Deep2ShallowLayer::setupNetworkPolygons(VAO *vao) {
    float postemplate[4*2];
    float *posattrs = new float[tiler_->numInputTiles()*2*4];
    float *attr0 = new float[tiler_->numInputTiles()*4*4];
    float *attr1 = new float[tiler_->numInputTiles()*4*4];
    float *attr2 = new float[tiler_->numInputTiles()*4*4];
    float *attr3 = new float[tiler_->numInputTiles()*4*4];
    memset(posattrs,0,tiler_->numInputTiles()*2*4*sizeof(float));
    memset(attr0,0,tiler_->numInputTiles()*4*4*sizeof(float));
    memset(attr1,0,tiler_->numInputTiles()*4*4*sizeof(float));
    memset(attr2,0,tiler_->numInputTiles()*4*4*sizeof(float));
    memset(attr3,0,tiler_->numInputTiles()*4*4*sizeof(float));
    postemplate[0*2+0] = 2.0f*((float)outputPadding_)/((float)(width_ + 2*outputPadding_))-1.0f;
    postemplate[0*2+1] = 2.0f*((float)outputPadding_)/((float)(height_ + 2*outputPadding_))-1.0f;
    postemplate[1*2+0] = postemplate[0*2+0];
    postemplate[1*2+1] = postemplate[0*2+1]+2.0f*(float)height_ / (float)(height_+2*outputPadding_);
    postemplate[2*2+0] = postemplate[0*2+0]+2.0f*(float)width_ / (float)(width_+2*outputPadding_);
    postemplate[2*2+1] = postemplate[1*2+1];
    postemplate[3*2+0] = postemplate[2*2+0];
    postemplate[3*2+1] = postemplate[0*2+1];
    int posoffset=0, texoffset = 0;
    std::vector<deep::DeepTiler::Tile> tiles = tiler_->createInputTiles(0,0);
    int tileoffset=0,quads=0;
    while (tileoffset < (int)tiles.size()) {
        for (int j=0; j < 4*2; j++) {
            posattrs[j+posoffset] = postemplate[j];
        }
        posoffset += 4*2;
        for (int rt=0; rt < maxRenderTargets_; rt++) {
            deep::DeepTiler::Tile tile = tiles.at(tileoffset++);
            switch (rt) {
            case 0:
                tile.toFloatVec(attr0,texoffset,4);
                break;
            case 1:
                tile.toFloatVec(attr0,texoffset+2,4);
                break;
            case 2:
                tile.toFloatVec(attr1,texoffset,4);
                break;
            case 3:
                tile.toFloatVec(attr1,texoffset+2,4);
                break;
            case 4:
                tile.toFloatVec(attr2,texoffset,4);
                break;
            case 5:
                tile.toFloatVec(attr2,texoffset+2,4);
                break;
            case 6:
                tile.toFloatVec(attr3,texoffset,4);
                break;
            case 7:
                tile.toFloatVec(attr3,texoffset+2,4);
                break;
            default:
                THROW_EXCEPTION_ARGS(FynException,"Unsupported number of render targets");
            }
            if (tileoffset >= (int)tiles.size()) {
                MRT_.push_back(rt+1);
                break;
            }
            if (rt == maxRenderTargets_-1) {
                MRT_.push_back(maxRenderTargets_);
            }
        }
        quads++;
        texoffset+=4*4;
    }
    posBuffer_ = new VBO(context_);
    attr0Buffer_ = new VBO(context_);
    attr1Buffer_ = new VBO(context_);
    attr2Buffer_ = new VBO(context_);
    attr3Buffer_ = new VBO(context_);
    vao->enableArray(0);
    vao->enableArray(1);
    posBuffer_->setBufferData(posattrs, (int)(quads*2*4*sizeof(float)), GL_STATIC_DRAW);
    posBuffer_->bind();
    vao->setVertexAttributeBuffer(0,2,GL_FLOAT,GL_FALSE,0,0);
    attr0Buffer_->setBufferData(attr0, (int)(texoffset*sizeof(float)), GL_STATIC_DRAW);
    attr0Buffer_->bind();
    vao->setVertexAttributeBuffer(1,4,GL_FLOAT,GL_FALSE,0,0);
    if (maxRenderTargets_ > 2) {
        vao->enableArray(2);
        attr1Buffer_->setBufferData(attr1, (int)(texoffset*sizeof(float)), GL_STATIC_DRAW);
        attr1Buffer_->bind();
        vao->setVertexAttributeBuffer(2,4,GL_FLOAT,GL_FALSE,0,0);
    }
    if (maxRenderTargets_ > 4) {
        vao->enableArray(3);
        attr2Buffer_->setBufferData(attr2, (int)(texoffset*sizeof(float)), GL_STATIC_DRAW);
        attr2Buffer_->bind();
        vao->setVertexAttributeBuffer(3,4,GL_FLOAT,GL_FALSE,0,0);
    }
    if (maxRenderTargets_ > 6) {
        vao->enableArray(4);
        attr3Buffer_->setBufferData(attr3, (int)(texoffset*sizeof(float)), GL_STATIC_DRAW);
        attr3Buffer_->bind();
        vao->setVertexAttributeBuffer(4,4,GL_FLOAT,GL_FALSE,0,0);
    }
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[quads*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i < quads; i++) {
        int offset = i*4;
        indices[i*6+0] = (GLshort)(offset+0);
        indices[i*6+1] = (GLshort)(offset+1);
        indices[i*6+2] = (GLshort)(offset+2);
        indices[i*6+3] = (GLshort)(offset+0);
        indices[i*6+4] = (GLshort)(offset+2);
        indices[i*6+5] = (GLshort)(offset+3);
    }
    indexBuffer_->setBufferData(indices, (int)(6*quads*sizeof(GLshort)), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] attr3;
    delete [] attr2;
    delete [] attr1;
    delete [] attr0;
    delete [] posattrs;
    delete [] indices;
}


/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 */
void Deep2ShallowLayer::setupShaders() {
    char preproc[1024] = {0};
    snprintf(preproc, sizeof(preproc), "#define NUM_MRT %d\n", maxRenderTargets_);
    preprocessor_.activationFunctionPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
    shader_  = compileShaderPair("shaders/deep2shallow.vert","shaders/deep2shallow.frag", preproc, typeid(this));
    try {
        shader_->bindAttributeLocation("posAttributes",0);
        shader_->bindAttributeLocation("attributes0",1);
        if (maxRenderTargets_ >= 2) {
            shader_->bindAttributeLocation("attributes1",2);
        }
        if (maxRenderTargets_ >= 4) {
            shader_->bindAttributeLocation("attributes2",3);
        }
        if (maxRenderTargets_ >= 6) {
            shader_->bindAttributeLocation("attributes3",4);
        }
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shader_->bind();
    shader_->setUniformValue("inputLayer0", 0);
    shader_->mapUniformLocation("useMRT", UNIFORM_MRT);
    shader_->unbind();
}



/**
 * @copydoc GPULayerBase::setupFBOs
 */
void Deep2ShallowLayer::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    int texoffset = 0;
    int rem = (int)outputTextures_.size();
    while (rem > 0) {
        FBO * fbo = new FBO(context_,viewport_[0],viewport_[1],outputTextures_.at(texoffset++));
        rem--;
        if (rem > 0) {
            for (int i=1; i < maxRenderTargets_; i++) {
                fbo->addTexture(GL_COLOR_ATTACHMENT0+i,outputTextures_.at(texoffset++));
                rem--;
                if (rem <= 0) break;
            }
        }
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
