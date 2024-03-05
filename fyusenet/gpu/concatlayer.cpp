//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Concatenation Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/glexception.h"
#include "../gl/glinfo.h"
#include "gfxcontextlink.h"

#include "concatlayer.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
ConcatLayer::ConcatLayer(const ConcatLayerBuilder & builder, int layerNumber) : GPULayerBase((const GPULayerBuilder &)builder, layerNumber) {
    for (int i=0;i<3*4;i++) concatShaders_[i]=nullptr;
    currentInputChannels_ = 0;
    consolidationRender_ = (inputPadding_ != outputPadding_);
    viewport_[0] = width_ + 2*outputPadding_;
    viewport_[1] = height_ + 2*outputPadding_;
    // TODO (mw) support individual activation types on different concatenation inputs
    int relucnt = 0;
    for (auto it = builder.inputs_.begin(); it != builder.inputs_.end(); ++it) {
        addInput((*it).channels,(*it).padding);
        if ((*it).flags & LayerFlags::PRE_RELU) relucnt++;
    }
    if (relucnt == (int)builder.inputs_.size()) flags_ |= LayerFlags::PRE_RELU;
    else if (relucnt > 0) FNLOGE("WARNING: reLU/non-reLU concats are not supported yet");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void ConcatLayer::cleanup() {
    delete vertexBuffer_;
    delete indexBuffer_;
    delete vertexArray_;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    // reset shaders here because the GL context is bound here (in case no cache is used)
    defaultShader_.reset();
    for (int i=0; i < 3*4; i++) concatShaders_[i].reset();
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
void ConcatLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @brief Add input tensor (channel) shape / padding for concatenation
 *
 * @param inputChannels Number of input channels for the input tensor to add
 * @param inputPadding Padding for the input tensor
 *
 * @throws FynException in case of invalid/incompatible data
 *
 * @note Currently this layer does not support "mixed" padding on input tensors, this means that
 *       all input tensors must have the same padding.
 */
void ConcatLayer::addInput(int inputChannels, int inputPadding) {
    if (inputChannels < 3) THROW_EXCEPTION_ARGS(FynException,"Input depth < 3 currently not supported");
    if (inputPadding != inputPadding_) THROW_EXCEPTION_ARGS(FynException,"Mismatch on input padding (%d vs %d)",inputPadding_,inputPadding);
    portChannels_.push_back(inputChannels);
    if (portOffsets_.empty()) portOffsets_.push_back(0);
    else {
        size_t last = portOffsets_.size()-1;
        int buffers = portChannels_.at(last);
        buffers = (buffers  / PIXEL_PACKING) + (((buffers % PIXEL_PACKING)==0) ? 0 : 1);
        portOffsets_.push_back(portOffsets_.at(last)+buffers);
    }
    if (currentInputChannels_ & 3) consolidationRender_ = true;
    currentInputChannels_ += inputChannels;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> ConcatLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    if (portChannels_.empty()) THROW_EXCEPTION_ARGS(FynException,"No inputs allocated, please use addInput()");
    int connindex=0;
    for (auto init = portChannels_.begin(); init != portChannels_.end(); ++init) {
        int channel=0;
        int rem = *init;
        while (rem > 0) {
            result.push_back(BufferSpec(channel++, connindex, width_ + 2*inputPadding_,
                                        height_ + 2*inputPadding_,
                                        TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                        BufferSpec::CONCAT_SOURCE, std::min(rem, PIXEL_PACKING)).interpolation(BufferSpec::interp::ANY));
            rem -= PIXEL_PACKING;
        }
        connindex++;
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> ConcatLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    int channel=0;
    int rem = outputChannels_;
    while (rem > 0) {
        result.push_back(BufferSpec(channel++, 0, viewport_[0], viewport_[1],
                                    TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::CONCAT_DEST, std::min(rem, PIXEL_PACKING)).passThrough(!consolidationRender_).interpolation(BufferSpec::interp::ANY));
        rem -= PIXEL_PACKING;
    }
    return result;
}


/**
 * @copydoc LayerBase::forward
 */
void ConcatLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if ((consolidationRender_)||(outputChanged_)) {
        if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
        int err = glGetError();
        if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
        if (outputChanged_) updateFBOs();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glDepthFunc(GL_GEQUAL);
        glDepthMask(GL_FALSE);
        glViewport(0,0,viewport_[0],viewport_[1]);
        vertexArray_->bind();
        int blockoffset=0;
        int layeroffset=0;
        int shift=0;
        int trail=PIXEL_PACKING;
        int rem = portChannels_.at(blockoffset);
        ShaderProgram * currentshader = defaultShader_.get();
        currentshader->bind(defaultShaderState_.get());
        glClearColor(0.0f,0.f,0.0f,0.0f);
        for (int outpass=0; outpass < (int)framebuffers_.size(); outpass++) {
            framebuffers_.at(outpass)->bind();
            framebuffers_.at(outpass)->setWriteMask();
            glClear(GL_COLOR_BUFFER_BIT);
            if ((rem < trail) && (outpass < (int)framebuffers_.size()-1)) {
                trail = rem;
                blockoffset++;
                rem=portChannels_.at(blockoffset)-(PIXEL_PACKING-trail);
            } else {
                rem -= PIXEL_PACKING;
                if ((rem<=0) && (outpass < (int)framebuffers_.size()-1)) {
                    blockoffset++;
                    shift = 0;
                    trail = PIXEL_PACKING;
                    rem = portChannels_.at(blockoffset);
                }
            }
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,inputTextures_.at(layeroffset++));
            if ((shift > 0) || (trail < PIXEL_PACKING)) {
                int shader = (trail-1)+3*shift;
                glActiveTexture(GL_TEXTURE1);
                if (rem > 0) glBindTexture(GL_TEXTURE_2D,inputTextures_.at(layeroffset));
                else glBindTexture(GL_TEXTURE_2D,inputTextures_.at(layeroffset-1));      // TODO (mw) actually bind a zero-texture here to be clean/r
                if (concatShaders_[shader].get() != currentshader) {
                    currentshader->unbind(true);
                    currentshader = concatShaders_[shader].get();
                    currentshader->bind(concatShaderStates_[shader].get());
                }
            } else {
                // shift==0
                if (currentshader != defaultShader_.get()) {
                    currentshader->unbind(true);
                    currentshader = defaultShader_.get();
                    currentshader->bind(defaultShaderState_.get());
                }
            }
            shift = PIXEL_PACKING - trail;
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
            framebuffers_.at(outpass)->unbind();
        }
        vertexArray_->unbind();
        if (currentshader) currentshader->unbind();
    }
}


/**
 * @copydoc GPULayerBase::addInputTexture
 */
void ConcatLayer::addInputTexture(GLuint textureID, int channelIndex) {
    GPULayerBase::addInputTexture(textureID,channelIndex);
    // if we are lucky, we do not need to render at all, in this case the output is the
    // same as the input
    if (!consolidationRender_) addOutputTexture(textureID,channelIndex, 0);
}


/**
 * @copydoc LayerBase::numInputPorts
 */
int ConcatLayer::numInputPorts() const {
    return (int)portChannels_.size();
}


/**
 * @copydoc LayerBase::getPortChannelIndex
 */
int ConcatLayer::getPortChannelIndex(int port) const {
    if (port >= numInputPorts()) THROW_EXCEPTION_ARGS(FynException,"Illegal input port %d specified",port);
    return portOffsets_.at(port);
}

/**
 * @copydoc LayerBase::numInputChannels
 */
int ConcatLayer::numInputChannels(int port) const {
    if (port >= numInputPorts()) THROW_EXCEPTION_ARGS(FynException,"Illegal input port %d specified",port);
    return portChannels_.at(port);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to vertex array object that maintains the geometry information
 *
 * @pre The supplied \p vao is already bound
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up a tensor. This function creates those
 * proxy polygons with their corresponding texture information.
 */
void ConcatLayer::setupNetworkPolygons(VAO *vao) {
    int vertsize = 4;
    float * attrs0 = new float[vertsize*4];
    float posleft  = -1.0f + ((float)(2*outputPadding_) / (float)viewport_[0]);
    float posright =  1.0f - ((float)(2*outputPadding_) / (float)viewport_[0]);
    float postop  =  -1.0f + ((float)(2*outputPadding_) / (float)viewport_[1]);
    float posbottom = 1.0f - ((float)(2*outputPadding_) / (float)viewport_[1]);
    float thspan=(float)(width_) / (float)(width_+2*inputPadding_);
    float tvspan=(float)(height_) / (float)(height_+2*inputPadding_);
    float tleft = (float)inputPadding_ / (float)(width_+2*inputPadding_);
    float ttop = (float)inputPadding_ / (float)(height_+2*inputPadding_);
    //-----------------------------------------------------
    // Positions first (output layers)
    //-----------------------------------------------------
    attrs0[0*vertsize+0] = posleft;
    attrs0[0*vertsize+1] = postop;
    attrs0[1*vertsize+0] = posleft;
    attrs0[1*vertsize+1] = posbottom;
    attrs0[2*vertsize+0] = posright;
    attrs0[2*vertsize+1] = posbottom;
    attrs0[3*vertsize+0] = posright;
    attrs0[3*vertsize+1] = postop;
    //-----------------------------------------------------
    // Texture offsets (input layers)
    //-----------------------------------------------------
    attrs0[0*vertsize+2] = tleft;
    attrs0[0*vertsize+3] = ttop;
    attrs0[1*vertsize+2] = tleft;
    attrs0[1*vertsize+3] = ttop+tvspan;
    attrs0[2*vertsize+2] = tleft+thspan;
    attrs0[2*vertsize+3] = ttop+tvspan;
    attrs0[3*vertsize+2] = tleft+thspan;
    attrs0[3*vertsize+3] = ttop;
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0,vertsize*4*sizeof(float),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,vertsize,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    // TODO (mw) this is a very often occuring pattern, create a global instance and reuse it
    // instead, reduces the number of handles floating around
    GLshort indices[6] = {0,1,2,0,2,3};
    indexBuffer_ = new IBO(context_);
    indexBuffer_->setBufferData(indices, 6*sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
}



/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 *
 * The following (shader) preprocessor macros are defined:
 *   - \c NUM_LANES
 *   - \c SHIFT
 *   - \c TRAIL
 *
 * @see compileShaderPair
 */
void ConcatLayer::setupShaders() {
    char preproc[1024] = {0};
    // TODO (mw) add support for activations here too
    snprintf(preproc, sizeof(preproc), "#define SHIFT 0\n#define TRAIL 4\n#define NUM_LANES 1\n");
    defaultShader_ = compileShaderPair("shaders/vanilla/concat.vert","shaders/vanilla/concat.frag",preproc,typeid(this));
    try {
        defaultShader_->bindAttributeLocation("attributes0",0);
        defaultShader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    defaultShaderState_ = UniformState::makeShared(defaultShader_);
    defaultShaderState_->setUniformValue("inputLayer0",0);
    int offset=0;
    for (int shift=0; shift < PIXEL_PACKING; shift++) {
        for (int trail=1; trail < PIXEL_PACKING; trail++) {
            // TODO (mw) add support for activations here too
            snprintf(preproc, sizeof(preproc), "#define SHIFT %d\n#define TRAIL %d\n#define NUM_LANES 1\n",shift,trail);
            concatShaders_[offset]=compileShaderPair("shaders/vanilla/concat.vert","shaders/vanilla/concat.frag",preproc,typeid(this));
            try {
                concatShaders_[offset]->bindAttributeLocation("attributes0",0);
                concatShaders_[offset]->link();
            } catch (GLException& ex) {
                FNLOGE("Cannot link shader for layer %s",getName().c_str());
                throw;
            }
            concatShaderStates_[offset] = UniformState::makeShared(concatShaders_[offset]);
            concatShaderStates_[offset]->setUniformValue("inputLayer0",0);
            concatShaderStates_[offset]->setUniformValue("inputLayer1",1);
            offset++;
        }
    }
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void ConcatLayer::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    for (int texoffset=0; texoffset < (int)outputTextures_.size(); texoffset++) {
        FBO * fbo = new FBO(context_,viewport_[0],viewport_[1],outputTextures_.at(texoffset));
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void ConcatLayer::updateFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in convlayer %s",getName().c_str());
    for (int texoffset=0; texoffset < (int)outputTextures_.size();texoffset++) {
        FBO * fbo = framebuffers_.at(texoffset);
        fbo->bind();
        fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0,outputTextures_.at(texoffset));
        fbo->unbind();
    }
    outputChanged_ = false;
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
