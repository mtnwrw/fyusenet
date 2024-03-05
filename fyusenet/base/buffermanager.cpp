//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Buffer Manager (GPU and CPU Buffers)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/miscdefs.h"
#include "../common/logging.h"
#include "asynclayerinterface.h"
#include "../gl/glexception.h"
#include "buffermanager.h"

//-------------------------------------- Global Variables ------------------------------------------


namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Boring constructor, idle
 */
BufferManager::BufferManager(const GfxContextLink& ctx) : GfxContextTracker() {
    setContext(ctx);
}



/**
 * @brief Destructor
 */
BufferManager::~BufferManager() {
    if (!texturePool_.empty()) {
        FNLOGE("Cleanup not called, GL memory leakage");
        assert(false);
    }
    texturePool_.clear();
    bufferPool_.clear();  // to make sure, should have been done in cleanup() already
}


/**
 * @brief Release resources (GL and non-GL) held by this manager instance
 *
 * @pre When OpenGL is used, the OpenGL context that was used to create the buffers/textures must be
 *      current to the calling thread
 */
void BufferManager::cleanup() {
    if (!texturePool_.empty()) {
        std::unique_ptr<GLuint[]> textures(new GLuint[texturePool_.size()]);
        int pi=0;
        for (auto ti = texturePool_.begin(); ti != texturePool_.end(); ++ti,pi++) {
            textures[pi]=(*ti).id_;
        }
        glDeleteTextures((int)texturePool_.size(),textures.get());
        texturePool_.clear();
    }
    bufferPool_.clear();
    estimatedTextureBytes_ = 0;
}


/**
 * @brief Create a CPU buffer and assign it as output buffer to a layer object
 *
 * @param outputLayer Pointer to layer which should be assigned output buffer(s)
 *
 * @param lock Prevent re-use of the created buffer when buffer pooling is used
 *
 * This function creates one or more CPU buffers and assigns them to \e all outputs of the
 * provided \p outputLayer.
 */
void BufferManager::createCPUOutput(LayerBase *outputLayer, bool lock) {
    // TODO (mw) support multiple output ports
    auto * cpuout = dynamic_cast<cpu::CPULayerInterface *>(outputLayer);
    if (!cpuout) {
        THROW_EXCEPTION_ARGS(FynException, "Cannot assign CPU output to class that does not implement CPU interface");
    }
    const std::vector<BufferSpec>& outputs = outputLayer->getRequiredOutputBuffers();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
        if (!cpuout->getCPUOutputBuffer((*it).port_)) {
            Buffer buf = createBuffer((*it).width_, (*it).height_, (*it).channels_, (*it).internalFormat_, (*it).type_,  (*it).dataOrder_);
            buf.locked_ = lock;
            cpuout->addCPUOutputBuffer(buf.buf_);
            outputLayer->addOutputConnection(0, nullptr, 0);
            bufferPool_.push_back(buf);
        }
    }
}



/**
 * @brief Create a (set of) GPU output texture(s) and assign it to the outputs of a layer
 *
 * @param outputLayer Pointer to layer which should be assigned the GPU output
 *
 * @param internalFormat Sized (internal) texture format for the output texture (e.g. \c GL_RGBA32F)
 *
 * @param pixelFormat Pixel format for the output texture (e.g. \c GL_RGBA )
 *
 * @param dataType GL datatype for the output texture (e.g. \c GL_FLOAT )
 *
 * This function designates the layer as a sink and adds a (set of) output texture(s) to this layer,
 * which is not connected to any other layer in the network and also not shared with any other layer
 * in the network.
 */
void BufferManager::createGPUOutput(gpu::GPULayerBase *outputLayer,
                                    BufferSpec::sizedformat internalFormat,
                                    BufferSpec::genericformat pixelFormat,
                                    BufferSpec::dtype dataType) {

    const std::vector<BufferSpec>& outputs = outputLayer->getRequiredOutputBuffers();
    for (auto texit = outputs.begin() ; texit != outputs.end(); ++texit) {
        Texture ot = createTexture((*texit).width_, (*texit).height_,
                                   internalFormat, pixelFormat, dataType, BufferSpec::interp::LINEAR);
        auto * gpu = dynamic_cast<gpu::GPULayerBase *>(outputLayer);
        if (!gpu) THROW_EXCEPTION_ARGS(FynException,"Cannot assign output texture to non-GPU layer");
        gpu->addOutputTexture(ot.id_, (*texit).channelIndex_, 0);
        gpu->addOutputConnection(0, nullptr, 0);
        texturePool_.push_back(ot);
    }
}



/**
 * @brief Connect the output of a layer to the input of the next layer
 *
 * @param outputLayer Pointer to network layer to use the output textures from
 *
 * @param inputLayer Pointer to network layer which is supposed to process the output textures
 *
 * @param port Port to connect on the input side, default is 0
 *
 * @param lock Flag that controls whether a connected input texture should be marked as
 *             locked after connecting it. Locked textures are exempt from later re-use.
 *
 * @throws FynException in case no connection could be established
 *
 * This function connects the output of the \p outputLayer to the selected input \p port of the
 * \p inputLayer. Prior to establish the connection, the output data and the input port are checked
 * for compatibility and only then a connection is established. For input layers that have more than
 * one port, all ports have to be connected individually.
 *
 * @see checkIOMatch()
 *
 * @note This function is \b not reentrant.
 */
void BufferManager::connectLayers(LayerBase *outputLayer, LayerBase *inputLayer, int port, bool lock) {
    std::vector<std::pair<BufferSpec,BufferSpec>> matches;
    if ((!outputLayer) || (!inputLayer)) {
        THROW_EXCEPTION_ARGS(FynException,"Illegal parameters out=%p in=%p",outputLayer, inputLayer);
    }
    const std::vector<BufferSpec> inputs = inputLayer->getRequiredInputBuffers();
    const std::vector<BufferSpec> outputs = outputLayer->getRequiredOutputBuffers();
    if (inputs.empty()) {
        THROW_EXCEPTION_ARGS(FynException,"Input layer %s has no inputs",inputLayer->getName().c_str());
    }
    if (outputs.empty()) {
        THROW_EXCEPTION_ARGS(FynException,"Input layer %s has no inputs",outputLayer->getName().c_str());
    }
    matches = checkIOMatch(inputLayer, inputs, outputs, port);
    if (matches.empty()) {
        THROW_EXCEPTION_ARGS(FynException,"Inputs/outputs do not match (I/O) for layers %s and %s",inputLayer->getName().c_str(), outputLayer->getName().c_str());
    }
    if (matches.at(0).first.device_ == BufferSpec::csdevice::COMP_STOR_GPU) {
        auto * outlayer = dynamic_cast<gpu::GPULayerBase *>(outputLayer);
        auto * inlayer = dynamic_cast<gpu::GPULayerBase *>(inputLayer);
        connectGPULayers(outlayer, inlayer, matches, port, lock);
    } else {
        connectCPULayers(outputLayer, inputLayer, matches, port, lock);
    }
}


/**
 * @brief Get size of memory required for the buffer
 *
 * @return Size (in bytes) for the buffer
 */
size_t BufferManager::Buffer::size() const {
    int bytesperchan = 1;
    int baseunit = 1;
    // NOTE (mw) ugly, make nicer when time
    switch ((GLint)internalFormat_) {
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            baseunit = 4;
            bytesperchan = 4;
            break;
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            baseunit = 3;
            bytesperchan = 4;
            break;
        case GL_R32F:
            bytesperchan = 4;
            break;
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            baseunit = 4;
            bytesperchan = 2;
            break;
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
            baseunit = 3;
            bytesperchan = 2;
            break;
        case GL_R16I:
        case GL_R16F:
            bytesperchan = 2;
            break;
        case GL_RGBA:
        case GL_RGBA8:
            baseunit = 4;
            break;
        case GL_RGB:
        case GL_RGB8:
            baseunit = 3;
            break;
        case GL_R8:
            break;
        default:
            THROW_EXCEPTION_ARGS(FynException, "Unsupported internal format");
    }
    int echan = baseunit * ((channels_ + baseunit-1) / baseunit);
    return width_ * height_ * echan * bytesperchan;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Internal helper function to connect two CPU layers
 *
 * @param outLayer Pointer to network layer to use the output textures from
 *
 * @param inLayer Pointer to network layer which is supposed to process the output textures
 *
 * @param matches List of pair-wise matches (usually only contains one element)
 *
 * @param port Port on the \p inLayer to connect the \p outLayer output to
 *
 * @param lock Flag that controls whether a connected input buffer should be marked as
 *             locked after connecting it. Locked buffers are exempt from later re-use.
 *
 * @throws FynException on detected errors
 *
 * This function connects the output of the \p outLayer to the specified \p port of the \p inLayer,
 * given that both layers are CPU-based, more precisely the output of the \p outLayer must not be
 * a texture and the input of the \p inLayer must not be a texture. Connectivity is established
 * by means of buffers (used as representation of "tensors" in ML lingo) which are supposed to
 * be written to by the \p outLayer and read from by the \p inLayer. These buffers are taken from
 * a buffer pool and - if not locked - will be re-used for different layers going forward.
 * It may make sense to lock buffers against re-use, i.e. only use a buffer for one pair of layers.
 * For this case, the \p lock parameter can be used, which prevents a buffer from being re-used
 * after this call. Note that this does not preclude the buffer from having been used prior to
 * the locking.
 */
void BufferManager::connectCPULayers(LayerBase *outLayer, LayerBase *inLayer, std::vector<std::pair<BufferSpec,BufferSpec>> & matches, int port, bool lock) {
    auto * cpuout = dynamic_cast<cpu::CPULayerInterface *>(outLayer);
    auto * cpuin = dynamic_cast<cpu::CPULayerInterface *>(inLayer);
    if ((!cpuout) || (!cpuin)) THROW_EXCEPTION_ARGS(FynException, "Illegal layers supplied");
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        //---------------------------------------------------------
        // Check if the associated output already has a buffer and
        // use that for the input...
        //---------------------------------------------------------
        if (cpuout->hasCPUOutputBuffer(it->first.port_)) {
            CPUBuffer * outbuf = cpuout->getCPUOutputBuffer(it->first.port_);
            if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) {
                cpuin->setCPUResidualBuffer(outbuf);
            }
            else {
                cpuin->setCPUInputBuffer(outbuf, port);
            }
            inLayer->addInputConnection(port, outLayer, it->first.port_);
            outLayer->addOutputConnection(it->second.port_, inLayer, port);
            updateLayerUseByBuffer(outbuf, inLayer->getNumber(), lock);
        } else {
            //-------------------------------------------------------
            // Check if we can re-use an existing buffer
            //-------------------------------------------------------
            int index = findBuffer(inLayer->getNumber(), outLayer->getNumber(), it->second.width_, it->second.height_, it->second.channels_, it->second.internalFormat_);
            if (index >= 0) {
                CPUBuffer * buf = bufferPool_.at(index).buf_;
                if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) {
                    cpuin->setCPUResidualBuffer(buf);
                }
                else {
                    cpuin->setCPUInputBuffer(buf, port);
                }
                inLayer->addInputConnection(port, outLayer, it->first.port_);
                cpuout->addCPUOutputBuffer(buf);
                outLayer->addOutputConnection(it->second.port_, inLayer, port);
                updateLayerUse(index, inLayer->getNumber(), lock);
            } else {
                //-------------------------------------------------------
                // Create a new buffer...
                //-------------------------------------------------------
                Buffer nb = createBuffer(it->second.width_, it->second.height_, it->second.channels_, it->second.internalFormat_, it->second.type_);
                nb.lastInputLayer_ = inLayer->getNumber();
                nb.locked_ = lock;
                bufferPool_.push_back(nb);
                cpuout->addCPUOutputBuffer(nb.buf_);
                if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) {
                    cpuin->setCPUResidualBuffer(nb.buf_);
                }
                else {
                    cpuin->setCPUInputBuffer(nb.buf_, port);
                }
                inLayer->addInputConnection(port,outLayer, it->first.port_);
                outLayer->addOutputConnection(it->second.port_, inLayer, port);
            }
        }
    }
}


/**
 * @brief Internal helper function to connect two GPU layers
 *
 * @param outLayer Pointer to network layer to use the output textures from
 *
 * @param inLayer Pointer to network layer which is supposed to process the output textures
 *
 * @param matches List of pair-wise matches, the first part of each pair is the spec for the
 *                input (=receiving) layer and the second part of each pair is the spec for the
 *                output (=sending) layer
 *
 * @param port Port on the \p inLayer to connect the \p outLayer output to
 *
 * @param lock Flag that controls whether a connected input buffer should be marked as
 *             locked after connecting it. Locked buffers are exempt from later re-use.
 *
 * @throws FynException on detected errors
 *
 * This function connects the output of the \p outLayer to the specified \p port of the \p inLayer,
 * given that both layers are GPU-based, more precisely the outputs of the \p outLayer be textures
 * texture and the inputs of the \p inLayer must also be textures. As conveniently dimensionalized
 * textures are limited in storage, a single port may require more than one texture to represent
 * a single "tensor", where a set of textures is stacked to form a single tensor. This is reflected
 * in the \p matches parameter, which assigns multiple textures to the in/out ports using offsets
 * within each port.
 *
 * The textures themselves are taken from a texture pool and - if not locked - will be re-used for
 * different layers going forward. It may make sense to lock textures against re-use, i.e. only use
 * a set of textures for one pair of layers.
 * For this case, the \p lock parameter can be used, which prevents the textures used here from
 * being re-used after this call. Note that this does not preclude the textures from having been
 * used prior to the locking.
 */
void BufferManager::connectGPULayers(gpu::GPULayerBase * outLayer, gpu::GPULayerBase * inLayer, std::vector<std::pair<BufferSpec,BufferSpec>> & matches, int port, bool lock) {    
    auto * asy = dynamic_cast<AsyncLayer *>(outLayer);
    //---------------------------------------------------------
    // In case of asynchronous layers, add the receiving layer
    // as dependency. Multiple calls are OK, the layer itself
    // takes care of uniqueness...
    //---------------------------------------------------------
    if (asy && asy->isAsync()) {
        lock = true;  // asynchronous layers always have locked output textures
        asy->addAsyncDependency(inLayer, matches.begin()->first.channelIndex_);
    }
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        //---------------------------------------------------------
        // Check if the associated output already has a texture and
        // use that for the input...
        //---------------------------------------------------------
        if (outLayer->hasOutputTexture(it->first.channelIndex_)) {
            GLuint tid = outLayer->getOutputTexture(it->first.channelIndex_);
            if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) {
                inLayer->addResidualTexture(tid, it->first.channelIndex_);
            }
            else {
                inLayer->addInputTexture(tid, it->first.channelIndex_ + inLayer->getPortChannelIndex(port));
            }
            inLayer->addInputConnection(port, outLayer, it->first.port_);
            outLayer->addOutputConnection(it->second.port_, inLayer, port);
            updateLayerUseByTextureID(tid,inLayer->getNumber(),lock);
        } else {
            //-------------------------------------------------------
            // Check if the output is in pass-through mode, then
            // we use the input texture of the same port / cidx for
            // the output...
            // FIXME (mw) this does not handle multi-ported output
            // nor does it handle shadow textures (those are not paired
            // with pass-through textures anyway)...
            //-------------------------------------------------------
            if (it->second.passThrough_) {
                GLuint pttex = outLayer->getInputTexture(it->second.channelIndex_);
                outLayer->addOutputTexture(pttex, it->second.channelIndex_, 0);
                inLayer->addInputTexture(pttex, it->first.channelIndex_ + inLayer->getPortChannelIndex(it->first.port_));
                inLayer->addInputConnection(port, outLayer, it->first.port_);
                outLayer->addOutputConnection(it->second.port_, inLayer, port);
                int index = findTexture(pttex);
                assert(index >= 0);
                if (index >= 0) updateLayerUse(index, inLayer->getNumber(), lock);
            } else {
                //-------------------------------------------------------
                // Check if we can re-use an old texture...
                //-------------------------------------------------------
                int index = findTexture(inLayer->getNumber(), outLayer->getNumber(),
                                        it->second.width_, it->second.height_,
                                        it->second.internalFormat_, it->second.interpolation_);
                if ((index >= 0) && (!lock) && (!it->second.lock_)) {
                    GLuint tid = texturePool_.at(index).id_;
                    if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) inLayer->addResidualTexture(tid, it->first.channelIndex_);
                    else {
                        inLayer->addInputTexture(tid, it->first.channelIndex_ + inLayer->getPortChannelIndex(port));
                    }
                    inLayer->addInputConnection(port, outLayer, it->first.port_);
                    outLayer->addOutputTexture(tid,it->second.channelIndex_, 0);
                    outLayer->addOutputConnection(it->second.port_, inLayer, port);
                    updateLayerUse(index,inLayer->getNumber(),lock);
                } else {
                    //-------------------------------------------------------
                    // No re-use possible or desired, create a new texture...
                    //-------------------------------------------------------
                    Texture nt = createTexture(it->second.width_, it->second.height_, it->second.internalFormat_, it->second.format_, it->second.type_);
                    nt.lastInputLayer_ = inLayer->getNumber();
                    nt.locked_ = lock | it->second.lock_;
                    texturePool_.push_back(nt);
                    if (it->first.usage_ == BufferSpec::RESIDUAL_SOURCE) inLayer->addResidualTexture(nt.id_, it->first.channelIndex_);
                    else {
                        inLayer->addInputTexture(nt.id_, it->first.channelIndex_ + inLayer->getPortChannelIndex(port));
                    }
                    outLayer->addOutputTexture(nt.id_, it->second.channelIndex_, 0);
                    inLayer->addInputConnection(port, outLayer, it->first.port_);
                    outLayer->addOutputConnection(it->second.port_, inLayer, port);
                    //-------------------------------------------------------
                    // If this layer wants "shadow textures", add them...
                    //-------------------------------------------------------
                    if (it->second.multiplicity_ > 1) {
                        for (int m = 0; m < it->second.multiplicity_ - 1; m++) {
                            Texture snt = createTexture(it->second.width_, it->second.height_, it->second.internalFormat_, it->second.format_, it->second.type_);
                            snt.lastInputLayer_ = inLayer->getNumber();
                            snt.locked_ = true;
                            texturePool_.push_back(snt);
                            outLayer->addOutputTexture(snt.id_, it->second.channelIndex_, m + 1);
                        }
                    }
                }
            }
        }
    }
}



/**
 * @brief Match the outputs of a sending layer to the inputs of a receiving layer
 *
 * @param inputLayer Pointer to layer that receives the input
 * @param inputs Buffer specifications for the input ports of \p inputLayer
 * @param outputs Buffer specification for the output ports of the sending layer
 * @param inputPort Port number on the \p inputLayer to match
 *
 * @return List of BufferSpec pairs that match an output buffer to the corresponding input buffer
 *
 * This function creates a list of BufferSpec pairs which identify correspondences between output
 * buffers of the sending output layer and input buffers of the receiving input layer. Because a
 * single tensor can be represented by multiple buffers (for example when textures are used as
 * buffer realizations), there is no 1:1 correspondence between output ports and input ports,
 * hence the requirement to return a list of matches.
 *
 * It also performs slight adjustments to the texture formats (if required).
 */
std::vector<std::pair<BufferSpec,BufferSpec>> BufferManager::checkIOMatch(LayerBase *inputLayer, const std::vector<BufferSpec>& inputs,const std::vector<BufferSpec>& outputs, int inputPort) {
    std::vector<std::pair<BufferSpec,BufferSpec>> result;
    if (inputLayer->isConnected(inputPort)) {
        return result;
    }
    for (auto it=inputs.begin(); it != inputs.end(); ++it) {
        const BufferSpec& inspec = *it;
        if (inspec.port_ != inputPort) continue;
        // TODO (mw) add a more thorough input-type check
        for (auto ot=outputs.begin(); ot != outputs.end(); ++ot) {
            const BufferSpec& outspec = *ot;
            bool intermatch = (outspec.interpolation_ == inspec.interpolation_) ||
                              (((outspec.interpolation_ == BufferSpec::interp::ANY) ||
                              (inspec.interpolation_ == BufferSpec::interp::ANY)));
            bool devmatch = (outspec.device_ == inspec.device_);
            bool idxmatch = (inspec.channelIndex_ == outspec.channelIndex_);
            if (devmatch && idxmatch && (outspec.width_ == inspec.width_) && (outspec.height_ == inspec.height_) && intermatch) {
                if ((outspec.device_ == BufferSpec::csdevice::COMP_STOR_CPU) && (outspec.channels_ != inspec.channels_)) continue;
                if ((outspec.internalFormat_ != inspec.internalFormat_) && (outspec.usage_ != BufferSpec::OES_DEST) && (outspec.dataOrder_ == BufferSpec::order::GPU_SHALLOW)) {
                    if (BufferSpec::isIntegral(inspec.internalFormat_) == BufferSpec::isIntegral(outspec.internalFormat_)) {
                        // output dominates because some GL(ES) implementations cannot write to RGB textures
                        BufferSpec buf = *it;
                        buf.internalFormat_ = outspec.internalFormat_;
                        buf.format_ = outspec.format_;
                        result.emplace_back(buf, outspec);
                    }
                } else {
                    result.emplace_back(inspec, outspec);
                }
            }
        }
    }
    return result;
}



/**
 * @brief Find matching texture in internal texture pool
 *
 * @param inputLayer Layer number of the input layer
 * @param outputLayer Layer number of the output layer
 * @param width Requested texture width
 * @param height Requested texture height
 * @param internalFormat Sized OpenGL texture format (e.g. \c GL_RGBA8)
 * @param interpolation Interpolation mode for the texture (e.g. nearest neighbor or bilinear)
 *
 * @return Index into the texture pool that a matching texture was found at or -1 if none was found.
 *
 * This function tries to find a (usable) texture in the pool that meets the supplied specification.
 * Textures that are marked as locked or are still in use (given by the output layer number recorded
 * in the pool), will not be returned.
 */
 // TODO (mw) improve the linear access time to logarithmic
int BufferManager::findTexture(int inputLayer, int outputLayer, int width, int height,
                               BufferSpec::sizedformat internalFormat, BufferSpec::interp interpolation) const {
    assert(inputLayer > outputLayer);
    for (int i=0; i < (int)texturePool_.size(); i++) {
        const Texture & tx = texturePool_.at(i);
        if ((tx.width_ == width) && (tx.height_ == height) && (tx.internalFormat_ == internalFormat) &&
            ((interpolation == BufferSpec::interp::ANY) || (tx.interpolation_ == interpolation))) {
            // we cannot use something as input for layer N which already has been input to layer N-1 or >=N
            if ((!tx.locked_) && (tx.lastInputLayer_ < inputLayer-1) && (outputLayer > tx.lastInputLayer_)) {
                return i;
            }
        }
    }
    return -1;
}


/**
 * @brief Find an existing texture in the pool by its OpenGL handle
 *
 * @param handle Handle to look for
 *
 * @return Index of the texture in the pool or -1 if not found
 */
int BufferManager::findTexture(GLuint handle) const {
    for (int i=0; i < (int)texturePool_.size(); i++) {
        const Texture &tx = texturePool_.at(i);
        if (tx.id_ == handle) {
            return i;
        }
    }
    return -1;
}


/**
 * @brief Look for an existing buffer in the pool that can be used
 *
 * @param inputLayer Receiving layer of the connection
 * @param outputLayer Sending layer of the connection
 * @param width Width of buffer in pixels
 * @param height Height of buffer in pixels
 * @param channels # of channels for the buffer
 * @param internalFormat An OpenGL internal format that is used for buffer representation (also used for the CPU side)
 *
 * @return Index of a usable buffer in the #bufferPool_ or -1 if no suitable buffer was found
 *
 * This function tries to find a (usable) buffer in the pool that meets the supplied specification.
 * Buffers that are marked as locked or are still in use (given by the output layer number recorded
 * in the pool), will not be returned.
 */
int BufferManager::findBuffer(int inputLayer, int outputLayer, int width, int height, int channels,
                              BufferSpec::sizedformat internalFormat) const {
    Buffer wanted(width, height, channels, internalFormat);
    for (int i=0; i < (int)bufferPool_.size(); i++) {
        const Buffer & buf = bufferPool_.at(i);
        if (buf.size() >= wanted.size()) {
            // we cannot use something as input for layer N which already has been input to layer N-1 or >=N
            if ((!buf.locked_) && (buf.lastInputLayer_ < inputLayer-1) && (outputLayer>buf.lastInputLayer_)) {
                return i;
            }
        }
    }
    return -1;
}



/**
 * @brief Update the usage of a texture (given by pool index) by a layer
 *
 * @param index Index of the texture to update
 * @param layerNumber New layer number that the texture is used as input for
 * @param lock Indicator if the texture should be locked
 */
void BufferManager::updateLayerUse(int index, int layerNumber, bool lock) {
    auto it = texturePool_.begin() + index;
    (*it).lastInputLayer_ = ((*it).lastInputLayer_ < layerNumber) ? layerNumber : (*it).lastInputLayer_;
    if (lock) {
        (*it).locked_ = true;
    }
}



/**
 * @brief Update the usage of a buffer by a layer
 *
 * @param buffer Pointer to (CPU) buffer that should be updated
 * @param layerNumber New layer number that the texture is used as input for
 * @param lock Indicator if the texture should be locked
 */
void BufferManager::updateLayerUseByBuffer(const CPUBuffer *buffer, int layerNumber, bool lock) {
    for (auto it = bufferPool_.begin(); it != bufferPool_.end(); ++it) {
        if ((*it).buf_ == buffer) {
            (*it).lastInputLayer_ = ((*it).lastInputLayer_ < layerNumber) ? layerNumber : (*it).lastInputLayer_;
            if (lock) {
                (*it).locked_ = true;
            }
            break;
        }
    }
}



/**
 * @brief Update the usage of a texture (given by handle/ID) by a layer
 *
 * @param id Raw GL texture handle that identifies the texture to be updated
 * @param layerNumber New layer number that the texture is used as input for
 * @param lock Indicator if the texture should be locked
 */
void BufferManager::updateLayerUseByTextureID(GLuint id, int layerNumber,bool lock) {
    for (auto it = texturePool_.begin(); it != texturePool_.end(); ++it) {
        if ((*it).id_ == id) {
            (*it).lastInputLayer_ = ((*it).lastInputLayer_ < layerNumber) ? layerNumber : (*it).lastInputLayer_;
            if (lock) {
                (*it).locked_ = true;
            }
            break;
        }
    }
}


/**
 * @brief Create a new CPU buffer
 *
 * @param width Width of the buffer to create
 * @param height Height of the buffer to create
 * @param channels Number of channel
 * @param iFormat OpenGL sized format that this buffer should be usable with
 * @param dType Data type to use for the buffer
 * @param order Optional data order, defaults to BufferShape::order::CHANNELWISE
 *
 * @return BufferManager::Buffer instance that wraps the allocated buffer
 */
BufferManager::Buffer BufferManager::createBuffer(int width, int height, int channels,
                                                  BufferSpec::sizedformat iFormat,
                                                  BufferSpec::dtype dType,
                                                  BufferShape::order order) {
    Buffer buf(height, width, channels, iFormat);
    BufferShape shape(height, width, channels, 0, dType, order);
    buf.buf_ = new CPUBuffer(shape);
    return buf;
}


/**
 * @brief Create a new texture
 *
 * @param width Width of texture
 * @param height Height of texture
 * @param internalFormat Sized texture format (e.g. \c GL_RGBA32F)
 * @param format Unsized texture format (e.g. \c GL_RGBA)
 * @param type GL datatype to use for the texture pixels (e.g. \c GL_FLOAT)
 * @param interpolation Interpolation mode to use
 *
 * @return BufferManager::Texture object that wraps the newly created texture
 */
BufferManager::Texture BufferManager::createTexture(int width, int height,
                                                    BufferSpec::sizedformat internalFormat,
                                                    BufferSpec::genericformat format,
                                                    BufferSpec::dtype type,
                                                    BufferSpec::interp interpolation) {
    GLuint texture=0;
    glGenTextures(1, &texture);
    if (texture == 0) THROW_EXCEPTION_ARGS(GLException,"Cannot create texture (err=0x%x)",glGetError());
    glBindTexture(GL_TEXTURE_2D,texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    switch (interpolation) {
        case BufferSpec::interp::LINEAR:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            break;
        case BufferSpec::interp::NEAREST:
            // intentional fallthrough
        default:
            // ANY interpolation defaults to nearest
            interpolation = BufferSpec::interp::NEAREST;
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            break;
    }
    CLEAR_GFXERR_DEBUG
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)internalFormat, width, height, 0, (GLenum)format, (GLenum)type, nullptr);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        THROW_EXCEPTION_ARGS(GLException,"Cannot parameterize texture (err=0x%x)",(int)err);
    }
#endif
    int elemsize = 1;
    switch ((GLint)internalFormat) {
        case GL_RGB16F:
        case GL_RGB16UI:
        case GL_RGB16I:
            elemsize = 3*2;
            break;
        case GL_RGBA16F:
        case GL_RGBA16UI:
        case GL_RGBA16I:
            elemsize = 4*2;
            break;
        case GL_RGB32F:
        case GL_RGB32UI:
        case GL_RGB32I:
            elemsize = 3*4;
            break;
        case GL_RGBA32F:
        case GL_RGBA32UI:
        case GL_RGBA32I:
            elemsize = 4*4;
            break;
        case GL_RGB8:
            elemsize = 3;
            break;
        case GL_RGBA8:
        case GL_R32UI:
        case GL_R32F:
        case GL_R32I:
            elemsize = 4;
            break;
        case GL_R16F:
        case GL_R16UI:
        case GL_R16I:
            elemsize = 2;
            break;
        default:
            break;
    }
    estimatedTextureBytes_ += width*height*elemsize;
    return {texture, width, height, internalFormat, interpolation};
}

} // fyusion::fyusenet namespace


// vim: set expandtab ts=4 sw=4:
