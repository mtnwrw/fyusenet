//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture -> CPU Buffer Download Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "downloadlayer.h"
#include "../gl/fbo.h"
#include "../gl/pbopool.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&,int)
 */
DownloadLayer::DownloadLayer(const UpDownLayerBuilder& builder, int layerNumber) :
    GPULayerBase((const GPULayerBuilder &)builder,layerNumber), CPULayerInterface(), DownloadLayerInterface() {
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
    assert(inputPadding_ == outputPadding_);
    assert(builder.direction_ == UpDownLayerBuilder::DOWNLOAD);
    assert(LayerBase::inputPadding_ == LayerBase::outputPadding_);      // NOTE (mw) for now we do not allow padding change in this layer
    if (builder.isSequence()) {
        maxSequence_ = builder.maxSequenceLen_;
        dataType_ = builder.dataType_;              // TODO (mw) also support different types for non-sequence data:
        chanPacking_ = builder.seqPacking_;
        bytesPerChan_ = BufferSpec::typeSize(dataType_, true);      // NOTE (mw) we assume that we never download FP16 without converting it to FP32
    }
#ifdef FYUSENET_MULTITHREADING
    if (builder.callback_) userCallback_ = builder.callback_;
    async_ = builder.async_;
#endif
}


/**
 * @copydoc DownloadLayerInterface::getOutputShape
 */
BufferShape DownloadLayer::getOutputShape(int port) const {
    assert(port == 0);
    if (maxSequence_> 0) {
        return {width_, maxSequence_, dataType_, chanPacking_};
    } else {
        return {width_, height_, inputChannels_, inputPadding_, dataType_, BufferShape::order::GPU_SHALLOW};
    }
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DownloadLayer::getRequiredInputBuffers() const {
    static BufferSpec::genericformat genint[4] = {BufferSpec::genericformat::SINGLE_INT, BufferSpec::genericformat::RG_INT,
                                                  BufferSpec::genericformat::RGB_INT, BufferSpec::genericformat::RGBA_INT};
    static BufferSpec::genericformat genfp[4] = {BufferSpec::genericformat::SINGLE, BufferSpec::genericformat::RG,
                                                 BufferSpec::genericformat::RGB, BufferSpec::genericformat::RGBA};
    std::vector<BufferSpec> result;
    if (maxSequence_ > 0) {
        // TODO (mw) support other integral data types here (this part is only for downloading tokens)
        auto genfmt = (isInt(dataType_)) ? genint[chanPacking_ - 1] : genfp[chanPacking_ - 1];
        auto spec = BufferSpec(0, 0, width_, maxSequence_, bufferFormat(dataType_, chanPacking_), genfmt, dataType_, BufferSpec::GPU_DEST,
                               inputChannels_).device(BufferSpec::csdevice::COMP_STOR_GPU).dataOrder(BufferSpec::order::GPU_SEQUENCE);
        result.push_back(spec);
    } else {
        int channel = 0;
        int rem = inputChannels_;
        while (rem > 0) {
            result.emplace_back(channel++, 0,
                                width_ + 2 * inputPadding_, height_ + 2 * inputPadding_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE, std::min(PIXEL_PACKING, rem));
            rem -= PIXEL_PACKING;
        }
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DownloadLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    if (maxSequence_ > 0) {
        bool isint = isInt(dataType_);
        // TODO (mw) support other integral data types here (this part is only for downloading tokens)
        result.push_back(BufferSpec(0, 0, width_ * chanPacking_, maxSequence_, (isint) ? BufferSpec::sizedformat::SINGLE32UI : BufferSpec::sizedformat::SINGLE32F,
                                    (isint) ? BufferSpec::genericformat::SINGLE_INT : BufferSpec::genericformat::SINGLE,
                                    dataType_, BufferSpec::CPU_DEST, 1)
                                   .device(BufferSpec::csdevice::COMP_STOR_CPU)
                                   .dataOrder(BufferSpec::order::GPU_SEQUENCE));

    } else {
        // NOTE (mw) many GL(ES) implementations cannot write to RGB textures, so we reserve enough
        // space to use RGBA textures as source(s) instead
        result.push_back(BufferSpec(0, 0,
                                    width_ + 2 * outputPadding_, height_ + 2 * outputPadding_,
                                    BufferSpec::sizedformat::SINGLE32F, BufferSpec::genericformat::SINGLE, BufferSpec::dtype::FLOAT,
                                    BufferSpec::CPU_DEST,
                                    outputChannels_ + PIXEL_PACKING - (outputChannels_ % PIXEL_PACKING))
                                   .device(BufferSpec::csdevice::COMP_STOR_CPU)
                                   .dataOrder(BufferSpec::order::GPU_SHALLOW));
    }
    return result;
}


/**
 * @copydoc cpu::CPULayerInterface::clearCPUOutputBuffers
 */
void DownloadLayer::clearCPUOutputBuffers(int port) {
    assert(port == 0);
    outputs_.clear();
}


/**
 * @copydoc LayerBase::setup
 */
void DownloadLayer::setup() {
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::forward
 */
void DownloadLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    assert(outputs_.size() == 1);
    // TODO (mw) implement optional rendering step here (for ReLU)
    if ((maxSequence_ > 0) && (!state)) THROW_EXCEPTION_ARGS(FynException, "Download layer requires state state in sequenceNo processing");
    sequenceLen_ = (state) ? state->seqLength : 0;
    ManagedPBO pbo = pboBlit();
#ifndef FYUSENET_MULTITHREADING
    if (true) {
#else
    if (!async_) {
#endif
        //-------------------------------------------------------------
        // Synchronous part, we still use a PBO here though there is no
        // advantage doing that. It just makes the code easier.
        //-------------------------------------------------------------
        // FIXME (mw) we actually don't need to download all tokens here, just one, make this a builder flag
        size_t read = (maxSequence_ > 0) ? sequenceLen_ * width_ * chanPacking_ * bytesPerChan_: 0;
        outputs_[0]->readFromPBO(*pbo, dataType_, sequenceNo, read);
    } else {
#ifdef FYUSENET_MULTITHREADING
        THROW_EXCEPTION_ARGS(FynException, "Layer is not synchronous");
#endif
    }
}

#ifdef FYUSENET_MULTITHREADING
/**
 * @copydoc DownloadLayerInterface::asyncForward
 */
void DownloadLayer::asyncForward(uint64_t sequenceNo, StateToken * token, const std::function<void (uint64_t)> &callback) {
    // TODO (mw) implement optional rendering step here (for ReLU)
    if (!async_) THROW_EXCEPTION_ARGS(FynException, "Layer is not asynchronous");
    ManagedPBO pbo = pboBlit();
    //-------------------------------------------------------------
    // We issue a fence here and start a thread that waits for the
    // fence before reading out the PBO...
    //-------------------------------------------------------------
    GLsync sync = context().issueSync();
    asyncLock_.lock();
    auto thread = AsyncPool::getDerivedContextThread(context());
    threads_[sequenceNo] = thread;
    thread->setTask(std::bind(&DownloadLayer::readoutPBO, this, thread, pbo, sync, sequenceNo, outputs_[0], callback));
    if (userCallback_) userCallback_(sequenceNo, outputs_[0], AsyncLayer::DOWNLOAD_COMMENCED);
    asyncLock_.unlock();
}
#endif


/**
 * @copydoc DownloadLayerInterface::wait
 */
void DownloadLayer::wait(uint64_t sequenceNo) {
#ifdef FYUSENET_MULTITHREADING
    if (async_) {
        asyncLock_.lock();
        auto it = threads_.find(sequenceNo);
        if (it != threads_.end()) {
            AsyncPool::GLThread thread = it->second;                   // make sure that we keep the refcount in the thread > 0
            asyncLock_.unlock();
            thread->wait();
        } else {
            asyncLock_.unlock();
        }
    }
#endif
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void DownloadLayer::updateFBOs() {
    outputChanged_ = false;
}


/**
 * @copydoc cpu::CPULayerInterface::addCPUOutputBuffer
 */
void DownloadLayer::addCPUOutputBuffer(CPUBuffer *buf, int port) {
    assert(buf);
    if (port != 0) THROW_EXCEPTION_ARGS(FynException, "Ports other than 0 are not supported");
    if (!outputs_.empty()) THROW_EXCEPTION_ARGS(FynException,"Only one output buffer is supported for this layer type");
    outputs_.push_back(buf);   
    outputChanged_ = true;
}


/**
 * @brief Update output CPU buffer
 *
 * @param buf Pointer to CPUBuffer that is to replace the currently set buffer
 *
 * @param port Output port number, currently only 0 is accepted
 *
 * @note This function does not assume ownership over the supplied \p buf
 */
void DownloadLayer::updateOutputBuffer(CPUBuffer *buf, int port) {
    assert(buf);    
    if (port != 0) THROW_EXCEPTION_ARGS(FynException, "Ports other than 0 are not supported");
#ifdef FYUSENET_MULTITHREADING
    asyncLock_.lock();
#endif
    if (outputs_.size() != 1) THROW_EXCEPTION_ARGS(FynException, "No buffer position to be updated");
    outputs_[port] = buf;
    outputChanged_ = true;
#ifdef FYUSENET_MULTITHREADING
    asyncLock_.unlock();
#endif
}


/**
 * @copydoc cpu::CPULayerInterface::hasCPUOutputBuffer
 */
bool DownloadLayer::hasCPUOutputBuffer(int port) const {
    assert(port >= 0);
    if ((int)outputs_.size() <= port) return false;
    return (outputs_.at(port) != nullptr);
}


/**
 * @copydoc cpu::CPULayerInterface::getCPUOutputBuffer
 */
CPUBuffer * DownloadLayer::getCPUOutputBuffer(int port) const {
    assert(port >= 0);
    if ((int)outputs_.size() <= port) return nullptr;
    return outputs_.at(port);
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Blit texture content into %PBO
 *
 * @return ManagedPBO instance that wraps the %PBO in the operation
 *
 * This function blits the texture data into a %PBO which has sufficient capacity to hold the
 * content.
 */
ManagedPBO DownloadLayer::pboBlit() {
    // a lot of PBO binds/unbinds here, maybe consolidate when there is time
    PBOPool *pool = context_.interface()->getReadPBOPool();
    assert(pool);
    bool seq = (maxSequence_ > 0);
    int paddedwidth = (seq) ? width_ : width_ + 2*inputPadding_;
    int paddedheight = (seq) ? maxSequence_ : height_ + 2*inputPadding_;
    int paddedchannels = (seq) ? chanPacking_ : LayerBase::PIXEL_PACKING * ((outputChannels_ + LayerBase::PIXEL_PACKING - 1) / LayerBase::PIXEL_PACKING);
    ManagedPBO pbo = pool->getAvailablePBO(paddedwidth, paddedheight, paddedchannels, bytesPerChan_);
    pbo->prepareForRead(paddedwidth * paddedheight * paddedchannels * bytesPerChan_);
    int readchans = 0;
    pbo->bind(GL_PIXEL_PACK_BUFFER);
    for (int fb = 0 ; fb < numFBOs(); fb++) {
        size_t offset = readchans * paddedwidth * paddedheight * bytesPerChan_;
        FBO *fbo = getFBO(fb);
        fbo->bind();
        int chans = chanPacking_ * fbo->numAttachments();
        if (maxSequence_ > 0) {
            fbo->copyToPBO(*pbo, width_, sequenceLen_, (GLenum)dataType_, chanPacking_, offset, false, isInt(dataType_));
        } else {
            // TODO (mw) support for different data types here too
            fbo->copyToPBO(*pbo, GL_FLOAT, chanPacking_, offset);
        }
        fbo->unbind();
        readchans += chans;
    }
    pbo->unbind(GL_PIXEL_PACK_BUFFER);
    if (async_) pbo.setPending();
    return pbo;
}

#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Perform readout of PBO memory buffer into destination CPUBuffer instance
 *
 * @param myThread Reference to GLThread that this function runs on
 * @param pbo Reference to a ManagedPBO instance which wraps the PBO to be read out
 * @param sync Handle of the OpenGL fence sync that indicates when the PBO is ready for readout
 * @param sequence Sequence number that refers to the content in the PBO to be read out
 * @param target Pointer to CPUBuffer where the data should be placed in
 * @param callback Callback function in the engine to pass notification about finished download
 *
 * @throw FynException in case the \p sync was not posted on the GL pipeline after less than 5s
 *
 * This function waits for the supplied \p sync to be issued on the GL pipeline in a background
 * thread (to be more precise, it is invoked in the background thread already). Once the sync has
 * been received, the \p pbo will be mapped into memory and the data will be copied to the buffer(s)
 * in #outputs_. After reading the data, two callbacks will be invoked:
 *   - \p callback which notifies the engine that the PBO has been read
 *   - #userCallback_ which notifies the API user that the PBO has been read
 *
 * The latter callback is optional and is supplied via the UpDownLayerBuilder.
 *
 * @see UpDownLayerBuilder, Engine::asyncDownloadDone
 */
void DownloadLayer::readoutPBO(AsyncPool::GLThread& myThread, opengl::ManagedPBO& pbo, GLsync sync, uint64_t sequence, cpu::CPUBuffer * target, const std::function<void(uint64_t)> & callback) {
    using namespace opengl;
    const GfxContextLink & ctx = myThread.context();
    bool rc = ctx.waitClientSync(sync, 5000000000);        // wait 5s max  (TODO (mw) configurable timeout)
    if (!rc) THROW_EXCEPTION_ARGS(FynException, "Cannot read out texture within 5s for sequence %ld", sequence);
    ctx.removeSync(sync);
    target->readFromPBO(*pbo, BufferShape::type::FLOAT32, sequence);
    pbo.clearPending();
    if (callback) callback(sequence);
    if (userCallback_) userCallback_(sequence, target, AsyncLayer::DOWNLOAD_DONE);
    asyncLock_.lock();
    auto it = threads_.find(sequence);
    assert(it != threads_.end());
    threads_.erase(it);
    asyncLock_.unlock();
}
#endif


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void DownloadLayer::setupFBOs() {
    assert(inputChannels_ == outputChannels_);
    if (flags_ & LayerFlags::PRE_ACT_MASK) THROW_EXCEPTION_ARGS(FynException,"Activation on download not implemented yet");
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"Residual add on download not implemented yet");
    // we can directly connect the input textures to the FBOs here for now, as we
    // currently only support downloading float data without any flags
    int numpasses = ((int)inputTextures_.size()  + maxRenderTargets_ - 1) / maxRenderTargets_;
    int texoffset = 0;
    for (int pass=0; pass < numpasses; pass++) {
        FBO * fbo = new FBO(context_, viewport_[0], viewport_[1], inputTextures_.at(texoffset++));
        int pack = 1;
        while ((pack < maxRenderTargets_) && (texoffset < (int)inputTextures_.size())) {
            fbo->addTexture(GL_COLOR_ATTACHMENT0+pack,inputTextures_.at(texoffset++), GL_TEXTURE_2D);
            pack++;
        }
        fbo->unbind();
        framebuffers_.push_back(fbo);
    }
    outputChanged_ = false;
}


/**
 * @brief Determine (sized) buffer format specified for given data type and packing
 *
 * @param type Data type in the buffer (per atom / channel)
 * @param packing Number of atoms packed into a single item (# of channels)
 *
 * @return Sized buffer format specifier
 *
 * This function computes a sized buffer format specifier which is to be used to create buffer
 * objects or provide information to the buffer manager. Depending on the data type of each
 * atom and the number of atoms per item (channel packing), different sized formats are required.
 */
BufferSpec::sizedformat DownloadLayer::bufferFormat(BufferSpec::dtype type, int packing) {
    assert(packing >= 1 && packing <= PIXEL_PACKING);
    using fmt = BufferSpec::sizedformat;
#ifndef FYUSENET_USE_EGL
    static fmt fmtfp32[4] = {fmt::SINGLE32F, fmt::RG32F, fmt::RGB32F, fmt::RGBA32F};
    static fmt fmtui32[4] = {fmt::SINGLE32UI, fmt::RG32UI, fmt::RGB32UI, fmt::RGBA32UI};
#else // EGL and WebGL
    static fmt fmtfp32[4] = {fmt::SINGLE32F, fmt::RG32F, fmt::RGBA32F, fmt::RGBA32F};
    static fmt fmtui32[4] = {fmt::SINGLE32UI, fmt::RG32UI, fmt::RGBA32UI, fmt::RGBA32UI};
#endif
    switch (type) {
        case BufferSpec::dtype::FLOAT16:
            // currently not supported, fallback to float
        case BufferSpec::dtype::FLOAT32:
            return fmtfp32[packing-1];
        case BufferSpec::dtype::INT32:
            // currently not supported, fallback to uint32
        case BufferSpec::dtype::UINT32:
            return fmtui32[packing-1];
        default:
            THROW_EXCEPTION_ARGS(FynException,"Unsupported combination of datatype %d and packing %d", (int)type, packing);
    }
}


/**
 * @brief Determine if given data type is an integral type
 *
 * @param type Data type to check
 *
 * @retval true Data type is integral
 * @retval false Otherwise
 */
inline bool DownloadLayer::isInt(BufferSpec::dtype type) {
    // TODO (mw) support other integral data types here
    return (type == BufferSpec::dtype::UINT32) || (type == BufferSpec::dtype::INT32) ||
           (type == BufferSpec::dtype::UINT16) || (type == BufferSpec::dtype::INT16);
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
