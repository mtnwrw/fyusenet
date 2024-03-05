//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Network Execution Engine
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <unordered_map>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "engine.h"
#include "../base/neuralnetwork.h"
#include "../common/performance.h"
#include "../gpu/uploadlayer.h"
#include "../gpu/downloadlayer.h"
#include "../gpu/deep/deepdownloadlayer.h"

//-------------------------------------- Global Variables ------------------------------------------


namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------

// Maximum type to wait for the fence sync in the GL pipeline (in ns)
#define SYNC_EXPIRY 5000000000


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param context Link to GL context that this engine instance should work under if not asynchronous
 * @param async Flag that controls whether the engine is supposed to run asynchronously
 *
 * Construct an Engine object around the supplied \p context. In case of a multi-threaded build,
 * and with the \p async parameter set to true, an engine thread is created with a GL context that
 * is derived/shared from/with the supplied \p context and this thread is and will be used to
 * handle the actual inference.
 *
 * @see #exec_
 */
Engine::Engine(const GfxContextLink& context, bool async) : GfxContextTracker() {
    setContext(context);
#ifdef FYUSENET_MULTITHREADING
    if (async) {
        exec_ = opengl::AsyncPool::getDerivedContextThread(context);
        async_ = true;
    }
#endif
}


/**
 * @brief Run network setup
 *
 * @param net Pointer to network that should be brought up
 *
 * This function runs the glSetup() method of the supplied network and registers the layers of
 * that net with the engine instance.
 */
void Engine::setup(NeuralNetwork *net) {
#ifndef FYUSENET_MULTITHREADING
    if (net) {
        setLayers(net->gpuSetup());
        setup_ = true;
    }
#else
    if (async_ && net) {
        net->setContext(exec_.context());
        auto init = [this, net]() { setLayers(net->gpuSetup()); };
        exec_->waitTask(init);
        setup_ = true;
        exec_->setTask(std::bind(&Engine::looper,this,exec_.context()));
    } else {
        if (net) {
            setLayers(net->gpuSetup());
            setup_ = true;
        }
    }
#endif
}

/**
 * @brief Release resources of layers in this engine
 *
 * @param broom Cleanup callback to invoke after deleting the layer
 *
 * This function releases all (GPU) resources occupied by the layers that are registered to it.
 * Note that the layer instances themselves are not destroyed at this point.
 *
 * @see CompiledLayers::cleanup, setLayers()
 */
void Engine::cleanup(const std::function<void()> & broom) {
#ifdef FYUSENET_MULTITHREADING
    if ((async_) && (setup_)) {
        // ---------------------------------------------
        // Tell the looper we want it to quit and make
        // sure it gets the message
        // ---------------------------------------------
        quit_ = true;
        looperLock_.lock();
        pendingStates_++;
        looperLock_.unlock();
        looperWait_.notify_all();
        exec_->wait();
#ifdef DEBUG
        // ---------------------------------------------
        // Some safeguards for debug builds
        // ---------------------------------------------
        asyncStateLock_.lock();
        assert(numBackgroundTasks_ == 0);
        assert(readyStates_.empty());
        assert(asyncDownloadDependencies_.empty());
        assert(asyncUploadDeferredDependencies_.empty());
        assert(asyncDownloadWaiters_.empty());
        asyncStateLock_.unlock();
#endif
    }
    // ---------------------------------------------
    // ..and run cleanup
    // ---------------------------------------------
    if (setup_) {
        auto brush = [broom, this]() {
            layers_.cleanup();
            if (broom) broom();
        };
        if (async_) exec_->waitTask(brush);
        else {
            layers_.cleanup();
            if (broom) broom();
        }
        setup_ = false;
    }
#else
    if (setup_) {
        layers_.cleanup();
        if (broom) broom();
        setup_ = false;
    }
#endif
}


/**
 * @brief Enable layer-by-layer output to files for debug purposes
 *
 * @param outputDir Directory on the file system to write the output data to
 *
 * @note This function only work when compiled in debug mode. Otherwise this is a no-op.
 *
 * This function enables the engine to write binary output files on a per-layer basis in order to
 * inspect the results of the network execution. Output files will be written as 32-bit floating-point
 * raw data during the calls to forwardLayers().
 *
 * @warning This function is not thread-safe, do not call it in parallel to forwardLayers()
 */
void Engine::enableIntermediateOutput(const std::string& outputDir) {
#ifndef DEBUG
    FNLOGW("Intermediate data output not available for non-debug builds");
#else
    outputDir_ = outputDir;
    writeResults_ = true;
#endif
}


/**
 * @brief Disable layer-by-layer output
 *
 * This disables layer-by-layer debug output.
 *
 * @warning This function is not thread-safe, do not call it in parallel to forwardLayers()
 */
void Engine::disableIntermediateOutput() {
#ifdef DEBUG
    writeResults_ = false;
#endif
}


/**
 * @brief Enable taking layer-by-layer timings during execution
 *
 * This enables taking timings for individual layer execution. Please note that these timings
 * usually <b>do not reflect</b> what the real timings on the GPU are, since the GPU execution
 * happens in its own command queue and usually is quite decoupled from the CPU issuing those
 * commands.
 *
 * @warning This function is not thread-safe, do not call it in parallel to forwardLayers()
 *
 * @see getTimings(), resetTimings(), disableTimings()
 */
void Engine::enableTimings() {
    timings_ = true;
}


/**
 * @brief Disable taking layer-by-layer timings
 *
 * @see enableTimings()
 */
void Engine::disableTimings() {
    timings_ = false;
}


/**
 * @brief Reset timing log data
 *
 * @warning This function is not thread-safe, do not call it in parallel to forwardLayers(),
 *          call it directly after a call to finish() to make sure that it does what it is
 *          supposed to it
 */
void Engine::resetTimings() {
    runs_ = 0;
    timingData_.clear();
}


/**
 * @brief Flushes pending operations and waits for their completion
 *
 * @throws FynException in case of errors during the execution or if a timeout occurs
 *
 * This function flushes pending operations in the network until \e all operations have been
 * fully executed. Use this function to make sure that no async operation is still running in the
 * background.
 *
 * @see execute(), forwardLayers()
 */
void Engine::finish() {
#ifdef FYUSENET_MULTITHREADING
    using namespace std::chrono_literals;
    if (async_) {
        std::lock_guard<std::mutex> guard(runGuard_);
        std::unique_lock<std::mutex> lck(sequenceLock_);
        // -------------------------------------------------
        // Wait for the engine thread to retire the last
        // sequence...
        // -------------------------------------------------
        while ((engineSequence_ + 1) < sequenceNo_) {
            sequenceDone_.wait(lck, [this]() { return ((engineSequence_ + 1) >= sequenceNo_);});
        }
        // -------------------------------------------------
        // Make sure there are no more downloads pending..
        // -------------------------------------------------
        asyncStateLock_.lock();
        int totalwait = 0;
        int bg = numBackgroundTasks_;
        while (bg > 0 && totalwait < 5000) {
            asyncStateLock_.unlock();
            std::this_thread::sleep_for(10ms);
            totalwait += 10;
            asyncStateLock_.lock();
            bg = numBackgroundTasks_;
        }
        asyncStateLock_.unlock();
        if (bg) THROW_EXCEPTION_ARGS(FynException, "Engine did not finish after 5s");
    }
#endif
}



/**
 * @brief Execute all registered layers in ascending order
 *
 * @param token Optional StateToken instance which keeps track of information in stateful networks,
 *              <i>the ownership is transferred to the engine</i>
 *
 * @return State of the engine on return of this function, see detailed description for more info
 *
 * @throws FynException in case of errors during the execution
 *
 * This function executes all network layers in order of their layer numbers. In a multi-threaded
 * build configuration, this function also takes care of asynchronous operations by first checking
 * if there are any pending async operations and continuing these before executing the next batch
 * of layer runs. On exit, this function returns the last state of the engine, which can take the
 * following values:
 *   - \c EXEC_DONE , the execution of a single run through all layers is complete (pending GL operations)
 *   - \c EXEC_DEFERRED , the execution was deferred due to an asynchronous operation and the engine
 *        background thread is still working on it
 *   - \c EXEC_ERROR, there was an error during execution
 *   - \c EXEC_STOPPED, the Engine is about to be taken down or has been taken down already
 *
 * @see execute(), setLayers(), finish(), lastSequenceNo()
 *
 * @note The combination of calling forwardLayers() and then retrieving the sequence number
 *       that was issued by that call is not thread-safe. It is up to the caller to ensure
 *       that the engine is locked or that all engine calls are made from the same thread,
 *       the latter being the recommended mode of operation.
 *
 * @see execute(), looper(), finish()
 */
Engine::execstate Engine::forwardLayers(StateToken * token) {
#ifdef FYUSENET_MULTITHREADING
    if (async_) {
        std::lock_guard<std::mutex> guard(runGuard_);
        if (quit_) return execstate::EXEC_STOPPED;
        std::unique_lock<std::mutex> seq(sequenceLock_);
        while ((engineSequence_ + 2) < sequenceNo_) {
            sequenceDone_.wait(seq, [this]() { return ((engineSequence_ + 1) >= sequenceNo_);});
        }
        assert((engineSequence_ + 2) >= sequenceNo_);
        ExecutionState estate(sequenceNo_++, layers_.begin());
        seq.unlock();
        if (newSeqCallback_) newSeqCallback_(sequenceNo_);
        asyncStateLock_.lock();
        looperLock_.lock();
        readyStates_.push_back(estate);
        pendingStates_++;
        looperLock_.unlock();
        asyncStateLock_.unlock();
        looperWait_.notify_one();
        return execstate::EXEC_DEFERRED;
    }
#endif
    ExecutionState estate(sequenceNo_++, layers_.begin(), token);
    state status = execute(estate, context_);
    glDisable(GL_BLEND);
    return (status == state::DONE) ? execstate::EXEC_DONE : execstate::EXEC_ERROR;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Perform execution of all network layers in ascending order
 *
 * @param state Execution state to start with (might not be the initial state in case of
 *              continuation of asynchronous operations).
 *
 * @param context Link to GL context to be used for executing the layers
 *
 * @return Final execution state on exit of loop
 *
 * This function dispatches the actual network inference by invoking LayerBase::forward() on each
 * layer in the network. The behaviour on a fully synchronous network is straightforward: each layer
 * is executed and this function will return Engine::state::DONE in that case. In case an
 * asynchronous layer is encountered, this function will dispatch the execution of that layer in a
 * different thread and will come to a halt in case a layer is encountered which depends on the
 * execution of the asynchronous layer.
 *
 * There are currently two types of asynchronous layers: upload and download layers. The handling
 * of asynchronous uploads works by moving the upload itself to a thread while memorizing the
 * execution context before exiting with the \c Engine::state::UPLOADING state. Once the upload thread
 * is finished, it will push the execution state to the #readyStates_ queue and upon the next call
 * to forward() or finish(), the execution state will be taken from the queue and the execution
 * will be continued from there, before moving on to the next inference run.
 *
 * For asynchronous downloads, the behaviour is different. Upon encountering an asynchronous
 * download layer, the download operation will again be pushed to a background thread and the
 * pipeline will continue executing. Once a layer is reached that is \e dependent on a pending
 * asynchronous download, the pipeline will push the current state to a set of waiting states
 * and exit with the \c Engine::state::DOWNLOADING state. When the pending download is ready, it
 * will check for any pending execution states and upon encountering those, will push that
 * state to the #readyStates_ list for deferred execution.
 *
 * On exit, this function returns the last state of the engine, which can take the following values:
 *   - \c DONE : the execution of a single run through all layers is complete (pending GL operations)
 *   - \c UPLOADING : the execution was deferred due to an asynchronous upload operation
 *   - \c DOWNLOADING : the execution was deferred due to an asynchronous download operation
 *   - \c NET_ERROR : there was an error during execution
 *
 * @see looper(), forwardLayers(), finish()
 *
 * @todo refactor
 */
// TODO (mw) this function literally screams "refactor me"
Engine::state Engine::execute(ExecutionState& state, const GfxContextLink & context) {
    using namespace gpu;
    tstamp start, end;
    std::string fname;
    StateToken * stoken = state.state_;
    //-----------------------------------------------------------
    // Traverse through layers in ascending order of layer number
    //-----------------------------------------------------------
    while (state.current != layers_.end()) {
        int idx = state.current.first;
        LayerBase * layer = state.current.second;
        assert(layer);
        if (layer) {
            bool masked = stoken && stoken->maskLayers.find(layer->getNumber()) != stoken->maskLayers.end();
            //---------------------------------------------------------------
            // If this layer is dependent on a currently running async
            // download or upload, mark down the state and bail out here,
            // even if the layer is masked. Otherwise, we break the sequence
            //---------------------------------------------------------------
#ifdef FYUSENET_MULTITHREADING
            // we assume that we don't have direct upload -> download connections
            asyncStateLock_.lock();
            if (asyncDependencies_.find(layer->getNumber()) != asyncDependencies_.end()) {
                for (auto & dep : asyncDownloadDependencies_) {
                    if (dep.dependency == layer->getNumber() && dep.sequenceNo == state.sequenceNo) {
                        if (minimumWaitingDependency_.find(state.sequenceNo) != minimumWaitingDependency_.end()) {
                            minimumWaitingDependency_[state.sequenceNo] = std::min(minimumWaitingDependency_[state.sequenceNo], layer->getNumber());
                        } else minimumWaitingDependency_[state.sequenceNo] = layer->getNumber();
                        asyncDownloadWaiters_.emplace_back(layer->getNumber(), (AsyncLayer *)dep.provider, state.sequenceNo, state.clone());
                        asyncStateLock_.unlock();
                        return state::DOWNLOADING;
                    }
                }
                for (auto & dep : asyncUploadDependencies_) {
                    if (dep.dependency == layer->getNumber() && dep.sequenceNo == state.sequenceNo) {
                        if (minimumWaitingDependency_.find(state.sequenceNo) != minimumWaitingDependency_.end()) {
                            minimumWaitingDependency_[state.sequenceNo] = std::min(minimumWaitingDependency_[state.sequenceNo], layer->getNumber());
                        } else minimumWaitingDependency_[state.sequenceNo] = layer->getNumber();
                        asyncUploadWaiters_.emplace_back(layer->getNumber(), (UploadLayer *)dep.provider, state.sequenceNo, state.clone());
                        asyncStateLock_.unlock();
                        return state::UPLOADING;
                    }
                }
            }
            asyncStateLock_.unlock();
#endif
            //-----------------------------------------------------------
            // Generate output filename if we are supposed to write
            // intermediate results...
            //-----------------------------------------------------------
            // TODO (mw) hacky, use some filesystem abstraction here
            if ((writeResults_) && (!masked)) {
                if (!outputDir_.empty()) fname = outputDir_ + std::string("/") + layer->getName() + std::string("_") + std::to_string(state.sequenceNo)+ std::string(".bin");
                else fname = layer->getName() + std::string("_") + std::to_string(state.sequenceNo) + std::string(".bin");
            }
            //-----------------------------------------------------------
            // Handle CPU layers...
            //-----------------------------------------------------------
            if ((layer->getDevice() == compute_device::DEV_CPU) && (!masked)) {
                auto * cpulay = dynamic_cast<cpu::CPULayerBase *>(layer);
                if (timings_) start = fy_get_stamp();
                cpulay->forward(state.sequenceNo, stoken);
                if (timings_) {
                    end = fy_get_stamp();
                    if (runs_ == 0) timingData_[idx] = 0;
                    timingData_[idx] += fy_elapsed_micros(start, end);
                }
                if (writeResults_) {
                    // NOTE (mw) we assume it is floating point data every time
                    cpulay->getCPUOutputBuffer()->write<float>(fname.c_str());
                }
            } else {
                //-----------------------------------------------------------
                // Handle upload layers..
                //-----------------------------------------------------------
                if (dynamic_cast<UploadLayer *>(layer) && !masked) {
                    auto * ul = dynamic_cast<UploadLayer *>(layer);
                    if ((ul)->getCPUInputBuffer() == nullptr) THROW_EXCEPTION_ARGS(FynException, "No input buffer in upload layer %s", ul->getName().c_str());
                    if (ul->isAsync()) {
#ifdef FYUSENET_MULTITHREADING
                        //-----------------------------------------------------------
                        // For async upload layers, we register two dependencies:
                        //  1. an early stage dep which is the _first_ layer that expects
                        //     an input from the UL
                        //  2. a deferred dep which is the _last_ layer that expects an
                        //     input from the UL
                        // We then invoke async processing on the layer and then
                        // continue execution. The deferred dependency is important to
                        // make sure that no texture is overwritten before it has been
                        // processed by the last dependent layer in the chain.
                        //-----------------------------------------------------------
                        upIssueLock_.lock();
                        bool issueok = ul->asyncForward(state.sequenceNo, stoken, std::bind(&Engine::uploadCallback, this, ul, std::placeholders::_1));
                        if (issueok) {
                            // transition to asyncStateLock_
                            asyncStateLock_.lock();
                            upIssueLock_.unlock();
                            int firstdep = ul->firstAsyncDependency();
                            int lastdep = ul->lastAsyncDependency();
                            asyncDependencies_.insert(firstdep);
                            //-----------------------------------------------------------
                            // Check if the layer is already used in an upload (from a
                            // previous run), if not then mark it as being active for
                            // this run...
                            //-----------------------------------------------------------
                            uint8_t depcount = 1;
                            if (activeUploadDependencies_.find(ul) == activeUploadDependencies_.end()) activeUploadDependencies_[ul] = state.sequenceNo;
                            else {
                                if (activeUploadDependencies_[ul] == 0) activeUploadDependencies_[ul] = state.sequenceNo;
                                else {
                                    depcount++;
                                }
                            }
                            Dependency<gpu::UploadLayer> early(firstdep, ul, depcount, state.sequenceNo);
                            Dependency<gpu::UploadLayer> late(lastdep, ul, 1, state.sequenceNo);
                            if (depcount == 2) early.deferredNo = activeUploadDependencies_[ul];
                            asyncUploadDependencies_.push_back(early);
                            asyncUploadDeferredDependencies_.push_back(late);
                            deferredAsyncDependencies_.insert(lastdep);
                            numBackgroundTasks_++;           // the async forward above triggers a background upload task
                        } else {
                            asyncStateLock_.lock();
                            upIssueLock_.unlock();
                            //-------------------------------------------------------
                            // There are no free PBO slots for the uploads, add a
                            // self-referential pending state to the upload waiters
                            // and let waitForUploadFence() unlock that later...
                            //-------------------------------------------------------
                            asyncUploadWaiters_.emplace_back(layer->getNumber(), ul, state.sequenceNo, state.clone());
                        }
                        asyncStateLock_.unlock();
#else
                        THROW_EXCEPTION_ARGS(FynException,"No multithreading support compiled in");
#endif
                    } else {
                        ul->forward(state.sequenceNo, stoken);
                        if (writeResults_) {
                            (dynamic_cast<GPULayerBase *>(layer))->writeResult(fname.c_str(), false);
                        }
                    }
                } else
                //-------------------------------------------------------
                // Handle download layers (shallow and deep)...
                //-------------------------------------------------------
                if (dynamic_cast<DownloadLayer *>(layer) && !masked) {
                    auto * dl = dynamic_cast<DownloadLayer *>(layer);
                    if (timings_) start = fy_get_stamp();
                    CPUBuffer * buf = dl->getCPUOutputBuffer(0);
                    if (!buf) THROW_EXCEPTION_ARGS(FynException,"No output buffer in download layer %s", dl->getName().c_str());
                    if (dl->isAsync()) {
#ifdef FYUSENET_MULTITHREADING
                        //-----------------------------------------------------------
                        // For async download layers we enter the layer as a download
                        // dependency, then invoke async processing on the layer
                        // before we continue execution...
                        //-----------------------------------------------------------
                        int fd = dl->firstAsyncDependency();
                        asyncStateLock_.lock();
                        if (fd >= 0) {
                            asyncDownloadDependencies_.emplace_back(fd, static_cast<AsyncLayer *>(dl), 1, state.sequenceNo);
                            if (asyncDependencies_.find(fd) != asyncDependencies_.end()) asyncDependencies_.insert(fd);
                        }
                        numBackgroundTasks_++;                  // the async forward below generates a new background task
                        asyncStateLock_.unlock();
                        dl->asyncForward(state.sequenceNo, stoken, std::bind(&Engine::asyncDownloadDone, this, dl, std::placeholders::_1));
#else
                        THROW_EXCEPTION_ARGS(FynException,"No multithreading support compiled in");
#endif
                    } else dl->forward(state.sequenceNo, stoken);
                    if (timings_) {
                        end = fy_get_stamp();
                        if (runs_ == 0) timingData_[idx] = 0;
                        timingData_[idx] += fy_elapsed_micros(start, end);
                    }
                    if ((writeResults_) && (!dl->isAsync())) {
                        // TODO (mw) also handle write-out for asynchronous layers, currently they are ignored
                        buf->write<float>(fname.c_str());
                    }
                } else
                if (dynamic_cast<deep::DeepDownloadLayer *>(layer) && !masked) {
                    auto * dl = dynamic_cast<deep::DeepDownloadLayer *>(layer);
                    if (timings_) start = fy_get_stamp();
                    CPUBuffer * buf = dl->getCPUOutputBuffer(0);
                    if (!buf) THROW_EXCEPTION_ARGS(FynException,"No output buffer in download layer %s", dl->getName().c_str());
                    if (dl->isAsync()) {
#ifdef FYUSENET_MULTITHREADING
                        //-----------------------------------------------------------
                        // For async download layers we enter the layer as a download
                        // dependency, then invoke async processing on the layer
                        // before we continue execution...
                        //-----------------------------------------------------------
                        int fd = dl->firstAsyncDependency();
                        asyncStateLock_.lock();
                        if (fd >= 0) {
                            asyncDownloadDependencies_.emplace_back(fd, dynamic_cast<AsyncLayer *>(dl), 1, state.sequenceNo);
                            if (asyncDependencies_.find(fd) != asyncDependencies_.end()) asyncDependencies_.insert(fd);
                        }
                        numBackgroundTasks_++;                      // the async forward below creates a background task
                        asyncStateLock_.unlock();
                        dl->asyncForward(state.sequenceNo, stoken, std::bind(&Engine::asyncDownloadDone, this, dl, std::placeholders::_1));
#else
                        THROW_EXCEPTION_ARGS(FynException,"No multithreading support compiled in");
#endif
                    } else dl->forward(state.sequenceNo, stoken);
                    if (timings_) {
                        end = fy_get_stamp();
                        if (runs_ == 0) timingData_[idx] = 0;
                        timingData_[idx] += fy_elapsed_micros(start, end);
                    }
                    if ((writeResults_) && (!dl->isAsync())) {
                        // TODO (mw) missing handling of asynchronous stuff elsewhere
                        buf->write<float>(fname.c_str());
                    }
                } else if (!masked) {
                    //-------------------------------------------------------
                    // Handle (standard) GPU layers...
                    //-------------------------------------------------------
                    if (timings_) start = fy_get_stamp();
                    layer->forward(state.sequenceNo, stoken);
                    if (timings_) {
                        end = fy_get_stamp();
                        if (runs_ == 0) timingData_[idx] = 0;
                        timingData_[idx] += fy_elapsed_micros(start, end);
                    }
                    if (writeResults_) {
                        (dynamic_cast<GPULayerBase *>(layer))->writeResult(fname.c_str(), false);
                    }
                }
            }
#ifdef FYUSENET_MULTITHREADING
            asyncStateLock_.lock();
            if (deferredAsyncDependencies_.find(layer->getNumber()) != deferredAsyncDependencies_.end()) {
                //----------------------------------------------------------------
                // This layer was the last layer dependent on an upload layer. To
                // make sure that we do not overwrite the texture with a new
                // upload, we have to make sure that the GL pipeline has fully read
                // the texture, so we use the built-in fencing mechanism of GL
                //----------------------------------------------------------------
                auto it = asyncUploadDeferredDependencies_.begin();
                while (it != asyncUploadDeferredDependencies_.end()) {
                    if ((it->dependency == layer->getNumber() && (it->sequenceNo == state.sequenceNo))) {
                        gpu::UploadLayer * ul = it->provider;
                        //----------------------------------------------------------------
                        // Try to find a matching dependency on the early-stage side. A
                        // matching dependency means an upload for the next sequence that
                        // will have to wait before overwriting the textures. In case
                        // there is a match, decrement the dependency counter and upon
                        // reaching zero, remove the dependency, set the output textures
                        // accordingly and update the active upload for that layer. Note
                        // that this will not release the textures for re-use, it will just
                        // swap to a different texture set on the output side. The re-use
                        // is done via unlocking the upload layer from the fence wait...
                        //----------------------------------------------------------------
                        uint64_t replacementseq = 0;
                        for (auto eit = asyncUploadDependencies_.begin(); eit != asyncUploadDependencies_.end(); ++eit) {
                            if ((eit->provider == ul) && (eit->deferredNo == it->sequenceNo)) {
                                eit->count--;
                                replacementseq = eit->sequenceNo;
                                if (eit->count == 0) {
                                    //----------------------------------------------------
                                    // This part is executed when the same upload layer
                                    // has already uploaded the next data-set and is ready
                                    // to be used, in this case we can activate the new
                                    // texture output set and also remove the (early-stage)
                                    // dependency...
                                    //----------------------------------------------------
                                    int depend = eit->dependency;
                                    ul->swapOutputTextures(eit->sequenceNo);
                                    asyncUploadDependencies_.erase(eit);
                                    for (auto ite=asyncUploadWaiters_.begin(); ite != asyncUploadWaiters_.end() ; ++ite) {
                                        if ((ite->provider == ul) && (ite->sequenceNo == replacementseq) &&
                                            (ite->dependency == depend)) {
                                            pushReadyState(ite->state);         // asyncStateLock_ held
                                            asyncUploadWaiters_.erase(ite);
                                            break;
                                        }
                                    }
                                }
                                break;  // we do not expect more than one dependency in that part of the chain
                            }
                        }
                        activeUploadDependencies_[ul] = replacementseq;
                        it = asyncUploadDeferredDependencies_.erase(it);
                        //----------------------------------------------------------------
                        // Issue a fence here and kick-off a task to wait for the fence.
                        // This is to make sure that all texture consumers of an upload
                        // layer have executed and the texture can be safely re-used. The
                        // upload layer will be in a (partially) locked state until the
                        // fence passes...
                        //----------------------------------------------------------------
                        GLsync snc = context.issueSync();
                        auto thread = opengl::AsyncPool::getDerivedContextThread(context_);
                        thread->setTask(std::bind(&Engine::waitForUploadFence, this, thread.context(), snc, ul, SYNC_EXPIRY, state.sequenceNo));
                        numBackgroundTasks_++;              // the upload fence waiting above constitutes a background task
                        break;
                    } else ++it;
                }
            }
            asyncStateLock_.unlock();
#endif
        } // if (layer)
        ++(state.current);
    } // while
    runs_++;
    return state::DONE;
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Callback for asynchronous upload layers
 *
 * @param layer Upload layer that calls back in
 * @param sequenceNo Sequence number of the inference the UploadLayer was called in
 *
 * This function is invoked by asynchronous upload layers when the upload has finished and the
 * textures are ready to use, which is \e after the user-supplied callback on the upload has
 * been invoked. Note that at this point, the receiving layers will not necessarily have the
 * texture IDs set for this run, as a previous run might still be processing.
 *
 * In case this function detects that a dependency of the upload layer has been met, it
 * will swap the output texture set, such that it will be used for the upcoming run.
 *
 * @see UploadLayer::asyncUploadTask(), UploadLayer::swapOutputTextures()
 */
void Engine::uploadCallback(gpu::UploadLayer * layer, uint64_t sequenceNo) {
    assert(layer);
    upIssueLock_.lock();
    asyncStateLock_.lock();
    upIssueLock_.unlock();
    //-------------------------------------------------------
    // Adjust the dependencies and also check if there are
    // any states waiting on this upload.
    //-------------------------------------------------------
    auto it = asyncUploadDependencies_.begin();
    while (it != asyncUploadDependencies_.end()) {
        if (it->provider == layer && it->sequenceNo == sequenceNo) {
            //---------------------------------------------------
            // Decrement dependency counter since the upload is
            // done, note that an upload layer may have 2
            // dependencies...
            //---------------------------------------------------
            it->count--;
            if (it->count == 0) {
                //-------------------------------------------------------
                // All dependencies cleared, activate the output textures
                // for the pending sequence number
                //-------------------------------------------------------
                layer->swapOutputTextures(it->sequenceNo);
                assert(activeUploadDependencies_[it->provider] == it->sequenceNo);
                //-------------------------------------------------------
                // Check if there is a pending state for the resolved
                // dependency and move it to the ready list...
                //-------------------------------------------------------
                for (auto ite=asyncUploadWaiters_.begin(); ite != asyncUploadWaiters_.end() ; ++ite) {
                    if ((ite->provider == layer) && (ite->dependency == it->dependency)) {
                        pushReadyState(ite->state);
                        asyncUploadWaiters_.erase(ite);
                        break;
                    }
                }
                //-------------------------------------------------------
                // Delete dependency...
                //-------------------------------------------------------
                it = asyncUploadDependencies_.erase(it);
            } else ++it;
        }
        else ++it;
    }
    numBackgroundTasks_--;
    asyncStateLock_.unlock();
}
#endif




#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Wait for GL fence on the client side and unlock the pertaining upload layer
 *
 * @param ctx Link to GL context
 * @param sync GL sync ID to wait for
 * @param target Upload layer that should be unlocked
 * @param timeout Timeout (in nanoseconds) after the wait expires
 * @param sequenceNo Sequence number under which the upload was issued
 *
 * @throws FynException in case the sync times out
 *
 * This function is used to make sure that an upload layer is not re-used before its \e last
 * dependency in the layer chain. It will unlock the upload layer for the next operation. Note
 * that it is up to the internal implementation of the upload layer to support more than one
 * upload in flight (it can use multi-buffering internally).
 *
 * @note This function is not executed within the pipeline thread, it runs on a background thread
 */
void Engine::waitForUploadFence(const GfxContextLink& ctx, GLsync sync, gpu::UploadLayer *target, GLuint64 timeout, uint64_t sequenceNo) {
    //-------------------------------------------------------
    // Wait for the fence to appear on the GL pipeline and
    // then unlock the target UploadLayer...
    //-------------------------------------------------------
    bool rc = ctx.waitClientSync(sync, timeout);
    if (!rc) THROW_EXCEPTION_ARGS(FynException,"Timeout while waiting on GL client sync");
    ctx.removeSync(sync);
    asyncStateLock_.lock();    
    target->unlock(sequenceNo);
    //-------------------------------------------------------
    // If there are self-referential
    //-------------------------------------------------------
    for (auto it = asyncUploadWaiters_.begin(); it != asyncUploadWaiters_.end(); ++it) {
        if ((it->provider == target) && (it->dependency == target->getNumber())) {
            pushReadyState(it->state);       // asyncStateLock_ held
            asyncUploadWaiters_.erase(it);
            break;
        }
    }
    numBackgroundTasks_--;
    asyncStateLock_.unlock();
}
#endif



#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Callback for asynchronous download layers
 *
 * @param download Pointer to download type layer that was running asynchronously
 * @param sequenceNo Sequence number of the execution state that triggered the asynchronous download
 *
 * This function is invoked by asynchronous download layers when the download from the GPU has
 * finished. It updates the internal data structures (list of dependent layers, list of ready
 * states).
 *
 * @see pushReadyState(), #asyncDownloadDependencies_, #asyncDownloadWaiters_, #asyncStateLock_
 * @see DownloadLayer::readoutPBO(), DeepDownloadLayer::readoutPBO()
 */
void Engine::asyncDownloadDone(AsyncLayer *download, uint64_t sequenceNo) {
    assert(download);
    asyncStateLock_.lock();
    //-------------------------------------------------------
    // Erase the dependencies and also check if there are
    // any states waiting on this download. In case this was
    // the last layer blocking the state, move the state to
    // the ready list...
    //-------------------------------------------------------
    auto it = asyncDownloadDependencies_.begin();
    while (it != asyncDownloadDependencies_.end()) {
        if (it->provider == download && it->sequenceNo == sequenceNo) {
            assert(it->count == 1);
            auto ite = asyncDownloadWaiters_.begin();
            while (ite != asyncDownloadWaiters_.end()) {
                if ((ite->provider == download) && (ite->sequenceNo == sequenceNo) &&
                        (ite->dependency == it->dependency)) {
                    pushReadyState(ite->state);     // asyncStateLock_ held
                    ite = asyncDownloadWaiters_.erase(ite);
                } else ++ite;
            }
        }
        else ++it;
    }
    numBackgroundTasks_--;
    asyncStateLock_.unlock();
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Add a state that is ready-to-be-processed to the processing queue
 *
 * @param state State to add to the processing queue
 *
 * @pre #asyncStateLock_ is held by the calling thread
 *
 * This function pushes the supplied state to the processing queue that is regularly checked
 * by the engine thread. In case of ready-to-execute states with the same sequence number already
 * on the #readyStates_ list, this function checks if the state already on the list has a higher or
 * lower layer number and either replaces the state on the list or discards the supplied
 * \p state. This can happen with asynchronous dependencies, for example an upload layer that
 * supplied data to two different subsequent layers. In those cases, the lower layer number
 * should be the one that is used.
 *
 * @see #readyStates_, #asyncStateLock_
 */
void Engine::pushReadyState(const ExecutionState& state) {
    //------------------------------------------------------------
    // We do not want more than one pending execution state per
    // sequence number. If we already have a ready-to-execute state
    // for the same seqno, we check if this state has a lower layer
    // number. If that is the case, we do not push to the ready
    // states list, otherwise we replace the state on the ready
    // list with this one.
    //------------------------------------------------------------
    bool foundseq = false;
    for (auto & it : readyStates_) {
        if (it.sequenceNo == state.sequenceNo) {
            assert(it.sequenceNo >= engineSequence_);
            assert(foundseq == false);
            foundseq = true;
            if (state.current < it.current) {
                it.current = state.current;
            } else {
                assert(false);
            }
        }
    }
    if (!foundseq) {
        readyStates_.push_back(state);
        looperLock_.lock();
        pendingStates_++;
        looperLock_.unlock();
        looperWait_.notify_one();
    }
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Update the minimum dependency number list for a sequence
 *
 * @param sequence Sequence number to update for
 *
 * This function traverses the waiting states for a given sequence number and records the lowest
 * layer number of all waiting states, which is then entered into the #minimumWaitingDependency_
 * map.
 *
 * @see minimumWaitingDependency_
 */
void Engine::updateWaitingLayers(uint64_t sequence) {
    std::lock_guard<std::recursive_mutex> lck(asyncStateLock_);
    int minlayer = 0;
    for (auto it = asyncDownloadWaiters_.begin(); it != asyncDownloadWaiters_.end(); ++it) {
        if (it->sequenceNo == sequence) minlayer = std::min(it->dependency, minlayer);
    }
    for (auto it = asyncUploadWaiters_.begin(); it != asyncUploadWaiters_.end(); ++it) {
        if (it->sequenceNo == sequence) minlayer = std::min(it->dependency, minlayer);
    }
    if (minlayer == 0) minimumWaitingDependency_.erase(sequence);
    else minimumWaitingDependency_[sequence] = minlayer;
}
#endif


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Engine background thread which performs processing in multi-threaded configurations
 *
 * @param context Link to GL context that is current to the looper thread
 *
 * This function runs a looper which waits for pending states on the #readyStates_ queue and
 * pushes these states through the execute() function of the engine, making sure that all
 * states are processed down to the last layer. It communicates with the rest of the engine
 * via condition variables.
 */
// TODO (mw) more docs
void Engine::looper(const GfxContextLink & context) {
    std::unique_lock<std::mutex> locke(looperLock_);
    while (!quit_) {
        // ---------------------------------------------------
        // Wait until we get some work assigned...
        // ---------------------------------------------------
        looperWait_.wait(locke, [this]() { return (pendingStates_ > 0); });
        if (pendingStates_ <= 1 && quit_) break;    // the quit signal is a pending state (kinda)
        locke.unlock();
        // ---------------------------------------------------
        // Fetch state to process and check if this state is
        // OK to run (must match the lowest waiting layer
        // number)....
        // ---------------------------------------------------
        asyncStateLock_.lock();
        ExecutionState estate = readyStates_.front();
        readyStates_.pop_front();
        auto minit = minimumWaitingDependency_.find(estate.sequenceNo);
        int lowestwait = (minit != minimumWaitingDependency_.end()) ? minit->second : 0;
        bool discard = false;
        // NOTE (mw) maybe we should the sequence lock here, also if we discard because of old sequence, we should update the minlist
        if ((estate.sequenceNo <= engineSequence_) || ((lowestwait > 0) && (estate.current.layer() != lowestwait))) {
            discard = true;
        }
        asyncStateLock_.unlock();
        locke.lock();
        pendingStates_--;
        locke.unlock();
        if (!discard) {
            updateWaitingLayers(estate.sequenceNo);
            state rc = execute(estate, context);
            if (rc == state::DONE) {
                sequenceLock_.lock();
                engineSequence_ = estate.sequenceNo;
                sequenceDone_.notify_one();
                sequenceLock_.unlock();
                if (sequenceCallback_) sequenceCallback_(estate.sequenceNo);
            } else if (rc == state::NET_ERROR) {
                // TODO (mw) handle error here
            }
        }
        locke.lock();
    }
}
#endif

} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
