//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Network Execution Engine (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <cstdint>
#include <list>
#include <atomic>
#include <functional>
#include <condition_variable>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerbase.h"
#include "asynclayerinterface.h"
#include "compiledlayers.h"
#include "../gpu/gpulayerbase.h"
#include "../gpu/downloadinterface.h"
#include "../gpu/gfxcontexttracker.h"
#ifdef FYUSENET_MULTITHREADING
#include "../gl/asyncpool.h"
#endif



namespace fyusion::fyusenet {

//------------------------------------- Public Declarations ----------------------------------------

namespace gpu {
    class UploadLayer;
}

class NeuralNetwork;

/**
 * @brief Neural network inference engine main dispatcher
 *
 * This class contains the inference engine driver code. It basically just iterates through the
 * layers of the network in ascending order and calls the LayerBase::forward() method on each
 * and every layer. The most complex part is the handling of "asynchronous" layers in multi-
 * threaded build configurations, which are upload or download layers with asynchronous processing
 * enabled. These layers - in particular the upload layers - can roll over from one run to the
 * next and need to be handled by an additional engine thread that pushes the execution states
 * forward.
 *
 * In multi-threaded (asynchronous) scenarios, the engine uses a queueing mechanism which tracks
 * the execution state and defers/resumes operation after dependencies of asynchronous layers have
 * been met.
 *
 * @note When using the engine, it is highly recommended to do so from a single thread.
 *
 * @todo The engine code is quite messy due to several revisions and changes in the underlying
 *       execution modalities. It should be refactored to simplify the code (especially the
 *       asynchronous operation part)
 *
 * @see Engine::ExecutionState, Engine::Dependency
 */
class Engine : public fyusenet::GfxContextTracker {
    // TODO (mw) the code in this class is rather messy and would benefit from a refactoring
 public:
    enum execstate : int8_t {
        EXEC_ERROR = -1,
        EXEC_DONE = 0,
        EXEC_DEFERRED = 1,
        EXEC_STOPPED = 2
    };
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit Engine(const GfxContextLink& context = GfxContextLink(), bool async=false);
    ~Engine() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    execstate forwardLayers(StateToken * state  = nullptr);
    void finish();
    void resetTimings();
    void enableIntermediateOutput(const std::string& outputDir);
    void disableIntermediateOutput();
    void enableTimings();
    void disableTimings();
    void setup(NeuralNetwork *net);
    void cleanup(const std::function<void()> & broom);

    /**
     * @brief Obtain timings on a per-layer basis
     *
     * @return Map which maps the layer index to a cumulative microsecond execution time (on the
     *         CPU)
     *
     *  This function returns a map that maps a layer index to a cumulative timings, taken in
     *  microseconds, when timings are enabled.
     *
     * @note Those timings are not necessarily indicative of the actual device (e.g. GPU) timings
     *       and should be taken with a couple of spoons of salt.
     *
     * @warning This function is not thread-safe, and it should not be run concurrently to the
     *          forward() method
     *
     * @see enableTimings(), disableTimings(), resetTimings()
     */
    const std::unordered_map<int, uint32_t> & getTimings() const {
        return timingData_;
    }

    /**
     * @brief Retrieve the last sequence number that was issued
     *
     * @return Last issued sequence number
     *
     * This returns the last sequence numbers that was issued. All sequence numbers (also called
     * sequence IDs) are greater than zero.
     *
     * @note The combination of calling forwardLayers() and then retrieving the sequence number
     *       that was issued by that call is not thread-safe. It is up to the caller to ensure
     *       that the engine is locked or that all engine calls are made from the same thread,
     *       the latter being the recommended mode of operation.
     */
    uint64_t lastSequenceNo() const {
        return sequenceNo_ - 1;
    }

    /**
     * @brief Retrieve the next sequence number to be issued
     *
     * @return Sequence number that will be issued with the next call to forwardLayers()
     *
     * This returns the next sequence number to be issued. All sequence numbers (also called
     * sequence IDs) are greater than zero.
     */
    uint64_t nextSequenceNo() const {
        return sequenceNo_;
    }


    /**
     * @brief Register network layer set for inference by this engine
     *
     * @param layers Set of layers returned by the LayerFactory instance
     *
     */
    void setLayers(const CompiledLayers& layers) {
        layers_ = layers;
    }


    /**
     * @brief Retrieve (writable) reference to layer set currently registered to this engine
     *
     * @return Reference to CompiledLayers object that is registered with this engine
     */
    CompiledLayers & getLayers() {
        return layers_;
    }


#ifdef FYUSENET_MULTITHREADING
    /**
     * @brief Set callback function that is invoked when a sequence has been fully executed
     *
     * @param callback Callback function that should be invoked when a sequence has been processed
     *
     * Registers a callback that is invoked whenever the engine (thread) has completed a full run
     * of the network
     *
     * @see #sequenceCallback_, looper()
     */
    void setSequenceCallback(const std::function<void(uint64_t)> & callback) {
        sequenceCallback_ = callback;
    }

    /**
     * @brief Set callback function when a new sequence number/ID has been issued for processing
     *
     * @param callback Callback function that should be invoked when a new sequence number was issued
     *
     * Registers a callback that is invoked whenever a new sequence number is issued (prior to it being
     * processed).
     *
     * @see #newSeqCallback_, forwardLayers()
     */
    void setNewSequenceCallback(const std::function<void(uint64_t)> & callback) {
        newSeqCallback_ = callback;
    }
#endif

 private:
    /**
     * @brief Execution status indicator for engine operation
     */
    enum state : uint8_t {
         DONE = 0,        //!< Network was fully executed (async ops may still be pending, but the net did a full run)
         UPLOADING,       //!< Network was not fully executed and is performing an asynchronous upload
         DOWNLOADING,     //!< Network is waiting for a download to finish
         NET_ERROR        //!< There was an error during the network execution
    };

    /**
     * @brief Compound structure that is used for memorizing the execution state of the pipeline
     *
     * This structure is used by the Engine in order to memorize execution states in cojunction
     * with asynchronous layer execution.
     */
    struct ExecutionState {
        /**
         * @brief Empty constructor
         */
        ExecutionState(StateToken * state = nullptr) : state_(state) {}

        /**
         * @brief Construct state object with sequence number and iterator
         *
         * @param seq Sequence number of the run this state encodes for
         * @param iter Position in the layer list to execute from
         * @param state Optional StateToken object that controls individual run layer behaviour
         */
        ExecutionState(uint64_t seq, const CompiledLayers::iterator& iter, StateToken *state = nullptr) :
            state_(state), sequenceNo(seq), current(iter) {}

        /**
         * @brief Create a clone of the current execution state
         *
         * @return Cloned state
         */
        [[nodiscard]] ExecutionState clone() const {
            ExecutionState dolly(sequenceNo, current);
            return dolly;
        }

        StateToken * state_ = nullptr;              //!< Pointer to object that keeps track of the state for stateful networks
        uint64_t sequenceNo = 0;                    //!< Sequence number of the run this state encodes for
        CompiledLayers::iterator current;           //!< Iterator for layer position at which the state shall execute
    };

    /**
     * @brief Encode dependency of a layer on an asynchronous layer
     *
     * This compound structure stores a dynamic dependency of a (synchronous) layer on an asynchronous
     * layer (upload or download). Dynamic dependencies include the sequence number of the run, as
     * dependencies are resolved during a run and are deleted from the list once the count reaches
     * zero.
     *
     * For upload layers we differentiate in an "early stage" and "deferred" dependency. The
     * former is simply the lowest-numbered layer getting its input from an asynchronous upload
     * layer, having to wait until the upload was pushed to the GL pipeline. The latter is also a
     * layer that takes input from an upload, but it has the highest number of all the dependent
     * layers. Keeping track of the last dependent layer is important to know when the upload
     * layer may switch the texture IDs, since it uses multiple textures internally for streamlining
     * purposes. A deferred dependency can also affect the upload layer itself running in the next
     * run (next sequence), as a new upload should not be started before the last upload has been
     * consumed.
     */
    template<class T>
    struct Dependency {
        explicit Dependency<T>(int dep=0, T * prov=nullptr, uint8_t cnt=1, uint64_t seq=0) :
            dependency(dep), sequenceNo(seq), provider(prov), count(cnt) {
        }
        int dependency = 0;                         //!< Number of the dependent layer
        uint64_t sequenceNo = 0;                    //!< Sequence number of the inference run
        uint64_t deferredNo = 0;                    //!< Sequence number of a previous run using the same upload layer
        T * provider = nullptr;                     //!< Provider layer that the dependent layer waits on
        uint8_t count = 1;                          //!< Dependency counter, if it reaches zero, the dependency can be removed
    };


    /**
     * @brief Compound structure for a state that is waiting to be released
     *
     * This structure encodes a state that is actively waiting for all the dependencies to
     * be resolved so that it can be pushed on the #readyStates_ queue.
     */
    template<class T>
    struct WaitingState {
        WaitingState(int dep, T * layer, uint64_t seq, const ExecutionState& st) :
            dependency(dep), sequenceNo(seq), provider(layer), state(st) {}
        int dependency = 0;                         //!< Number of the dependent (waiting) layer
        uint64_t sequenceNo = 0;                    //!< Sequence number of the inference run that it is waiting for
        T * provider = nullptr;                     //!< Provider layer that the dependent layer waits on
        ExecutionState state;                       //!< Actual state that is pending execution
    };

    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    state execute(ExecutionState& state, const GfxContextLink & context);
#ifdef FYUSENET_MULTITHREADING
    void waitForUploadFence(const GfxContextLink& ctx, GLsync sync, gpu::UploadLayer *target, GLuint64 timeout, uint64_t sequenceNo);
    void uploadCallback(gpu::UploadLayer * layer, uint64_t cbSequenceNo);
    void pushReadyState(const ExecutionState& state);
    void asyncDownloadDone(AsyncLayer *target, uint64_t sequenceNo);
    void updateWaitingLayers(uint64_t sequence);
    void looper(const GfxContextLink & context);
#endif

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------ ------------------
    uint64_t sequenceNo_ = 1;        //!< Sequence number (is strictly monotonous and starts at 1), see #sequenceLock_
    uint32_t runs_ = 0;              //!< Number of runs since last timing reset
    std::string outputDir_;          //!< Output directory where to write intermediate (layer-by-layer) results to
    std::mutex runGuard_;            //!< Simple guard to create partial thread-safety
    bool writeResults_ = false;      //!< Flag that controls if intermediate (layer-by-layer) results should be written to disk for debugging purposes
    bool timings_ = false;           //!< Flag that controls whether or not \b CPU timings should be kept on a layer-by-layer basis
    bool setup_ = false;             //!< Indicator if engine was setup
    CompiledLayers layers_;          //!< Set of runnable layers generated by the network-specific code

    /**
     * Timing data on a per-layer basis. Index is the layer number and the values are the timings
     * per layer given in microseconds.
     */
    std::unordered_map<int, uint32_t> timingData_;

#ifdef FYUSENET_MULTITHREADING

    std::mutex looperLock_;                     //!< Looper runtime lock, used in conjunction with #looperWait_
    std::condition_variable looperWait_;        //!< Condition that the looper waits on for new states being pushed to #readyStates_
    int numBackgroundTasks_ = 0;                //!< Tracks the number of background tasks (for upload / download)
    int pendingStates_ = 0;                     //!< Number of states in the #readyStates_ list, @see #looperLock_
    std::mutex sequenceLock_;                   //!< Lock that is used in conjunction with #sequenceDone_ and #engineSequence_
    uint64_t engineSequence_ = 0;               //!< Highest sequence number that has been completed by the engine
    bool async_ = false;                        //!< Flag that indicates if the engine shall run asynchronously
    std::mutex upIssueLock_;                    //!< Lock for issueing asynchronous upload operations

    /**
     * @brief asyncStateLock_
     */
    std::recursive_mutex asyncStateLock_;

    /**
     * List of states that are ready to be executed by the engine thread
     *
     * @see asyncStateLock_, looper(), pushReadyState()
     */
    std::list<ExecutionState> readyStates_;

    /**
     * GL thread that runs the looper()
     */
    opengl::AsyncPool::GLThread exec_;

    /**
     * @brief Bookkeeping for deferred-stage asynchronous upload dependencies
     *
     * This keeps track of deferred-stage dependencies introduced by asynchronously-operating upload
     * layers. A deferred stage dependency is basically the last layer in the network chain that depends
     * on an output from an upload layer. The reason why this is requires  on upload layers is that
     * not only the layer that performs the asynchronous upload is to be waited on (technically not
     * fully true), but the layers \e consuming those textures and writing their results to other
     * textures.
     *
     * @note The data structure used here does not offer quick random access, however we usually do
     *       not encounter more than a handful of these dependency types in a network and we can
     *       therefore live with the worse complexity here.
     *
     * @see #asyncUploadDependencies_, waitForUploadFence()
     */
    std::list<Dependency<gpu::UploadLayer>> asyncUploadDeferredDependencies_;

    /**
     * @brief Bookkeeping for early-stage asynchronous upload dependencies
     *
     * This keep track of the set of first layers that are dependent on an asynchronous upload
     * layer. These layers cannot execute as long as the asynchronous upload is not issued to the
     * GL pipeline. Once it is on the pipeline, GL will keep track of the dependencies there.
     *
     * @note The data structure used here does not offer quick random access, however we usually do
     *       not encounter more than a handful of these dependency types in a network and we can
     *       therefore live with the worse complexity here.
     *
     * @see uploadCallback(), #asyncStateLock_
     */
    std::list<Dependency<gpu::UploadLayer>> asyncUploadDependencies_;

    /**
     * @brief Bookkeeping for asynchronous download dependencies
     *
     * This list keeps track of layers that are dependent on (currently running) asynchronous
     * downloads. Those layers are blocked from execution until the download layer they are
     * dependent on have completed.
     *
     * @see asyncDownloadDone(), #asyncStateLock_
     *
     * @note The data structure used here does not offer quick random access, however we usually do
     *       not encounter more than a handful of these dependency types in a network and we can
     *       therefore live with the worse complexity here.
     */
    std::list<Dependency<AsyncLayer>> asyncDownloadDependencies_;

    /**
     * @brief States that are waiting for a download to complete
     *
     * This list features states that have encountered an unresolved dependency on a download
     * layer and are therefore waiting for the dependency to be resolved and then pushed to the
     * #readyStates_ queue.
     *
     * @see asyncDownloadDone(), #asyncStateLock_
     *
     * @note The data structure used here does not offer quick random access, however we usually do
     *       not encounter more than a handful of these dependency types in a network and we can
     *       therefore live with the worse complexity here.
     */
    std::list<WaitingState<AsyncLayer>> asyncDownloadWaiters_;

    /**
     * @brief States that are waiting for an upload to complete
     *
     * This list features states that have encountered an unresolved dependency on an upload
     * layer and are therefore waiting for the dependency to be resolved and then pushed to the
     * #readyStates_ queue.
     *
     * @see uploadCallback(), #asyncStateLock_
     *
     * @note The data structure used here does not offer quick random access, however we usually do
     *       not encounter more than a handful of these dependency types in a network and we can
     *       therefore live with the worse complexity here.
     */
    std::list<WaitingState<gpu::UploadLayer>> asyncUploadWaiters_;

    /**
     * @brief Static set of layer numbers that have a dependency on an asynchronous layer
     *
     * In order to quickly determine whether or not a layer is subject to waiting, this set
     * may be queried for the layer number.
     */
    std::unordered_set<int> asyncDependencies_;

    /**
     * @brief Static set of layer numbers that have a deferred dependency on an upload layer
     *
     * In order to quickly determine whether or not a layer is subject to waiting, this set
     * may be queried for the layer number.
     *
     * @see Dependency
     */
    std::unordered_set<int> deferredAsyncDependencies_;

    /**
     * Maps an upload layer to a sequence number whenever the upload layer is \e engaged in either
     * the actual upload or still providing the uploaded data to subsequent layers. Once an upload
     * layer has no more dependent layers in a single run, the sequence number it maps to will be
     * reset to 0 (invalid sequence number).
     *
     * @see #asyncStateLock_
     */
    std::unordered_map<gpu::UploadLayer *, uint64_t> activeUploadDependencies_;


    /**
     * @brief Maps sequence numbers to the lowest layer number in the waiting queues
     *
     * This map keeps track of the lowest layer number within a sequence that is waiting for an
     * asynchronous dependency to resolve.
     *
     * @see looper()
     */
    std::unordered_map<uint64_t, int> minimumWaitingDependency_;


    /**
     * Flag for asynchronous operation that (if set) instructs the looper to terminate.
     *
     * @see #looperLock_, #runGuard_
     */
    bool quit_ = false;

    /**
     * Condition that will be notified by the looper or the forwarding when an inference run has completed.
     *
     * @see #sequenceLock_, #engineSequence_
     */
    std::condition_variable sequenceDone_;

    /**
     * Optional callback function that is invoked by the looper if a sequence has been fully processed.
     *
     * @note A fully processed sequence does not mean that \e all layers are done. Asynchronous download layers
     *       could still be running in the background.
     */
    std::function<void(uint64_t)> sequenceCallback_;

    /**
     * Optional callback function that is invoked (not by the looper) when a new sequence is about to be
     * registered with the looper.
     */
    std::function<void(uint64_t)> newSeqCallback_;

#endif
};


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
