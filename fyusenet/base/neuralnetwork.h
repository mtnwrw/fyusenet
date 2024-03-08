//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------
#pragma once

//-------------------------------------- System  Headers -------------------------------------------

#include <cstddef>
#include <cstdint>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "engine.h"
#include "layerfactory.h"
#include "buffermanager.h"
#include "compiledlayers.h"
#ifdef FYUSENET_GL_BACKEND
#include "../gl/gl_sys.h"
#endif

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet {

namespace gpu {
    class GPULayerBase;
}

namespace cpu {
    class CPULayerBase;
}

/**
 * @brief Base class for neural networks
 *
 * This class serves as base for neural network representations used in FyuseNet. It encapsulates
 * a set of layers which are executed by an internal Engine instance using the forward() call
 * (resembling PyTorch in that respect). In order to use this class, it has to be derived for
 * every particular type of neural net and all its pure virtual functions need to be implemented
 * by the derived class, which are:
 *   - buildLayers()
 *   - connectLayers()
 *   - initializeWeights()
 *
 * To use such a derived network instance, the following steps should be taken:
 *  1. Create an OpenGL context and make it current to the calling thread
 *  2. Instantiate derived network class
 *  3. Call setup() on the network object
 *  4. Do network-specific preparations (set inputs etc.)
 *  5. Call forward() on the network object
 *  6. Repeat 5 ad nauseam
 *  7. Call cleanup() on the network object
 *  8. Delete network instance
 *  9. Take down GL context (if appropriate)
 *
 * The example above illustrates simple \e synchronous operation. Due to the lag when uploading or
 * downloading textures, a neural network also allows for \e asynchronous operation. The way that
 * this is implemented in the underlying Engine is by means of a command queue which needs to be
 * manually pushed. Every call to forward() triggers the execution of the command queue while the
 * only work that (automatically created) threads are performing is to wait and signal for
 * synchronization primitives and copy data from/to CPU buffers. Execution of the computational
 * network layers will not be deferred to background threads to provide more control over when
 * to execute computationally heavy functions. See the forward() function and the Engine
 * documentation for more details.
 *
 * @see Engine, Engine::forwardLayers
 */
class NeuralNetwork : public GfxContextTracker {
    friend class Engine;
 public:
    using state = Engine::execstate;

     /**
     * @brief Aggregate for returning execution state and sequence number
     */
    struct execstate {
        state status;                   //!< Status code for the run
        uint64_t sequenceNo = 0;        //!< Sequence number that was issued for the run
    };

#ifdef FYUSENET_MULTITHREADING
    /**
     * @brief Compound class for specification of callback functions for asynchronous operation
     *
     * This class aggregates a set of callbacks that can be used for asynchronous communication
     * with the network.
     */
    class AsyncAdapter {
     public:
        AsyncAdapter() = default;
        /**
         * @brief Set callback function to be invoked when a new sequence number has been issued
         *
         * @param callback Callback function that is invoked when a new sequence number has been issued
         *                 prior to execution of the sequence
         *
         * @return Reference to self (current object)
         *
         * The provided \p callback function will be called (from the same thread, so be aware of locks
         * held in your code) when a new sequence number has been issued.
         */
        AsyncAdapter & newSequence(const std::function<void(uint64_t)> & callback) {
            newSeq_ = callback;
            return *this;
        }

        /**
         * @brief Set callback function to be invoked when a sequence has been processed
         *
         * @param callback Callback function that is invoked (from the engine thread) when a sequence
         *                 has been processed
         *
         * @return Reference to self (current object)
         *
         * The provided \p callback function will be called (from a different thread) when a single run or
         * "sequence" has been completed, providing the sequence number of the completed run. Though the
         * completion of a sequence means that all layers have been started, it does not include the
         * completion of asynchronous GPU downloads. It is up to the individual network implementation
         * to provide callbacks for those.
         */
        AsyncAdapter & sequenceDone(const std::function<void(uint64_t)> & callback) {
            seqDone_ = callback;
            return *this;
        }

        /**
         * @brief Set callback function to be invoked when a download has been completed
         *
         * @param callback Callback function that is invoked (from the engine thread) when a download
         *                 has been completed
         *
         * @return Reference to self (current object)
         *
         * The provided \p callback function will be called (from a different thread) when a download
         * layer has completed a download and buffer data is available. Depending on the
         * implementation of the individual network, the buffer data may change after the callback
         * has been invoked, please consult the subclassed network implementation on how it behaves.
         * The parameters supplied to the callback are the name of the download layer, a sequence
         * number and a pointer to the CPUBuffer that has been filled with data. Please consider
         * the callback itself as time-sensitive and do not perform long computations in it.
         */
        AsyncAdapter & downloadReady(const std::function<void(const std::string&, uint64_t, cpu::CPUBuffer *)> & callback) {
            downReady_ = callback;
            return *this;
        }

        /**
         * @brief Set callback function to be invoked when an upload has been completed
         *
         * @param callback Callback function that is invoked (from the engine thread) when an
         *                 upload has been completed
         *
         * @return Reference to self (current object)
         *
         * The provided \p callback function will be called (from a different thread) when an upload
         * layer has completed copying the supplied CPU buffer into a GL buffer and can be
         * provided with new data.
         * The parameters supplied to the callback are the name of the upload layer and a sequence
         * number. Please consider the callback itself as time-sensitive and do not perform long
         * computations in it.
         */
        AsyncAdapter & uploadReady(const std::function<void(const std::string&, uint64_t)> & callback) {
            upReady_ = callback;
            return *this;
        }

        std::function<void(uint64_t)> newSeq_;
        std::function<void(uint64_t)> seqDone_;
        std::function<void(const std::string&, uint64_t, cpu::CPUBuffer *)> downReady_;
        std::function<void(const std::string&, uint64_t)> upReady_;
    };
#endif

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit NeuralNetwork(const GfxContextLink & ctx = GfxContextLink());
    ~NeuralNetwork() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup();
    virtual void setup();
    execstate forward();
    virtual execstate forward(StateToken * token);
    virtual execstate finish();
#ifdef FYUSENET_MULTITHREADING
    virtual void asynchronous(const AsyncAdapter & adapter = AsyncAdapter());
#endif

    /**
     * @brief Obtain sequence number that will be issued with the next call to forward()
     *
     * @return Sequence number that will be issued with the next call to forward
     *
     * The return value of this function comes in handy during asynchronous operations, for example
     * to set input buffers for the next run and make sure that there are no clashes.
     */
    [[nodiscard]] uint64_t nextSequenceNo() const {
        return (engine_) ? engine_->nextSequenceNo() : 0;
    }

    /**
     * @brief Obtain sequence number that was issued by the last call to forward()
     *
     * @return Sequence number that was issued by the last call to forward()
     *
     * @see forward(), Engine::lastSequenceNo()
     */
    [[nodiscard]] uint64_t lastSequenceNo() const {
        return (engine_) ? engine_->lastSequenceNo() : 0;
    }


 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual CompiledLayers gpuSetup();
    std::shared_ptr<LayerFactory> getLayerFactory(compute_device dev = compute_device::DEV_GPU);

    /**
     * @brief Initialize all weights in weight-bearing layers
     *
     * This function loads weights/biases and similar parameters into the individual layers
     * that need them. If no weights have been supplied to the network at time of initialization,
     * the network will load all-zeros into the affected layers.
     *
     * This function has to be implemented in subclasses.
     */
    virtual void initializeWeights(CompiledLayers& layers) = 0;

    /**
     * @brief Assemble and compile layers of the network
     *
     * @return List of compiled layers, generated by the LayerFactory
     *
     * This function creates all layers of the network by instantiating a set of builders and
     * pushing those to a LayerFactory instance, which will then compile the layers into a
     * runnable layer set which is then returned by this function.
     *
     * This function has to be implemented in subclasses.
     */
    virtual CompiledLayers buildLayers() = 0;


    /**
     * @brief Establish connectivity between layers
     *
     * @param layers
     * @param buffers
     *
     * @pre Layers have been built by using buildLayers()
     */
    virtual void connectLayers(CompiledLayers & layers, BufferManager * buffers) = 0;

#ifdef FYUSENET_GL_BACKEND
    static fyusion::opengl::FBO * getFBO(const gpu::GPULayerBase * layer, int index=0);
#endif

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
#ifdef FYUSENET_MULTITHREADING
    bool async_ = false;                              //!< Indicator if network runs asynchronously
    AsyncAdapter asyncCallbacks_;                     //!< Optional callbacks for asynchronous operation
#endif
    Engine * engine_ = nullptr;                       //!< Pointer to execution engine
    BufferManager * bufferMgr_ = nullptr;             //!< Texture/buffer manager TODO (mw) move buffermanager out of the network
    bool setup_ = false;                              //!< Indicator if network was set up
};


} // fyusion::fyusenet namespace


// vim: set expandtab ts=4 sw=4:
