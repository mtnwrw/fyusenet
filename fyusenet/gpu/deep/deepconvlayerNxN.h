//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ odd-sized NxN mask and N > 3 (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <memory>
#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../gfxcontextlink.h"
#include "../../base/bufferspec.h"
#include "../uniformweightarray.h"
#include "deepconvlayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::deep {

/**
 * @brief NxN convolution layer for deep tensor format
 *
 * This class implements a deep-tensor 2D convolution as laid out in deep::DeepConvLayerBase
 * for odd kernel sizes of odd sizes equal or larger than 3x3 running on the GPU.
 */
class DeepConvLayerNxN : public DeepConvLayerBase {
 public:
    constexpr static int BASE_VECTORS = 2;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepConvLayerNxN(const ConvLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void cleanup() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void compileConvolutionShaders(const char *preproc) override;
    unistateptr initShader(programptr shader, int horizOffset, int kernelOffset, int kernelY);
    void partialRender();
    void nonPartialRender();

    static void appendOffsetDefs(char *string, int kernel, size_t maxChars);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int maxVectors_ = 0;                            //!< Maximum varying vectors (4-vec entities) that can be passed from vertex to fragment shader
    int maxKernelWidth_ = 0;                        //!< Maximum kernel width that can be handled based on #maxVectors_
    int numSplits_ = 0;                             //!< Number of horizontal kernel splits required to work on target hardware
    std::vector<int> horizSplits_;                  //!< Size of partial kernels after horizontal split
    bool partialConv_ = false;                      //!< Indicator if the convolution done here requires multiple render passes due to splitting of the kernel
    std::vector<programptr> shaders_;               //!< Convolution shader programs
    std::vector<programptr> noBiasShaders_;         //!< Convolution shader programs that do not include the network bias
    std::vector<unistateptr> shaderStates_;         //!< Uniform-variable state for #shaders_
    std::vector<unistateptr> noBiasShaderStates_;   //!< Uniform-variable state for #noBiasShaders_

};

} // fyusion::fyusenet::gpu::deep namespace


// vim: set expandtab ts=4 sw=4:
