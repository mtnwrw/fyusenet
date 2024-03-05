//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep to Shallow Layer Converter (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/fbo.h"
#include "../base/bufferspec.h"
#include "deep/deepfunctionlayer.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu {

/**
 * @brief Convert deep tensor format to shallow tensor format
 *
 * FyuseNet differentiates between tensors with a low channel count, which we call \e shallow tensors,
 * and tensors with a high channel count, which we call \e deep tensors. The specifics for those
 * are laid out @ref GPULayerBase "here". The purpose of this class it to convert the tensor data
 * from the \e deep representation format to the \e shallow representation format.
 *
 * This class supports multiple render targets in order to minimize the number of render passes.
 */
class Deep2ShallowLayer : public deep::DeepLayerBase {
 public:

    enum {
        UNIFORM_MRT = 1         //!< Index in the uniforms for the MRT flag (see deep2shallow.frag)
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    Deep2ShallowLayer(const GPULayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void cleanup() override;
    void forward(uint64_t sequence, StateToken * state) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupNetworkPolygons(VAO *vao);
    void setupShaders();
    virtual void setupFBOs() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO * vertexArray_ = nullptr;   //!< Pointer to vertex array object that tracks the buffer objects
    VBO * posBuffer_ = nullptr;     //!< Pointer to vertex buffer object for polygon vertices
    VBO * attr0Buffer_ = nullptr;   //!< Pointer to vertex buffer object for render target #0 texture coordinates
    VBO * attr1Buffer_ = nullptr;   //!< Pointer to vertex buffer object for render target #1 texture coordinates
    VBO * attr2Buffer_ = nullptr;   //!< Pointer to vertex buffer object for render target #2 texture coordinates
    VBO * attr3Buffer_ = nullptr;   //!< Pointer to vertex buffer object for render target #3 texture coordinates
    IBO * indexBuffer_ = nullptr;   //!< Pointer to index buffer object that defines the polygon connectivity
    programptr shader_;             //!< Shader program for the conversion
    std::vector<int> MRT_;          //!< Number of render targets per pass
    int maxRenderTargets_ = 1;      //!< Maximum number of simultaneous render targets for this device
};

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
