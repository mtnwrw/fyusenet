//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep-tensor Type-Casting Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../base/bufferspec.h"
#include "deepfunctionlayer.h"
#include "../castlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::deep {


/**
 * @brief Type-casting emulation for deep-tensor data
 *
 * This layer \e emulates a type-casting operation by either performing no operation at all, or
 * by performing a rounding and clamping operation on the tensor data. Though OpenGL(ES) supports
 * integer datatypes, rendering to integer textures is not guaranteed to be supported on mainstream
 * (embedded) GPUs as of time of writing. For this reason, even though "type-casting" is emulated
 * by this layer, the result will still be a floating-point texture.
 *
 * Note that remaining in floating-point representation has an impact on the range of integer
 * numbers that can be represented. Especially when using 16-bit floating-point numbers, do not
 * rely on an exact representation of any integer. As a rule-of-thumb, representing any integer
 * number that requires more than 10 bits (excluding the sign bit) in a 16-bit floating-point
 * representation will become problematic.
 *
 * @warning The output of this layer will still be a floating-point texture, possibly 16-bit float,
 *          be aware of range errors with this internal data-type.
 *
 * @todo Think about forcing the texture to be used as output by this layer to 32-bit FP to alleviate
 *       the integer accuracy problem.
 *
 * @see GPULayerBase::TEXTURE_IFORMAT_4, GPULayerBase::TEXTURE_FORMAT_4, GPULayerBase::TEXTURE_TYPE_DEFAULT
 * @see GPULayerBase::TEXTURE_PIXTYPE
 */
class DeepCastLayer : public DeepFunctionLayer {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepCastLayer(const CastLayerBuilder & builder,int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void cleanup() override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders() override;
    void renderChannelBatch() override;
    void beforeRender() override;
    void afterRender() override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    programptr shader_;           //!< Shader program for the casting
    unistateptr shaderState_;     //!< UniformState object for the #shader_
    CastTarget target_;
};

} // fyusion::fyusenet::gpu::deep namespace


// vim: set expandtab ts=4 sw=4:
