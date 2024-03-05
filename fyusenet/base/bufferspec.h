//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Texture (and Buffer) Specification (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "layerflags.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet {

/**
 * @brief Computation buffer specification
 *
 * Computation buffers are used as two-sided buffers that store the results of a layer computation
 * and make it available to subsequent layers. These buffers always follow a simple policy that
 * there is at max one writer, and at min one reader to the buffer.
 *
 * The buffer specification is used to query the buffer manager for actual buffers that fulfill
 * the query criteria that are provided in the BufferSpec object. As such, the specification is
 * not the buffer itself but a descriptor for the buffer.
 *
 * Due to the way that data is laid out, network layers \e may require a set of buffers to handle
 * data that has more than 4 channels (in the GPU case). Using multiple textures for high-channel
 * buffers is accomplished by using the #channelIndex_. Providing a specifier with a channel index
 * of 0 refers to the first 4 channels in the GPU buffer case, whereas a channel index of 2 refers
 * to channel 8..11 (inclusive).
 *
 * In order to support layer types that have multiple inputs, the buffer specifier uses the #port_
 * to determine which input facility of a layer is to be used. Take for example a concatenation
 * layer that is supposed to concatenate the results of 3 layers. This layer will have 3 ports,
 * numbered 0..2 .
 *
 * In most of the cases, the layers themselves are responsible for generating the buffer specifiers.
 * This is done in LayerBase::getRequiredInputBuffers() and LayerBase::getRequiredOutputBuffers().
 *
 * @see LayerBase::getRequiredInputBuffers(), LayerBase::getRequiredOutputBuffers()
 *
 * @note Historically the BufferSpec class was used to define textures, therefore a lot of rather
 *       texture-specific details are supplied to the specifier. The CPU buffer support was added
 *       later.
 */
class BufferSpec {
 public:

    /**
     * @brief Enumerator that broadly categorizes buffer usage
     */
    enum usage {
        RESIDUAL_SOURCE,            //!< Buffer serves as input for a residual block (add buffer to input of this layer)
        FUNCTION_SOURCE,            //!< Buffer serves as input for a layer that executes a function (convolution, add, pooling etc.)
        FUNCTION_DEST,              //!< Buffer serves as output of a function layer (convolution, add, pooling etc.)
        CONCAT_SOURCE,              //!< Buffer serves as input of a concatenation layer
        CONCAT_DEST,                //!< Buffer serves as output of a concatenation layer
        OES_DEST,                   //!< Buffer serves as destination for OES converter
        CPU_SOURCE,                 //!< Buffer serves as input for a upload-to-GPU operator or a bridge layer
        GPU_DEST,                   //!< Buffer serves as output for a upload-to-GPU operator
        CPU_DEST                    //!< Buffer serves as destination for a download-to-CPU operator or a bridge layer
    };

    /**
     * @brief Enumerator for image interpolation types
     */
    enum class interp : uint8_t {
        ANY = 0,                    //!< Interpolation not specified / not relevant
        NEAREST,                    //!< Use nearest-neighbor interpolation
        LINEAR                      //!< Use linear interpolation
    };


    /**
     * @brief Enumerator that defines \e internal (or sized) GL texture formats on the GPU
     */
    enum class sizedformat : GLint {
        RGBA32F   = GL_RGBA32F,       //!< 32-bit 4-channel floating-point elements
        RGB32F    = GL_RGB32F,        //!< 32-bit 3-channel floating-point elements
        RG32F     = GL_RG32F,         //!< 32-bit two-channel floating-point elements
        RED32F    = GL_R32F,          //!< 32-bit single-channel floating-point elements
        RGBA16F   = GL_RGBA16F,       //!< 16-bit 4-channel half-precision floating-point elements
        RGB16F    = GL_RGB16F,        //!< 16-bit 3-channel floating-point elements
        RG16F     = GL_RG16F,         //!< 16-bit two-channel floating-point elements
        RED16F    = GL_R16F,          //!< 16-bit single-channel floating-point elements
        RGBA8     = GL_RGBA8,         //!< 8-bit 4-channel unsigned byte elements
        RGB8      = GL_RGB8,          //!< 8-bit 3-channel unsigned byte elements
        RG8       = GL_RG8,           //!< 8-bit two-channel unsigned byte elements
        RED8      = GL_R8,            //!< 8-bit single-channel unsigned byte elements
        SINGLE32F = GL_R32F,          //!< 32-bit single-channel floating-point elements (alias for \c RED32F )
        SINGLE16F = GL_R32F,          //!< 16-bit single-channel floating-point elements (alias for \c RED16F )
#ifdef GL_R32UI
        SINGLE32UI = GL_R32UI,        //!< 32-bit single-channel integer (unsigned)
        RG32UI     = GL_RG32UI,       //!< 32-bit 2-channel integer (unsigned)
        RGB32UI    = GL_RGB32UI,      //!< 32-bit 3-channel integer (unsigned)
        RGBA32UI   = GL_RGBA32UI      //!< 32-bit 4-channel integer (unsigned)
#else
        SINGLE32UI = GL_R32UI_EXT     //!< 32-bit single-channel integer (unsigned)
        RG32UI     = GL_RG32UI_EXT,   //!< 32-bit 2-channel integer (unsigned)
        RGB32UI    = GL_RGB32UI_EXT,  //!< 32-bit 3-channel integer (unsigned)
        RGBA32UI   = GL_RGBA32UI_EXT  //!< 32-bit 4-channel integer (unsigned)
#endif
    };

    /**
     * @brief Enumerator that maps some GL formats to generic formats
     *
     * Generic formats in texture up/download operations specify the format of the data that is
     * passed to the texture up/download functions. The generic format is not necessarily the same
     * as the internal format, but it is the format that is used to interpret the data that is
     * passed to the texture up/download functions.
     */
    enum class genericformat : GLenum {
        RGBA       = GL_RGBA,
        RGB        = GL_RGB,
        RG         = GL_RG,
        RED        = GL_RED,
        SINGLE     = GL_RED,
        RGBA_INT   = GL_RGBA_INTEGER,
        RGB_INT    = GL_RGB_INTEGER,
        RG_INT     = GL_RG_INTEGER,
        RED_INT    = GL_RED_INTEGER,
        SINGLE_INT = GL_RED_INTEGER
    };

    /**
     * @brief Enumerator that maps some GL types to generic types
     */
    enum class dtype : GLenum {
        FLOAT16 = GL_HALF_FLOAT,
        FLOAT   = GL_FLOAT,
        FLOAT32 = GL_FLOAT,
        UINT32  = GL_UNSIGNED_INT,
        INT32   = GL_INT,
        UINT16  = GL_UNSIGNED_SHORT,
        INT16   = GL_SHORT,
        UBYTE   = GL_UNSIGNED_BYTE,
        UINT8   = GL_UNSIGNED_BYTE
    };

    /**
     * @brief Enumerator for location of storage / computing domains for tensor data
     *
     * Currently supported are CPU and GPU, with the latter being the preferred one.
     */
    enum class csdevice : uint8_t {
        COMP_STOR_GPU = 0,
        COMP_STOR_CPU
    };

    /**
     * @brief Specifier for the data order
     *
     * This defines the data storage order, which is device-specific as well as tensor-format
     * specific. Especially for GPU-based storage, we differentiate between \e shallow and \e deep
     * tensor storage order, as they are vastly different.
     *
     * @see GPULayerBase
     */
    enum class order : uint8_t {
        GPU_SHALLOW,        //!< Data is in GPU shallow format
        GPU_DEEP,           //!< Data is in GPU deep format (uses tiles to make the most of the texture cache)
        GPU_SEQUENCE,       //!< Data is in GPU sequence format (uses lines, not optimal but easier to process for attention layers)
        CHANNELWISE         //!< Data is in CPU 3D tensor format, stored as 3D array with the channels being the outermost index (w,h,c)
    };

    /**
     * @brief Create a buffer specifier
     *
     * @param channelIndex For multi-texture buffers, this provides the index to the texture
     * @param port Port number for layers that have multiple input/output ports
     * @param width Width of the buffer
     * @param height Height of the buffer
     * @param sizedFormat Sized (internal) format, akin to OpenGL internal texture formats
     * @param format Generic format, akin to OpenGL unsized texture formats
     * @param type Data type for the buffer
     * @param usage For what the buffer will be used
     * @param channels Total number of channels for the buffer
     *
     * This creates a buffer specifier with the parameters populated as supplied. The constructor
     * is idle otherwise.
     */
    BufferSpec(int channelIndex, int port, int width, int height, sizedformat sizedFormat, genericformat format, dtype type, usage usage, int channels = gpu::PIXEL_PACKING):
        usage_(usage), width_(width), height_(height), channels_(channels),
        channelIndex_(channelIndex), port_(port), internalFormat_(sizedFormat),
        format_(format), type_(type) {
    }

    /**
     * @brief Set data order for the buffer specifier
     *
     * @param dOrder Data order to set
     *
     * @return Reference to current BufferSpec object
     */
    BufferSpec& dataOrder(order dOrder) {
        dataOrder_ = dOrder;
        return *this;
    }

    /**
     * @brief Set interpolation for GPU-based tensors
     *
     * @param interpolation Interpolation mode to set
     *
     * @return Reference to current BufferSpec object
     *
     * This sets the interpolation for GPU-based tensors. If no interpolation is set to a buffer
     * specifier, the default wil be nearest-neighbor interpolation.
     */
    BufferSpec& interpolation(interp interpolation) {
        interpolation_ = interpolation;
        return *this;
    }

    /**
     * @brief Set target storage/compute device for the buffer specifier
     *
     * @param dev Device to set for computing and/or storage
     *
     * @return Reference to current BufferSpec object
     *
     * This sets the target device for computing/storage of that buffer. If no device is specified,
     * the target defaults to \c COMP_STOR_GPU
     */
    BufferSpec& device(csdevice dev) {
        device_ = dev;
        return *this;
    }

    /**
     * @brief Mark a buffer specifier for a pass-through buffer
     *
     * @param enable if set to \c true, the buffer/tensor is marked as passthrough
     *
     * @return Reference to current BufferSpec object
     *
     * Passthrough buffers are usually not allocated by the BufferManager, instead the original
     * buffer is used. This happens on layers that do not alter the data at all. An example would
     * be a concatenation layer for GPU-based shallow tensors, where some textures (each texture is
     * a single buffer) may just be passed unaltered.
     *
     * @see ConcatLayer
     */
    BufferSpec& passThrough(bool enable) {
        passThrough_ = enable;
        return *this;
    }

    /**
     * @brief Mark buffer specifier for an asynchronously operated buffer
     *
     * @param enable If set to \c true, buffer will be marked as asynchronously operated
     *
     * @return Reference to current BufferSpec object
     *
     * Some layers, as the upload and download layers, are able to operate asynchronously. For
     * these layers, the buffers that are read or written asynchronously should be marked as
     * asynchronous buffers.
     *
     * @see UploadLayer, DownloadLayer, DeepDownloadLayer
     *
     * @note Enabling asynchronicity implies locking the texture too
     */
    BufferSpec& async(bool enable) {
        async_ = enable;
        lock_ |= enable;
        return *this;
    }

    /**
     * @brief Set texture/buffer multiplicity
     *
     * @param multiplier Multiplicity to use
     *
     * @return Reference to current BufferSpec object
     *
     * Layers that can make use of multi-buffering can instruct the BufferManager to pass in more
     * than one buffer- or texture-set. The total number of sets is passed via the \p multiplier
     * variable.
     *
     * @note Having "shadow" texture output configuration implies locking
     *
     * @see UploadLayer, #multiplicity_
     */
    BufferSpec& multi(int multiplier) {
        multiplicity_ = multiplier;
        lock_ |= (multiplier > 1);
        return *this;
    }

    /**
     * @brief Lock the texture(s) / buffer for re-use, which means that it will be exempt from re-use
     *
     * @return Reference to current BufferSpec object
     */
    BufferSpec& lock() {
        lock_ = true;
        return *this;
    }


    /**
     * @brief Check if a sized format is an integral (integer) datatype 
     * 
     * @retval true sized format is an integral datatype
     * @retval false otherwise
     *
     * Integral datatypes on the GPU require to be read as integers in the shaders, whereas non-integral
     * datatypes are read as floating-point values in the shaders.
     */
    static bool isIntegral(sizedformat fmt) {
        switch (fmt) {
            case sizedformat::SINGLE32UI:
            case sizedformat::RG32UI:
            case sizedformat::RGB32UI:
            case sizedformat::RGBA32UI:
                return true;
            default:
                return false;
        }
    }


    /**
     * @brief Get sized format by number of channels and data type
     *
     * @param channels Number of channels, must be in {1,2,3,4}
     * @param type Data type, currently only \c FLOAT and \c UBYTE are supported
     *
     * @return Pair of sized and generic format that fulfils supplied channel/type criteria
     *
     * This function is helpful in conjunction with texture upload and download, to determine
     * the sized and generic format for a BufferSpec structure.
     */
    static std::pair<sizedformat, genericformat> formatByChannels(int channels, dtype type) {
        using sz = sizedformat;
        using gn = genericformat;
        static sizedformat flt32sfmt[4] = {sz::SINGLE32F, sz::RG32F, sz::RGB32F, sz::RGBA32F};
        static sizedformat flt16sfmt[4] = {sz::SINGLE16F, sz::RG16F, sz::RGB16F, sz::RGBA16F};
        static sizedformat uintsfmt[4] = {sz::SINGLE32UI, sz::RG32UI, sz::RGB32UI, sz::RGBA32UI};
        static sizedformat bytesfmt[4] = {sz::RED8, sz::RG8, sz::RGB8, sz::RGBA8};
        static genericformat gfmt[4] = {gn::RED, gn::RG, gn::RGB, gn::RGBA};
        static genericformat gifmt[4] = {gn::RED_INT, gn::RG_INT, gn::RGB_INT, gn::RGBA_INT};
        assert((channels > 0) && (channels <= 4));
        sizedformat sf;
        genericformat gf = gfmt[channels-1];
        switch (type) {
            case dtype::UBYTE:
                sf = bytesfmt[channels-1];
                break;
            case dtype::FLOAT16:
                sf = flt16sfmt[channels-1];
                break;
            case dtype::UINT32:
                sf = uintsfmt[channels-1];
                gf = gifmt[channels-1];
                break;
            default:  // we assume float32 by default
                sf = flt32sfmt[channels-1];
        }
        return std::pair<sizedformat, genericformat>(sf, gf);
    }

    /**
     * @brief Retrieve atomic (channel) size of a datatype
     *
     * @param type Data type to retrieve size for
     *
     * @param fp16To32 Flag that when set to \c true will assume FP16 to be 32-bit
     *
     * @return Atomic (channel) size of data type (in bytes)
     */
    static int typeSize(BufferSpec::dtype type, bool fp16To32 = true) {
        switch (type) {
            case dtype::FLOAT16:
                return (fp16To32) ? 4 : 2;
            case dtype::FLOAT32:
            case dtype::UINT32:
            case dtype::INT32:
                return 4;
            case dtype::UINT16:
            case dtype::INT16:
                return 2;
            case dtype::UINT8:
                return 1;
            default:
                return 4;
        }
    }

    usage usage_;                     //!< What the buffer is supposed to be used for
    int width_;                       //!< Width of the buffer
    int height_;                      //!< Height of the buffer
    int channels_;                    //!< Number of channels per pixel
    int channelIndex_;                //!< Offset/index for multi-texture-buffers
    int port_;                        //!< Layer port to connect to (for layers with multiple input ports)
    sizedformat internalFormat_;      //!< Sized buffer/texture format (matches OpenGL internal texture format as of now)
    genericformat format_;            //!< Generic buffer/texture format (matches OpenGL texture format as of now)
    dtype type_;                      //!< Data type for this buffer (matches OpenGL data types)
    bool async_ = false;              //!< Flag that indicates that the buffer is subject to an asynchronous read or write operation (texture uploads and downloads)
    bool lock_ = false;               //!< Flag that indicates that the buffer should be exempt from re-use and only be used for this layer's (output)

    /**
     * Spatial interpolation for the buffer (either linear or nearest-neighbor)
     */
    interp interpolation_ = interp::NEAREST;

    /**
     * Device type where the buffer should be allocated on (GPU or CPU)
     */
    csdevice device_ = csdevice::COMP_STOR_GPU;

    /**
     * In case multiple sets of the same textures are required, this defines how many sets will
     * be generated. This functionality is for example used in the UploadLayer for asynchronous
     * operation. It uses the \c shadowIndex parameter of GPULayerBase::addOutputTexture()
     * to inform the layer about which of the multibuffers to use.
     */
    int multiplicity_ = 1;

    /**
     * Flag that indicates that the buffer specifier was created by a layer that does not
     * write to that buffer, but the buffer is merely a passed-through input buffer or a part
     * of such. This can happen in concatenation layers in \e shallow mode or in layers that are
     * using some internal textures and are able to overwrite the input texture (for example in
     * more complex sequence processing layers).
     */
    bool passThrough_ = false;

    /**
     * Data order for the buffer, can be:
     *   - \c GPU_SHALLOW
     *   - \c GPU_DEEP
     *   - \c GPU_SEQUENCE
     *   - \c CHANNELWISE
     *
     * Defaults to \c GPU_SHALLOW.
     * 
     * @see CPUBuffer
     */
    order dataOrder_ = order::GPU_SHALLOW;
};

} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
