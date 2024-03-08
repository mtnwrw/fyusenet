//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context Manager (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <vector>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "gfxcontextlink.h"
#include "../gl/glcontextinterface.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {

namespace opengl {
    class GLContext;
    class ScopedTexturePool;
}

namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Manager instance for graphics / OpenGL contexts on a per-GPU basis
 *
 * This class serves as a singleton per GPU/GL-device which issues and maintains OpenGL contexts
 * that can be used for operation.
 *
 * @note The header file of this link resides in the GPU subfolder instead of the backend-
 *       specific GL folder, whereas the implementation is placed in the GL subfolder. In
 *       addition, the class resides within the fyusenet namespace. This is done for extension
 *       reasons (change of backend) and not by accident. Not very clean though.
 *
 * @warning We currently support only one GPU/device. Though the context manager has some
 *          preparations for multi-GPU support done already, the tear-down mechanism
 *          currently assumes that the context manager is a singleton. For multi-GPU support,
 *          the teardown of the GL thread pool and the shader cache need to be adjusted for
 *          multi-GPU support.
 *
 * @see GfxContextLink
 */
class GfxContextManager {
    friend class opengl::GLContext;
    friend class GfxContextLink;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
#ifdef DEBUG
    ~GfxContextManager() noexcept(false);
#else
    ~GfxContextManager();
#endif

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    fyusenet::GfxContextLink context(int ctxIdx = 0) const;
    fyusenet::GfxContextLink createMainContextFromCurrent();
#ifdef FYUSENET_USE_WEBGL
    fyusenet::GfxContextLink createMainContext(char *canvas, int width, int height, bool makeCurrent = true);
#else
    fyusenet::GfxContextLink createMainContext(bool makeCurrent = true);
#endif
    fyusenet::GfxContextLink createDerived(const fyusenet::GfxContextLink& ctx);
    fyusenet::GfxContextLink createDerived(const opengl::GLContextInterface * intf);
    fyusenet::GfxContextLink getDerived(const fyusenet::GfxContextLink& ctx, int derivedIndex) const;
    void setupPBOPools(int readPoolSize, int writePoolSize);
    void setupTexturePool();
    static std::shared_ptr<GfxContextManager> instance(int device=0);
    static void tearDown();
    void cleanup();

    /**
     * @brief Retrieve pointer to texture pool (if it exists)
     *
     * @return Pointer to BasicTexturePool instance or \c nullptr if no pool exists
     *
     * @see setupTexturePool(), #texturePool_
     */
    opengl::ScopedTexturePool * texturePool() const {
        return texturePool_;
    }

    /**
     * @brief Retrieve pointer to read-type PBOPool instance for texture download
     *
     * @return Pointer to PBOPool instance (may be \c nullptr if pool is not set up)
     */
    opengl::PBOPool * getReadPBOPool() const {
        return pboReadPool_;
    }

    /**
     * @brief Retrieve pointer to write-type PBOPool instance for texture upload
     *
     * @return Pointer to PBOPool instance (may be \c nullptr if pool is not set up)
     */
    opengl::PBOPool * getWritePBOPool() const {
        return pboWritePool_;
    }


 protected:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------

    GfxContextManager(int device);
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Get pointer to main GL context
     *
     * @return Pointer to context that was created as "main" GL context
     */
    opengl::GLContextInterface * getMain() const {
        return reinterpret_cast<opengl::GLContextInterface *>(mainContext_);
    }

    static opengl::GLContext * findCurrentContext(opengl::GLContextInterface * candidate = nullptr);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int deviceID_ = 0;                                    //!< Device/GPU ID for this manager instance
    mutable std::vector<opengl::GLContext *> contexts_;   //!< List of GL contexts held by the manager instance
    opengl::GLContext * mainContext_ = nullptr;           //!< Pointer to main (first) OpenGL context
    opengl::PBOPool * pboReadPool_ = nullptr;             //!< Pointer to PBOPool used for reading/downloading textures
    opengl::PBOPool * pboWritePool_ = nullptr;            //!< Pointer to PBOPool used for writing/uploading textures
    opengl::ScopedTexturePool * texturePool_ = nullptr;   //!< Pointer to optional texture pool

    /**
     * List of manager singletons, indexed by device ID (starting at 0)
     */
    static std::vector<std::shared_ptr<GfxContextManager>> managers_;
};

} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
