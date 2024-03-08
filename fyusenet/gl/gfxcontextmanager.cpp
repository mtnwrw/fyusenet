//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context manager
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/miscdefs.h"
#include "glexception.h"
#include "../gpu/gfxcontextmanager.h"   // NOTE (mw) hacky
#include "glcontext.h"
#include "pbopool.h"
#include "shadercache.h"
#include "shadersnippet.h"
#include "scoped_texturepool.h"
#ifdef FYUSENET_MULTITHREADING
#include "asyncpool.h"
#endif

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------

std::vector<std::shared_ptr<GfxContextManager>> GfxContextManager::managers_;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Destructor
 *
 * Clear all GL contexts maintained by the manager,
 *
 * @throws GLException (when in debug build) in case there was any context that is still in use
 *
 * @pre There shall be no GfxContextLink instances linking to any of the GL contexts.
 */
#ifdef DEBUG
GfxContextManager::~GfxContextManager() noexcept(false) {
#else
GfxContextManager::~GfxContextManager() {
#endif
    if (!contexts_.empty()) {
#ifdef DEBUG
        THROW_EXCEPTION_ARGS(opengl::GLException,"Context manager was not torn down before destruction");
#else
        FNLOGE("Context manager was not torn down before destruction, expect GL memory leaks");
#endif
    }
}


/**
 * @brief Create a context link from an existing context
 *
 * @param ctxIdx Index of the context to get the link to
 *
 * @return Valid link to context with the supplied \p ctxIdx or an invalid/empty context.
 *
 * @see GLContextInterface::index()
 *
 * @note This function does \b not create any context.
 */
fyusenet::GfxContextLink GfxContextManager::context(int ctxIdx) const {
    if (ctxIdx >= (int)contexts_.size()) return fyusenet::GfxContextLink::EMPTY;
    return fyusenet::GfxContextLink(contexts_[ctxIdx]);
}


/**
 * @brief Creates a GL context wrapper from the currently active (external) GL context to be used as main context
 *
 * @return Context link to the currently active GL context
 *
 * This function creates a wrapper around the currently bound GL context and stores it to the
 * internal context list. This is meant for use-cases such as using FyuseNet inside an application
 * that already has a GL context running, which should be shared.
 * The external context is designated as \e main GL context for the manager (ideally there is only
 * one manager per process). If more than one context is needed, it is advised to use #createDerived
 * to create a \e shared context.
 */
fyusenet::GfxContextLink GfxContextManager::createMainContextFromCurrent() {
    // TODO (mw) thread safety
#if !defined(FYUSENET_USE_GLFW) && !defined(__APPLE__)
    int idx = (int)contexts_.size();
    opengl::GLContext * ctx = opengl::GLContext::createFromCurrent(idx, this);
    if (!ctx) THROW_EXCEPTION_ARGS(opengl::GLException, "Cannot wrap external GL context");
    contexts_.push_back(ctx);
    mainContext_ = ctx;
    opengl::GLInfo::init(false);
    return fyusenet::GfxContextLink(ctx);
#else
    THROW_EXCEPTION_ARGS(opengl::GLException, "Not implemented");
#endif
}

#ifdef FYUSENET_USE_WEBGL
/**
 * @brief Create a new GL context on the manager-associated device and use it as main context
 *
 * @param canvas Target canvas to attach this context to
 * @param width Width of the canvas
 * @param height Height of the canvas
 * @param makeCurrent If set to \c true (default), the context will be made current to the calling
 *                    thread.
 *
 * @return Context link
 *
 * @note If \p makeCurrent is set, any previously bound context in this thread will be kicked
 *       off ths thread.
 *
 * This function creates a new GL context that is stored in the internal context list and will
 * return a link to this newly created context. The context is designated as \e main GL context
 * for the manager (ideally there is only one manager per process), if more than one context is
 * needed, it is advised to use #createDerived to create a \e shared context.
 *
 * @note If you want to create a \e shared GL context, check #createDerived .
 * @note This function is not thread-safe.
 *
 * @see createDerived
 */
GfxContextLink GfxContextManager::createMainContext(char *canvas, int width, int height, bool makeCurrent) {
    // TODO (mw) thread safety
    opengl::GLContext * ctx = new opengl::GLContext(canvas, 0, this, width, height);
    ctx->init();
    if (makeCurrent) {
        if (!ctx->makeCurrent()) THROW_EXCEPTION_ARGS(opengl::GLException,"Cannot make GL context %p the current one",ctx);
    }
#ifdef DEBUG
    opengl::GLInfo::init(true);
#else
    opengl::GLInfo::init(false);
#endif
    contexts_.push_back(ctx);
    mainContext_ = ctx;
    return GfxContextLink(ctx);
}
#else
/**
 * @brief Create a new GL context on the manager-associated device and use it as main context
 *
 * @param makeCurrent If set to \c true (default), the context will be made current to the calling
 *                    thread.
 *
 * @return Context link
 *
 * @note If \p makeCurrent is set, any previously bound context in this thread will be kicked
 *       off ths thread.
 *
 * This function creates a new GL context that is stored in the internal context list and will
 * return a link to this newly created context. The context is designated as \e main GL context
 * for the manager (ideally there is only one manager per process), if more than one context is
 * needed, it is advised to use #createDerived to create a \e shared context.
 *
 * @note If you want to create a \e shared GL context, check #createDerived .
 * @note This function is not thread-safe.
 *
 * @see createDerived
 */
fyusenet::GfxContextLink GfxContextManager::createMainContext(bool makeCurrent) {
    // TODO (mw) thread safety
    int idx = (int)contexts_.size();
    auto * ctx = new opengl::GLContext(idx, deviceID_, this);
    ctx->init();
    if (makeCurrent) {
        if (!ctx->makeCurrent()) THROW_EXCEPTION_ARGS(opengl::GLException,"Cannot make GL context %p the current one",ctx);
    }
#ifdef DEBUG
    opengl::GLInfo::init(true);
#else
    opengl::GLInfo::init(false);
#endif
    contexts_.push_back(ctx);
    mainContext_ = ctx;
    return fyusenet::GfxContextLink(ctx);
}
#endif

/**
 * @brief Create a new GL context on the manager-associated device that is sharing with an existing context
 *
 * @param ctx Link to context that is supposed to share data with the newly created one
 *
 * @return Link to context that shares resources with the supplied context
 *
 * This function will create a new GL context by "deriving" it from the main context of the supplied
 * context link. In case the supplied \p ctx is linking to a main context (i.e. a context that has
 * not been derived from another context) then this context will be used directly.
 * Deriving in our case means that the new context will have the supplied (main) context being
 * entered as a context to share resources with. It is best practice that if you want to create
 * several shared contexts, that you derive a set of subordinate contexts from a main context.
 *
 * The main context serves as anchor for the derived context, such that the derived context is
 * assigned the main context as its parent (see GLContextInterface::isDerivedFrom() and
 * GLContextInterface::derivedIndex() ) and will be addressed by the parent and a derived
 * index.
 *
 * @throws GLException on errors or in case an invalid context has been supplied to derive from.
 *
 * @note The newly created context will \b not be current to the calling thread.
 * @note This function is not thread-safe.
 *
 * @see GLContextInterface::isDerivedFrom, GLContextInterface::derivedIndex
 */
fyusenet::GfxContextLink GfxContextManager::createDerived(const fyusenet::GfxContextLink& ctx) {
    if (!ctx.context_) THROW_EXCEPTION_ARGS(opengl::GLException,"Illegal (empty) context supplied");
    auto * context = dynamic_cast<const opengl::GLContext *>(ctx.interface());
    assert(context);
    int idx = context->derivedCounter_.fetch_add(1);
    opengl::GLContext * derived = context->derive((int)contexts_.size(), idx);
    assert(derived);
    contexts_.push_back(derived);
    return fyusenet::GfxContextLink(derived);
}



/**
 * @brief Create a new GL context on the manager-associated device that is sharing with an existing context
 *
 * @param intf GLContextInterface of the context that is supposed to share data with the newly created one
 *
 * @return Link to context that shares resources with the supplied context
 *
 * This function will create a new GL context by "deriving" it from the supplied context interface.
 * Deriving in our case means that the new context will have the supplied context being
 * entered as a context to share resources with. It is best practice that if you want to create
 * several shared contexts, that you derive a set of subordinate contexts from a main context.
 *
 * The main context serves as anchor for the derived context, such that the derived context is
 * assigned the main context as its parent (see GLContextInterface::isDerivedFrom() and
 * GLContextInterface::derivedIndex() ) and will be addressed by the parent and a derived
 * index.
 *
 * @throws GLException on errors or in case an invalid context has been supplied to derive from.
 *
 * @note The newly created context will \b not be current to the calling thread.
 * @note This function is not thread-safe.
 *
 * @see GLContextInterface::isDerivedFrom, GLContextInterface::derivedIndex
 */
fyusenet::GfxContextLink GfxContextManager::createDerived(const opengl::GLContextInterface * intf) {
    if (!intf) THROW_EXCEPTION_ARGS(opengl::GLException,"Illegal (empty) context supplied");
    auto * context = dynamic_cast<const opengl::GLContext *>(intf);
    assert(context);
    int idx = context->derivedCounter_.fetch_add(1);
    opengl::GLContext * derived = context->derive((int)contexts_.size(), idx);
    assert(derived);
    contexts_.push_back(derived);
    return fyusenet::GfxContextLink(derived);
}


/**
 * @brief Retrieve derived GL context link
 *
 * @param ctx Main context to retrieve a derived context from
 * @param derivedIndex Index within the list of derived contexts (specific to the main context)
 *
 * @return Link to the derived context, or empty context if the derived context was not found
 *
 * @note This function is not thread-safe.
 *
 * @throws GLException in case an invalid context was supplied
 */
fyusenet::GfxContextLink GfxContextManager::getDerived(const fyusenet::GfxContextLink& ctx, int derivedIndex) const {
    if (!ctx.context_) THROW_EXCEPTION_ARGS(opengl::GLException,"Illegal (empty) context supplied");
    for (auto * candidate : contexts_) {
        if (candidate->isDerivedFrom(ctx.interface())) {
            if (candidate->derivedIdx_ == derivedIndex) {
                return fyusenet::GfxContextLink(candidate);
            }
        }
    }
    return fyusenet::GfxContextLink::EMPTY;
}


/**
 * @brief Retrieve/create instance of the GfxContextManager for a GPU/GL-device
 *
 * @param device Device ID (starting at 0)
 *
 * @return Pointer to instance of the context manager for the specified \p device.
 *
 * @note This function is not thread-safe.
 *
 * @warning We currently support only one GPU/device. Though the context manager has some
 *          preparations for multi-GPU support done already, the tear-down mechanism
 *          currently assumes that the context manager is a singleton. For multi-GPU support,
 *          the teardown of the GL thread pool and the shader cache need to be adjusted for
 *          multi-GPU support.
 */
std::shared_ptr<GfxContextManager> GfxContextManager::instance(int device) {
    assert(device >= 0);
    // TODO (mw) thread safety
    if ((int)managers_.size() <= device) {
        managers_.resize(device+1, nullptr);
    }
    if (!managers_[device]) {
        managers_[device] = std::shared_ptr<GfxContextManager>(new GfxContextManager(device));
    }
    return managers_[device];
}


/**
 * @brief GfxContextManager::tearDown
 *
 * This function tears down the GL resources with singleton character, including the AsyncPool, the
 * ShaderCache and \e all GfxContextManager instances that have been created. This should be done as
 * the very last operation in a program from the main thread.
 *
 * @note This function is not thread-safe. It is recomended to tear down the context manager from
 *       the main thread as last action.
 */
void GfxContextManager::tearDown() {
    opengl::ShaderCache::tearDown();
    opengl::ShaderSnippet::tearDown();
#ifdef FYUSENET_MULTITHREADING
    opengl::AsyncPool::tearDown();
    if (!opengl::AsyncPool::isEmpty()) {
        THROW_EXCEPTION_ARGS(FynException, "There are still GL threads pending");
    }
#endif
    while (!managers_.empty()) {
        auto mgr = managers_.front();
        mgr->cleanup();
    }
    assert(managers_.empty());
}


/**
 * @brief Tear down context manager
 *
 * @pre The ShaderCache and the AsyncPool have been torn down
 *
 * @throws GLException on errors and precondition violations (in debug builds)
 *
 * This function deletes the context manager from memory and all contexts and pool associated
 * with it.
 */
void GfxContextManager::cleanup() {
    if (mainContext_) {
        if (!mainContext_->isCurrent()) {
            bool ctxok = mainContext_->makeCurrent();
            if (!ctxok) {
#ifdef DEBUG
                THROW_EXCEPTION_ARGS(opengl::GLException,"Cannot tear down context manager from outside the main context");
#else
                FNLOGE("Tearing down context manager without context current, expect GL memory leaks");
#endif
            }
        } else {
            FNET_DEL_AND_CLEAR(pboReadPool_);
            FNET_DEL_AND_CLEAR(pboWritePool_);
            FNET_DEL_AND_CLEAR(texturePool_);
            for (opengl::GLContext * ctx : contexts_) {
                assert(ctx);
                if ((ctx->uses() > 0) && (!ctx->isExternal())) {
#ifdef DEBUG
                    THROW_EXCEPTION_ARGS(opengl::GLException,"Context %p (idx=%d) on device %d has %d uses left, not deleting -> memory leak", ctx, ctx->index(), deviceID_, ctx->uses());
#else
                    FNLOGE("Context %p (idx=%d) on device %d has %d uses left, not deleting -> memory leak", ctx, ctx->index(), deviceID_, ctx->uses());
#endif
                } else {
                    delete ctx;         // even if the context is external, we delete the wrapper
                }
            }
            contexts_.clear();
        }
    } else {
        if (!contexts_.empty()) THROW_EXCEPTION_ARGS(opengl::GLException, "No main context set, yet this manager has %ld contexts, canot teardown",contexts_.size());
    }
    // TODO (mw) thread-safety
    for (auto it=managers_.begin(); it != managers_.end(); ++it) {
        if (it->get() == this) {
            managers_.erase(it);
            break;
        }
    }
}


/**
 * @brief Setup %PBO pools
 *
 * @param readPoolSize Number of PBOs in the read pool
 * @param writePoolSize Number of PBOs in the write pool
 *
 * This function allocates two PBOPool instances, one for uploading (write) textures and one for
 * downloading (read) textures.
 */
void GfxContextManager::setupPBOPools(int readPoolSize, int writePoolSize) {
    // TODO (mw) thread-safety and make sure that we are running in the right context
    assert(pboReadPool_ == nullptr);
    assert(pboWritePool_ == nullptr);
    pboReadPool_ = new opengl::PBOPool(readPoolSize);
    pboWritePool_ = new opengl::PBOPool(writePoolSize);
}


/**
 * @brief Setup texture pool
 *
 * This function creates a texture pool for the current context as well as shared contexts.
 * A texture pool is useful for dealing with transient textures as they appear in textures that are
 * "internal" to layers in case they need to do some multi-pass computations.
 */
void GfxContextManager::setupTexturePool() {
    texturePool_ = new opengl::ScopedTexturePool(context(0));
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Create context manager instance
 *
 * @param device Device to create the manager for
 *
 * Idle constructor
 */
GfxContextManager::GfxContextManager(int device) : deviceID_(device ) {
}


/**
 * @brief Find GL context managed by any instance of the context manager
 *
 * @param candidate Optional existing GL context interface, if \c nullptr is provided,
 *                  the context (if any) of the current thread is looked for
 *
 * @return Pointer to GLContext that matches the search request
 */
opengl::GLContext * GfxContextManager::findCurrentContext(opengl::GLContextInterface * candidate) {
    if (!candidate) {
#ifdef FYUSENET_USE_EGL
        auto context = eglGetCurrentContext();
#elif defined(FYUSENET_USE_GLFW)
        auto context = glfwGetCurrentContext();
#elif defined(__linux__) && !defined(FYUSENET_USE_EGL)
        auto context = glXGetCurrentContext();
#elif defined(__APPLE__)
        auto context = CGLGetCurrentContext();
#elif defined(FYUSENET_USE_WEBGL)
        auto context = emscripten_webgl_get_current_context();
#else
        HGLRC context = wglGetCurrentContext();
#endif
        for (auto & mgr : managers_) {
            for (opengl::GLContext * ctx : mgr->contexts_) {
                if (ctx->matches(context)) return ctx;
            }
        }
    } else {
        for (auto & mgr : managers_) {
            for (opengl::GLContext * ctx : mgr->contexts_) {
                if (candidate == static_cast<opengl::GLContextInterface *>(ctx)) return ctx;
            }
        }
    }
    return nullptr;
}


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
