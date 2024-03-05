//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context for EGL
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#ifdef FYUSENET_USE_EGL

//--------------------------------------- System Headers -------------------------------------------

#ifdef USE_ANDROID_WINDOW
#include <android_native_app_glue.h>
#endif
#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl_sys.h"
#include "../glexception.h"
#include "../../common/logging.h"
#include "../glcontext.h"
#include "../../gpu/gfxcontextmanager.h"

//-------------------------------------- Global Variables ------------------------------------------

#ifndef FYUSENET_USE_EGL
#error This file should not be compiled
#endif

static PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT = nullptr;
static PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = nullptr;
static PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = nullptr;
static PFNEGLQUERYDISPLAYATTRIBEXTPROC eglQueryDisplayAttribEXT = nullptr;
static PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = nullptr;

namespace fyusion::opengl {

//-------------------------------------- Local Definitions -----------------------------------------

#ifdef ANDROID
#define ES3BIT EGL_OPENGL_ES3_BIT_KHR
#else
#define ES3BIT EGL_OPENGL_ES3_BIT
#endif

static bool initEGLExtensions();

static const EGLint displayConfig16Bit[] = {
    EGL_RENDERABLE_TYPE, ES3BIT,
#ifndef USE_ANDROID_WINDOW
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
#else
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
#endif
    EGL_RED_SIZE,   5,
    EGL_GREEN_SIZE, 6,
    EGL_BLUE_SIZE,  5,
    EGL_DEPTH_SIZE, 24,
    EGL_STENCIL_SIZE, 8,
    EGL_NONE
};


static const EGLint displayConfig24Bit[] = {
    EGL_RENDERABLE_TYPE, ES3BIT,
#ifndef USE_ANDROID_WINDOW
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
#else
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
#endif
    EGL_RED_SIZE,   8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE,  8,
    EGL_DEPTH_SIZE, 24,
    EGL_STENCIL_SIZE, 8,
    EGL_NONE
};


static const EGLint displayConfig32Bit[]= {
    EGL_RENDERABLE_TYPE, ES3BIT,
#ifndef USE_ANDROID_WINDOW
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
#else
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
#endif
    EGL_RED_SIZE,   8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE,  8,
    EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_STENCIL_SIZE, 8,
    EGL_NONE
};


const EGLint surfaceAttribs[]= {
  EGL_WIDTH,GLContext::DEFAULT_WIDTH,
  EGL_HEIGHT,GLContext::DEFAULT_HEIGHT,
  //EGL_TEXTURE_FORMAT,EGL_NO_TEXTURE,
  //EGL_TEXTURE_TARGET,EGL_NO_TEXTURE,
  EGL_NONE
};

const EGLint * EGLConfigs[3]={displayConfig32Bit,displayConfig24Bit,displayConfig16Bit};


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Idle constructor
 *
 * @param idx Context index as given by the GfxContextManager
 * @param device Device ID to run the context on
 * @param manager Pointer to GfxContextManager instance that manages this context
 * @param width Width of underlying surface
 * @param height Height of underlying surface
 */
GLContext::GLContext(int idx, int device, fyusenet::GfxContextManager * manager, int width,int height) :
    GLContextInterface(idx, device), width_(width), height_(height), manager_(manager) {
}

/**
 * @brief Destructor
 *
 * Takes down GL context and releases resources held by it.
 */
GLContext::~GLContext() {
#ifdef DEBUG
    if (uses() > 0) {
        FNLOGW("Destroying GL context with %d active links, check your code", uses());
    }
#endif
   eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display_,context_);
   context_ = nullptr;
   eglDestroySurface(display_, defaultSurface_);
   if (!derivedFrom_) {
       eglTerminate(display_);
   }
   defaultSurface_ = nullptr;
   display_ = nullptr;
}


/**
 * @copydoc GLContextInterface::getWritePBOPool
 */
PBOPool * GLContext::getWritePBOPool() const {
    if (isDerived()) {
        auto * main = getMain();
        if (!main) THROW_EXCEPTION_ARGS(GLException,"No main context registered");
        return main->getWritePBOPool();
    } else {
        assert(manager_);
        return manager_->getWritePBOPool();
    }
}

/**
 * @copydoc GLContextInterface::getReadPBOPool
 */
PBOPool * GLContext::getReadPBOPool() const {
    if (isDerived()) {
        auto * main = getMain();
        if (!main) THROW_EXCEPTION_ARGS(GLException,"No main context registered");
        return main->getReadPBOPool();
    } else {
        assert(manager_);
        return manager_->getReadPBOPool();
    }
}


/**
 * @copydoc GLContextInterface::isCurrent()
 */
bool GLContext::isCurrent() const {
    if (!context_) return false;
    EGLContext ctx = eglGetCurrentContext();
    if ((ctx == EGL_NO_CONTEXT) || (!ctx)) return false;
    return (ctx == context_);
}


/**
 * @copydoc GLContextInterface::init()
 */
void GLContext::init() {
    const int MAX_DEVICES = 8;
    EGLint major, minor, configs;
    EGLDeviceEXT eglDevices[MAX_DEVICES];
    EGLint numdevs;
    // TODO (mw) check if this works under Android
    EGLDisplay display = EGL_NO_DISPLAY;
    if (!initEGLExtensions()) {
        display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (!eglInitialize(display_, &major, &minor)) {
            THROW_EXCEPTION_ARGS(GLException, "Cannot initialize EGL subsystem");
        }
    } else {
        // ------------------------------------------------------------------
        // Iterate through EGL backends, assuming that the "fastest" ones
        // are first in the list...
        // ------------------------------------------------------------------
        eglQueryDevicesEXT(MAX_DEVICES, eglDevices, &numdevs);
        for (EGLint i = 0; i < numdevs ; i++) {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevices[i], 0);
            if (display != EGL_NO_DISPLAY) {
                if (const char *vendor = eglQueryDeviceStringEXT(eglDevices[i], EGL_VENDOR); vendor) {
                    // TODO (mw) filter by vendor
                }
#ifdef EGL_RENDERER_EXT
                if (const char *render = eglQueryDeviceStringEXT(eglDevices[i], EGL_RENDERER_EXT); render) {
                    // TODO (mw) filter by renderer
                }
#endif
                // --------------------------------------------------------------
                // If we found a display, try to use it, break out of the loop on
                // success
                // --------------------------------------------------------------
                if (eglInitialize(display, &major, &minor)) {
                    display_ = display;
                    break;
                }
            }
        }
    }
    if (display_ == EGL_NO_DISPLAY) {
        THROW_EXCEPTION_ARGS(GLException, "Unable to open any EGL display");
    }
    bool success = false;
    for (int i=0; i < 3; i++) {
        EGLBoolean rc = eglChooseConfig(display_, EGLConfigs[i], &activeConfig_, 1, &configs);
        if (rc) {
            success=true;
            break;
        }
    }
    EGLint attribs[] ={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
    if (!success) THROW_EXCEPTION_ARGS(GLException,"Cannot get EGL context to work with");
    context_ = eglCreateContext(display_, activeConfig_, EGL_NO_CONTEXT, attribs);
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot create EGL context");
#ifdef USE_ANDROID_WINDOW
    if (!nativeWindow_) THROW_EXCEPTION_ARGS(GLException,"No native window supplied");
    EGLint fmt=0;
    eglGetConfigAttrib(display_,activeConfig_,EGL_NATIVE_VISUAL_ID,&fmt);
    int rc=AnativeWindow__setBuffersGeometry(nativeWindow_,0,0,fmt);
    defaultSurface_ = eglCreateWindowSurface(display_,activeConfig_,nativeWindow_,NULL);
#else
    defaultSurface_ = eglCreatePbufferSurface(display_, activeConfig_, surfaceAttribs);
#endif
    if (defaultSurface_ == EGL_NO_SURFACE) {
        EGLint err = eglGetError();
        THROW_EXCEPTION_ARGS(GLException,"Unable to generate EGL surface, errcode 0x%X",err);
    }
    activeSurface_ = defaultSurface_;
}

#ifdef ANDROID
void GLContext::setNativeWindow(NativeWindowType win) {
    nativeWindow_ = win;
}
#endif


/**
 * @copydoc GLContextInterface::makeCurrent()
 */
bool GLContext::makeCurrent() const {
    if (external_) {
        return false;
    } else {
        if ((activeSurface_ == EGL_NO_SURFACE) || (!context_)) {
            return false;
        }
        EGLBoolean rc = eglMakeCurrent(display_, activeSurface_, activeSurface_, context_);
        if (!rc) {
            return false;
        }
        assert(eglGetCurrentContext() == context_);
        return true;
    }
}


/**
 * @copydoc GLContextInterface::releaseCurrent()
 */
bool GLContext::releaseCurrent() const {
    if (isCurrent()) {
        return eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    } else return false;
}


/**
 * @copydoc GLContextInterface::sync()
 */
void GLContext::sync() const {
#ifdef ANDROID
#ifdef USE_ANDROID_WINDOW
    eglSwapBuffers(display_,defaultSurface_);
#else
    glFinish();
#endif
#else
    eglSwapBuffers(display_, activeSurface_);
#endif
}


/**
 * @copydoc GLContextInterface::useDefaultSurface()
 */
void GLContext::useDefaultSurface() {
    activeSurface_ = defaultSurface_;
    eglMakeCurrent(display_, activeSurface_, activeSurface_, context_);
}


/**
 * @copydoc GLContextInterface::isDerivedFrom()
 */
bool GLContext::isDerivedFrom(const GLContextInterface * main) const {
    return (static_cast<const GLContextInterface *>(derivedFrom_) == main);
}


/**
 * @copydoc GLContextInterface::hash()
 */
uint64_t GLContext::hash() const {
  return (uint64_t )context_;
}


/**
 * @copydoc GLContextInterface::getMain()
 */
GLContextInterface * GLContext::getMain() const {
    assert(manager_);
    return manager_->getMain();
}


/**
 * @copydoc GLContextInterface::texturePool()
 */
ScopedTexturePool * GLContext::texturePool() const {
    assert(manager_);
    return manager_->texturePool();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Create a wrapped GL context from the currently active context
 *
 * @param idx Index that will be used by the GfxContextManager to store the new context
 * @param mgr Pointer to GfxContextManager that manages this context
 *
 * @return Pointer to OpenGL context object or \c nullptr if there was no context current
 *
 * This function wraps the currently active GL context into this wrapper class, such that
 * it can be used with the context manager, for example to create shared contexts.
 */
GLContext * GLContext::createFromCurrent(int idx, fyusenet::GfxContextManager *mgr) {
    EGLContext ccontext = eglGetCurrentContext();
    if (ccontext == EGL_NO_CONTEXT) return nullptr;
    GLContext * ctx = new GLContext(ccontext, idx, mgr);
    ctx->display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    ctx->defaultSurface_ = eglGetCurrentSurface(EGL_DRAW);
    ctx->activeSurface_ = ctx->defaultSurface_;
    ctx->external_ = true;
    return ctx;
}


/**
 * @brief Derive shared GL context from current context
 *
 * @param idx Index for the new context as determined by the GfxContextManager
 * @param dIdx Derived index as determined by the GfxContextManager
 *
 * @return Pointer to derived GL context that is shared with the current context
 *
 * This function creates a new GLContext instance that has shared resources with the context
 * represented by this object.
 */
GLContext * GLContext::derive(int idx, int dIdx) const {
    assert(manager_);
    assert(derivedFrom_ == nullptr);
    if (!context_) THROW_EXCEPTION_ARGS(GLException,"Cannot derive context from empty context");
    EGLint attribs[] = {EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
    auto newctx = eglCreateContext(display_,activeConfig_,context_,attribs);
    if (!newctx) THROW_EXCEPTION_ARGS(GLException,"Unable to derive context");
    auto * ctx = new GLContext(newctx, this, idx, dIdx, manager_);
    ctx->display_ = display_;
    ctx->defaultSurface_ = eglCreatePbufferSurface(display_, activeConfig_, surfaceAttribs);
    ctx->activeSurface_ = ctx->defaultSurface_;
    return ctx;
}

/**
 * @brief Constructor for shared context
 *
 * @param ctx GLXContext to wrap
 * @param from Context that this context is derived from (shared with)
 * @param idx Context index as determined by the GfxContextManager
 * @param dIdx Derived context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(EGLContext ctx, const GLContext * from, int idx, int dIdx, fyusenet::GfxContextManager * mgr) :
    GLContextInterface(idx, from->device()), context_(ctx), derivedFrom_(from), manager_(mgr) {
    derivedIdx_ = dIdx;
}


/**
 * @brief Constructor from existing context
 *
 * @param ctx GLXContext to wrap
 * @param idx Context index as determined by the GfxContextManager
 * @param mgr Pointer to GfxContextManager that manages this and the \p from context
 */
GLContext::GLContext(EGLContext ctx, int idx, fyusenet::GfxContextManager * mgr) :
    // FIXME (mw) we always assume a zero device here
    GLContextInterface(idx, 0), context_(ctx), manager_(mgr) {
}

/**
 * @brief Initialize EGL extensions that are required for operation
 */
bool initEGLExtensions() {
    // TODO (mw) check if this works under Android
    if (eglQueryDeviceAttribEXT == nullptr) {
        eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
        eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
        eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        eglQueryDisplayAttribEXT = (PFNEGLQUERYDISPLAYATTRIBEXTPROC)eglGetProcAddress("eglQueryDisplayAttribEXT");
        eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (!eglQueryDeviceAttribEXT || !eglQueryDeviceStringEXT || !eglQueryDevicesEXT || !eglQueryDisplayAttribEXT ||
            !eglGetPlatformDisplayEXT) return false;
    }
    return true;
}

} // fyusion::opengl namespace


#endif
// vim: set expandtab ts=4 sw=4:
