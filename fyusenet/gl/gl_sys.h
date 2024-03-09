//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL System-Specific Headers
// Creator: Ferry Tanu
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

/**
 * @file gl_sys.h
 *
 * @warning This file should not be included directly by API users, if you want access to
 *          graphics contexts, use gpu/gfxcontextlink.h instead. In case of implementing a custom
 *          layer, it is fine to include this file directly.
 */

#ifndef FYUSENET_GL_BACKEND
#define FYUSENET_GL_BACKEND
#endif

#ifndef FYUSENET_INTERNAL
#cmakedefine FYUSENET_USE_EGL
#cmakedefine FYUSENET_USE_GLFW
#cmakedefine FYUSENET_USE_WEBGL
#ifndef FYUSENET_MULTITHREADING
#cmakedefine FYUSENET_MULTITHREADING
#endif
#endif

#if defined(FYUSENET_USE_EGL)
// headers for Android and/or EGL/GLES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#ifdef FYUSENET_USE_GLES_31
#include <GLES3/gl31.h>
#endif
#include <GLES2/gl2ext.h>
#endif

#if defined(__linux__) && !defined(FYUSENET_USE_EGL)
// headers for linux
#ifdef FYUSENET_USE_GLFW
#include <GLFW/glfw3.h>
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#ifdef __APPLE__
// headers for MacOS
#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLContext.h>
#include <OpenGL/gl3.h>
#include <OpenGL/OpenGL.h>
#endif
#ifdef FYUSENET_USE_WEBGL
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>
#include <GLES/gl.h>
#include <GLES3/gl3.h>
#endif
#if defined(WIN32) || defined(WIN64)
#include <GL/glew.h>
#endif

// vim: set expandtab ts=4 sw=4:
