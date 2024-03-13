if (BUILD_TARGET STREQUAL "Web")
  set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/webgl/glcontext_webgl.cpp)
else()
  if (USE_EGL)
    set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/egl/glcontext_egl.cpp)
  else()
    if (APPLE)
      if (USE_GLFW)
        set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/glfw/glcontext_glfw.cpp)
      else()
        set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cgl/glcontext_cgl.cpp)
      endif()
    elseif(WIN32)
      set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/wgl/glcontext_wgl.cpp)
    else()
      if (USE_GLFW)
        set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/glfw/glcontext_glfw.cpp)
      else()
        set(GL_CTX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/glx/glcontext_glx.cpp)
      endif(USE_GLFW)
    endif(APPLE)
  endif(USE_EGL)
endif()

set(SHADERRSRC ${CMAKE_CURRENT_SOURCE_DIR}/shaderresource.cpp PARENT_SCOPE)

file(GLOB GL_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

list(REMOVE_ITEM GL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaderresource.cpp)

if (NOT USE_MULTITHREADING)
  list(REMOVE_ITEM GL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/asyncpool.cpp)
  list(REMOVE_ITEM GL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/asyncpool.h)
endif()

if (APPLE OR BUILD_TARGET STREQUAL "Web" OR (USE_EGL AND NOT USE_GLES_31))
  list(REMOVE_ITEM GL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/computedispatch.cpp)
endif()

set(GL_SOURCES ${GL_SOURCES})
set(GL_CTX_SOURCES ${GL_CTX_SOURCES})

add_library(opengl STATIC ${GL_SOURCES} ${GL_CTX_SOURCES})

if (WIN32)
  add_dependencies(opengl glew)
endif()

#----------------------------------------------------------------------------
# Installation files
#----------------------------------------------------------------------------

file(GLOB FYUSENET_GL_HEADERS *.h)
list(REMOVE_ITEM FYUSENET_GL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/gl_sys.h)
list(REMOVE_ITEM FYUSENET_GL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/glcontext.h)
install(FILES ${FYUSENET_GL_HEADERS} DESTINATION include/fyusenet/gl)

configure_file(gl_sys.h ${CMAKE_CURRENT_BINARY_DIR}/gl_sys.h )
configure_file(glcontext.h ${CMAKE_CURRENT_BINARY_DIR}/glcontext.h )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gl_sys.h DESTINATION include/fyusenet/gl)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/glcontext.h DESTINATION include/fyusenet/gl)

if (USE_EGL)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/egl/eglhelper.h DESTINATION include/fyusenet/gl/egl)
endif()


# vim: set expandtab ts=2 sw=2:
