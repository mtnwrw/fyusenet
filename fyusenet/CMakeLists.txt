add_subdirectory(common)
add_subdirectory(base)
add_subdirectory(gl)
add_subdirectory(gpu)
add_subdirectory(cpu)

set(SHADERRSRC ${SHADERRSRC} PARENT_SCOPE)
set(SHADERMETA ${SHADERMETA} PARENT_SCOPE)
set(FYUSENET_HEADER fyusenet.h)

include_directories( ../ )

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND BUILD_TARGET STREQUAL "Desktop")
  set(LIBNAME "fyusenet_debug")
else()
  set(LIBNAME "fyusenet")
endif()

if (APPLE)
  # apple stuff
else(APPLE)
  if (USE_GLFW)
    set(GL_SYS_DEPS glfw X11)
  elseif (USE_EGL)
    # Android / EGL stuff
    # fixme
    #if (ANDROID)
    #  set(GL_SYS_DEPS EGL GLESv3)
    #else()
    #  set(GL_SYS_DEPS EGL GLESv2)
    #endif()
  else()
    # linux desktop GL
    set(GL_SYS_DEPS X11)
  endif()
endif()

#----------------------------------------------------------------------------
# FyuseNet shared library
#----------------------------------------------------------------------------

if (BUILD_TARGET STREQUAL "Web")
  add_library(${LIBNAME} STATIC ${SHADERMETA} ${SHADERRSRC}
              $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
              $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
              $<TARGET_OBJECTS:common>)
  add_dependencies(${LIBNAME} shader-meta)
  target_link_libraries(${LIBNAME} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
  set(FYUSENET_LIB ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.a PARENT_SCOPE)
endif()

if (BUILD_LIBRARY OR BUILD_TARGET STREQUAL "Android")
  add_library(${LIBNAME} SHARED ${SHADERMETA} ${SHADERRSRC}
              $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
              $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
              $<TARGET_OBJECTS:common>)
  set_target_properties(${LIBNAME} PROPERTIES SOVERSION 1)
  set_property(TARGET ${LIBNAME} PROPERTY POSITION_INDEPENDENT_CODE 1)
  add_dependencies(${LIBNAME} shader-meta)
  target_link_libraries(${LIBNAME} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
  install(TARGETS ${LIBNAME} DESTINATION lib)
endif()



if (BUILD_TARGET STREQUAL "Android")
  target_link_libraries(${LIBNAME} android log)
  if (BUILD_SAMPLES)
      set(LIBFILENAME ${CMAKE_SHARED_LIBRARY_PREFIX}${LIBNAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
      add_custom_command(TARGET ${LIBNAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy  ${LIBFILENAME} ${CMAKE_SOURCE_DIR}/samples/android/app/src/main/jniLibs/${ANDROID_ABI})
      configure_file(${CMAKE_SOURCE_DIR}/data/stylenet3x3_112_v3.dat ${CMAKE_SOURCE_DIR}/samples/android/app/src/main/assets/style.bin COPYONLY)
  endif()
endif()


#----------------------------------------------------------------------------
# Misc installation files (headers etc.)
#----------------------------------------------------------------------------

configure_file(${FYUSENET_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/fyusenet.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${FYUSENET_HEADER} DESTINATION include/fyusenet)


# vim: set expandtab ts=2 sw=2:
