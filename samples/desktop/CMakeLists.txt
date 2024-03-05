#----------------------------------------------------------------------------------
# Required packages for the desktop samples
#----------------------------------------------------------------------------------

if (NOT WIN32)
  find_package(JPEG REQUIRED)
endif()

#----------------------------------------------------------------------------------
# Misc compiler flags and default link libraries (esp. when using ninja as backend)
#----------------------------------------------------------------------------------

if (NOT WIN32)
  set(DEFAULT_LIBS stdc++ pthread m atomic)
else()
  set(DEFAULT_LIBS "")
endif()

set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)

include_directories(${JPEG_INCLUDE_DIR})

#----------------------------------------------------------------------------------
# Add sources..
#----------------------------------------------------------------------------------

set(STYLE_SOURCES stylenet.cpp ../samplenetworks/stylenet_base.cpp ../samplenetworks/stylenet_base.h
                  ../samplenetworks/stylenet3x3.cpp ../samplenetworks/stylenet3x3.h
                  ../samplenetworks/stylenet9x9.cpp ../samplenetworks/stylenet9x9.h
                  ../helpers/stylenet_provider.cpp ../helpers/stylenet_provider.h
                  ../helpers/jpegio.cpp ../helpers/jpegio.h)

set(RESNET_SOURCES resnet.cpp ../samplenetworks/resnet50.cpp ../samplenetworks/resnet50.h
                  ../helpers/resnet_provider.h ../helpers/resnet_provider.cpp
                  ../helpers/jpegio.cpp ../helpers/jpegio.h)

#----------------------------------------------------------------------------------
# Set FyuseNet libraries...
#----------------------------------------------------------------------------------

set(FYUSENET_LIBS $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
                  $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
                  $<TARGET_OBJECTS:common>)

#----------------------------------------------------------------------------------
# Actual test executables
#----------------------------------------------------------------------------------

add_executable(stylenet ${STYLE_SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(stylenet shader-meta)

add_executable(resnet ${RESNET_SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(resnet shader-meta)

if (WIN32)
  add_dependencies(stylenet ${JPEG_LIBRARIES})
  add_dependencies(resnet ${JPEG_LIBRARIES})
endif()

if (USE_CUSTOM AND NOT ANDROID_ABI AND NOT BUILD_TARGET STREQUAL "Web")
  add_executable(llama llama.cpp ${SHADERMETA} ${SHADERRSRC}
                 ../samplenetworks/llama_4bit.cpp ../samplenetworks/llama_4bit.h
                 ../helpers/zipwalker.cpp ../helpers/zipwalker.h
                 ../helpers/llama_4bit_params.cpp ../helpers/llama_4bit_params.h
                 ../helpers/sentencepiece_tokenizer.cpp ../helpers/sentencepiece_tokenizer.h)
  add_dependencies(llama shader-meta)
endif()


if (APPLE)
  # apple stuff
else(APPLE)
  if (USE_GLFW)
    set(GL_SYS_DEPS glfw X11)
  elseif (USE_EGL)
    # EGL stuff
    set(GL_SYS_DEPS EGL GLESv2)
  elseif (WIN32)
    set(GL_SYS_DEPS "")
  else()
    # linux desktop GL
    set(GL_SYS_DEPS X11)
  endif()
endif()

target_link_libraries(stylenet PRIVATE ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${JPEG_LIBRARIES} ${GL_SYS_DEPS})
target_link_libraries(resnet PRIVATE ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${JPEG_LIBRARIES} ${GL_SYS_DEPS})
if (USE_CUSTOM)
  target_link_libraries(llama PRIVATE ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${JPEG_LIBRARIES} ${GL_SYS_DEPS})
endif()

# vim: set expandtab ts=2 sw=2:
