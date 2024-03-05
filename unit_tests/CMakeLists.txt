#----------------------------------------------------------------------------------
# Required packages for the unit tests
#----------------------------------------------------------------------------------

if (NOT APPLE)
  if(NOT ANDROID AND NOT WIN32)
    find_package(JPEG REQUIRED)
  endif()
else()
  # FIXME (mw) find something that works better under MacOS
  set(JPEG_INCLUDE_DIR /opt/homebrew/include)
  set(JPEG_LIBRARIES /opt/homebrew/lib/libturbojpeg)
endif()

#----------------------------------------------------------------------------------
# Misc compiler flags and default link libraries
#----------------------------------------------------------------------------------

if (NOT ANDROID AND NOT WIN32)
  set(CXX_LIBS stdc++ m)
endif()

set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)

#----------------------------------------------------------------------------------
# Add helper sources...
#----------------------------------------------------------------------------------

set(HELPER_SOURCES gltesthelpers.cpp layertestbase.cpp number_render.cpp)
set(HELPER_HEADERS gltesthelpers.h layertestbase.h number_render.h)
set(HELPERS ${HELPER_SOURCES} ${HELPER_HEADERS})

#----------------------------------------------------------------------------------
# Set FyuseNet libraries...
#----------------------------------------------------------------------------------

set(FYUSENET_LIBS $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
                  $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
                  $<TARGET_OBJECTS:common>)

set(IO_HELPERS ../samples/helpers/jpegio.cpp)

#----------------------------------------------------------------------------------
# Add GoogleTest framework...
#----------------------------------------------------------------------------------

include_directories(${gtest_SOURCE_DIR}/googletest/include ${gtest_SOURCE_DIR}/googlemock/include ${gtest_SOURCE_DIR}/googletest)
add_subdirectory(${gtest_SOURCE_DIR} ${gtest_BINARY_DIR})

#----------------------------------------------------------------------------------
# Test data
#----------------------------------------------------------------------------------

configure_file(../data/stylenet3x3_112_v3.dat ${CMAKE_CURRENT_BINARY_DIR}/stylenet3x3_112_v3.dat COPYONLY)
configure_file(../data/butterfly_1524x1856.jpg ${CMAKE_CURRENT_BINARY_DIR}/butterfly_1524x1856.jpg COPYONLY)
configure_file(../data/butterfly_512x624.jpg ${CMAKE_CURRENT_BINARY_DIR}/butterfly_512x624.jpg COPYONLY)

#----------------------------------------------------------------------------------
# Link libraries
#----------------------------------------------------------------------------------

if (APPLE)
  # Apple stuff
else(APPLE)
  if (USE_GLFW)
    set(GL_SYS_DEPS glfw X11)
  elseif (USE_EGL)
    # Android / EGL stuff
    if (ANDROID)
      set(GL_SYS_DEPS EGL GLESv3)
    else()
      set(GL_SYS_DEPS EGL GLESv2)
    endif()
  elseif(WIN32)
    # MS Windows
    set(GL_SYS_DEPS "")
  else()
    # Linux desktop GL / GLX
    set(GL_SYS_DEPS X11)
  endif()
endif()

if (ANDROID)
  set(DEFAULT_LIBS ${CXX_LIBS} gtest log)
else()
  set(DEFAULT_LIBS ${CXX_LIBS} gtest atomic)
endif()

#----------------------------------------------------------------------------------
# Misc include directories 
#----------------------------------------------------------------------------------

include_directories(${CMAKE_SOURCE_DIR} ${JPEG_INCLUDE_DIR})

#----------------------------------------------------------------------------------
# Actual test executables
#----------------------------------------------------------------------------------

if (NOT ANDROID)
  add_executable(networktests networktests.cpp ${BASE_SOURCES} ${HELPERS} ${SHADERMETA} ${SHADERRSRC})
  target_link_libraries(networktests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
  add_dependencies(networktests shader-meta)
endif()

if (USE_MULTITHREADING AND NOT ANDROID)
  add_executable(asynctests asynctests.cpp ../samples/samplenetworks/stylenet_base.cpp ../samples/samplenetworks/stylenet3x3.cpp 
                 ../samples/helpers/stylenet_provider.cpp ${BASE_SOURCES} ${HELPERS} ${IO_HELPERS} ${SHADERMETA} ${SHADERRSRC})
  target_link_libraries(asynctests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${JPEG_LIBRARIES} ${GL_SYS_DEPS})
  add_dependencies(asynctests shader-meta)
endif()

add_executable(arithtests arithtests.cpp ${BASE_SOURCES} ${HELPERS} ${SHADERMETA} ${SHADERRSRC})
target_link_libraries(arithtests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
add_dependencies(arithtests shader-meta)

add_executable(pooltests pooltests.cpp ${BASE_SOURCES} ${HELPERS} ${SHADERMETA} ${SHADERRSRC})
target_link_libraries(pooltests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
add_dependencies(pooltests shader-meta)

add_executable(convlayertests convlayertests.cpp ${BASE_SOURCES} ${HELPERS} ${SHADERMETA} ${SHADERRSRC})
target_link_libraries(convlayertests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
add_dependencies(convlayertests shader-meta)

add_executable(misctests misctests.cpp ${BASE_SOURCES} ${HELPERS} ${SHADERMETA} ${SHADERRSRC})
target_link_libraries(misctests ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${GL_SYS_DEPS})
add_dependencies(misctests shader-meta)

# vim: set expandtab ts=2 sw=2:
