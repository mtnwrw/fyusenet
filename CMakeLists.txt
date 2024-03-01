cmake_minimum_required(VERSION 3.20)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0")
  cmake_policy(SET CMP0072 NEW) # Prefer GLVND over GL Legacy
endif()

project(fyusenet LANGUAGES CXX)

set(BUILD_TARGET "Desktop" CACHE STRING "Target platform to build on")
set(MT_DEFAULT ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
  set(BUILD_TARGET "Web")
  set(MT_DEFAULT OFF)
endif()

if (APPLE)
  set(MT_DEFAULT OFF)
endif()

option(BUILD_LIBRARY "Build shared libraries for SDK use" OFF)
option(BUILD_TESTS "Build unit-tests" ON)
option(BUILD_SAMPLES "Build sample networks" ON)
option(USE_MULTITHREADING "Enable multi-threading" ${MT_DEFAULT})
option(USE_EGL "Use embedded GL" OFF)
option(BUILD_DOCS "Build doxygen documentation" OFF)
option(HIGH_PRECISION "Use 32-bit FP precision on buffers / shaders" OFF)
option(USE_CUSTOM "Use custom layers" ON)

if (APPLE)
  message(STATUS "Forcing high-precision shaders/buffers on MacOS/CoreGL")
  set(HIGH_PRECISION ON)
endif()

if (ANDROID_ABI)
  set(BUILD_TARGET "Android")
endif()

if (BUILD_TARGET STREQUAL "Android")
  set(USE_EGL ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_TARGET STREQUAL "Desktop")
  option(USE_GLFW "Use GLFW (use this for using NVIDIA Insight for example)" OFF)
endif()

if (BUILD_TARGET STREQUAL "Desktop")
  if (USE_EGL STREQUAL ON)
    find_package(OpenGL REQUIRED)
  else
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
    find_package(OpenGLES REQUIRED)
    find_package(EGL REQUIRED)
  endif()
endif()

if (APPLE OR BUILD_TARGET STREQUAL "Web")
  # Force multithreading off for Apple platforms and WebGL platforms (for now)
  set(USE_MULTITHREADING OFF)
  set(BUILD_TESTS OFF)
endif()

if (NOT USE_MULTITHREADING)
  message(STATUS "Disabling multithreading")
endif()

if (BUILD_LIBRARY)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# C++ standard to use
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags, general C++ compiler
if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_SANITIZE "/Od /Zi /DDEBUG /fsanitize=address")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall  -O0 -g -DDEBUG -ffast-math -fPIC")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -fPIC")
endif()

# Compiler and linker flags for WebAssembly / WebGL
set(WASM_GEN_LINKER_FLAGS "-sWASM=1 -sUSE_WEBGL2=1 -sMAX_WEBGL_VERSION=2 -sALLOW_MEMORY_GROWTH -sEXPORTED_RUNTIME_METHODS=\"stringToUTF8,lengthBytesUTF8\" -sEXPORTED_FUNCTIONS=\"[_malloc]\"")

set(CMAKE_CXX_FLAGS_EMSCRIPTEN_DEBUG "-Wall -gsource-map -flto -ffast-math -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_RELEASE "-ffast-math -flto -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_SMALL "-ffast-math -flto -Oz -DNDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_PROFILE "-g --profiling -ffast-math -flto -O3 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_DEBUG "${CMAKE_CXX_FLAGS_EMSCRIPTEN_DEBUG} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=1 -s FILESYSTEM=0 -s NO_DISABLE_EXCEPTION_CATCHING --source-map-base http://localhost:8000/ -s ERROR_ON_UNDEFINED_SYMBOLS=0")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_RELEASE "${CMAKE_CXX_FLAGS_EMSCRIPTEN_RELEASE} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_SMALL "${CMAKE_CXX_FLAGS_EMSCRIPTEN_SMALL} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_PROFILE "${CMAKE_CXX_FLAGS_EMSCRIPTEN_PROFILE} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")

include(FetchContent)

add_compile_definitions(FYUSENET_GL_BACKEND FYUSENET_INTERNAL FYUSENET_CPU_FLOAT_ONLY)

if (BUILD_TESTS)
  if (NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(WARNING "Explicitly setting the DEBUG preprocessor definition for unit-testing.\
                     You should not use the results of this build as release binaries.\
                     Disable the unit tests for release builds")
    add_compile_definitions(DEBUG)
  endif()
endif()

if (HIGH_PRECISION)
  add_compile_definitions(HIGH_PRECISION)
endif()

if (APPLE)
  add_compile_definitions(GL_SILENCE_DEPRECATION)
endif()

if (WIN32)
  find_package(Python COMPONENTS Interpreter)
endif()

if (BUILD_TESTS)
  message(STATUS "Using GoogleTest 1.14.0 from GitHub repository")
  FetchContent_Declare(gtest
          GIT_REPOSITORY https://github.com/google/googletest/
          GIT_TAG v1.14.0)

  FetchContent_GetProperties(gtest)
  if (NOT gtest_POPULATED)
    FetchContent_Populate(gtest)
  endif()
endif()

if (WIN32)
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
  endif()
  include(ExternalProject)

  # Specify GLEW as external project in order to have up-to-date GL headers
  # and correct interfacing to the GL library on the system.
  message(STATUS "Using GLEW 2.2.0 from GitHub repository")
  FetchContent_Declare(glew_fetch
          URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip
          URL_HASH SHA256=EA6B14A1C6C968D0034E61FF6CB242CFF2CE0EDE79267A0F2B47B1B0B652C164)
  FetchContent_GetProperties(glew_fetch)
  if (NOT glew_fetch_POPULATED)
    FetchContent_Populate(glew_fetch)
  endif()

  ExternalProject_Add(glew
          PREFIX ${CMAKE_BINARY_DIR}/_deps/
          SOURCE_DIR ${glew_fetch_SOURCE_DIR}
          CONFIGURE_COMMAND ""
          BUILD_COMMAND ""
          INSTALL_COMMAND ""
  )
  add_compile_definitions(GLEW_STATIC)
  include_directories(AFTER ${glew_fetch_SOURCE_DIR}/include)
  list(APPEND OPENGL_LIBRARIES ${glew_fetch_SOURCE_DIR}/lib/Release/x64/glew32s.lib)

  # Download libjpeg from a GitHub repo and use it directly
  message(STATUS "Using libjpeg from GitHub repository")
  FetchContent_Declare(
          libjpeg
          GIT_REPOSITORY https://github.com/csparker247/jpeg-cmake
          GIT_TAG v1.2.1
  )
  FetchContent_GetProperties(libjpeg)
  if (NOT libjpeg_POPULATED)
    FetchContent_Populate(libjpeg)
  endif()
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
  add_subdirectory(${libjpeg_SOURCE_DIR})
  set(JPEG_INCLUDE_DIR ${libjpeg_SOURCE_DIR}/libjpeg)
  set(JPEG_LIBRARIES jpeg_static)
endif()

if (NOT USE_EGL AND NOT APPLE)
  add_compile_definitions(GL_GLEXT_PROTOTYPES)
endif()

if (USE_GLFW)
  if (WIN32)
    message(ERROR "GLFW is not available under MS Windows builds")
  endif()
  add_compile_definitions(FYUSENET_USE_GLFW)
endif()

if (USE_EGL)
  if (WIN32)
    message(ERROR "EGL is not available under MS Windows builds")
  endif()
  add_compile_definitions(FYUSENET_USE_EGL)
endif()

if (BUILD_TARGET STREQUAL "Web")
  add_compile_definitions(FYUSENET_USE_WEBGL)
endif()

if (USE_MULTITHREADING)
  add_compile_definitions(FYUSENET_MULTITHREADING)
endif()

if (WIN32)
  add_compile_definitions(NOMINMAX)
endif()

add_subdirectory(fyusenet)

if (BUILD_TESTS AND NOT (BUILD_TARGET STREQUAL "Web"))
  add_subdirectory(unit_tests)
endif()

if (BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if (BUILD_DOCS)
  add_subdirectory(doxygen)
endif()

