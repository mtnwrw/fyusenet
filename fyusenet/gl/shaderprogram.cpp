//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Program
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "shaderprogram.h"
#include "shaderexception.h"
#include "uniformstate.h"
#include "shaderexception.h"
#include "../gpu/gfxcontextlink.h"
#include "../common/logging.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion {
namespace opengl {

#ifdef DEBUG
#define UNIFORM_BOUND_CHECK if (!bound_) { FNLOGW("Trying to set uniform to an unbound shader"); }
#else
#define UNIFORM_BOUND_CHECK
#endif

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Destructor
 *
 * @pre The GL context under which the shader program was created is bound to the current thread
 *
 * Removes the program object from the GL resources.
 */
ShaderProgram::~ShaderProgram() {
    shaders_.clear();
    if (handle_ != 0) {
        assertContext();
        glUseProgram(0);
        glDeleteProgram(handle_);
        handle_ = 0;
    }
    hasFragment_ = false;
    hasVertex_ = false;
    hasCompute_ = false;
}

/**
 * @brief Create a reference counted pointer instance for the shader program
 *
 * @param link Optional link to GL context as GfxContextLink, that is to be used to host the shader.
 *             If no context is supplied, the context current to this thread is used.
 *
 * @return Reference counted pointer (shared pointer) with an empty shader program
 *
 * Use this function to create an empty shader program instance. Currently the instances are simple
 * shared pointers.
 */
programptr ShaderProgram::createInstance(const fyusenet::GfxContextLink& link) {
    return programptr(new ShaderProgram(link));
}


/**
 * @brief Log the source of all shaders that are linked to this shader program
 *
 * This is a debug facility that will dump the source code of all shaders linked to that program
 * to the logging facility.
 */
void ShaderProgram::log() const {
    if (hasVertex_) {
        FNLOGD("Vertex Shader:");
        for (shaderptr shader : shaders_) {
            if (shader->getType() == GL_VERTEX_SHADER) shader->log();
        }
    }
    if (hasFragment_) {
        FNLOGD("Fragment Shader:");
        for (shaderptr shader : shaders_) {
            if (shader->getType() == GL_FRAGMENT_SHADER) shader->log();
        }
    }
#if (!defined(__APPLE__) && !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_WEBGL)) || defined(FYUSENET_USE_GLES_31)
    if (hasCompute_) {
        FNLOGD("Compute Shader:");
        for (shaderptr shader : shaders_) {
            if (shader->getType() == GL_COMPUTE_SHADER) shader->log();
        }
    }
#endif
}


/**
 * @brief Add shader to shader program
 *
 * @param shader Shared pointer to shader object that should be added
 *
 * This function adds the supplied \p shader to the list of shaders for this program object.
 * No compilation or linking is done at this point. See compile() and link() for further steps.
 */
void ShaderProgram::addShader(shaderptr shader) {
    if (shader.get()) {
        switch (shader->getType()) {
            case GL_FRAGMENT_SHADER:
                hasFragment_ = true;
                break;
            case GL_VERTEX_SHADER:
                hasVertex_ = true;
                break;
#if (!defined(__APPLE__) && !defined(FYUSENET_USE_EGL) && !defined(FYUSENET_USE_WEBGL)) || defined(FYUSENET_USE_GLES_31)
            case GL_COMPUTE_SHADER:
                hasCompute_ = true;
                break;
#endif
        }
        shaders_.push_back(shader);
    }
}


/**
 * @brief Bind shader program as active program object
 *
 * @param state Optional (raw) pointer to a UniformState object which will be automatically applied
 *              after the shader has been bound. Default is \c nullptr, which is a no-op with
 *              regards to the uniform variables in the shader.
 *
 * This function binds the shader program as active program object and sets the internal state of
 * the program to bound. When supplying a pointer to a valid UniformState object that was
 * initialized around this shader, the state from that object will be applied to the bound
 * shader. Make sure to unbind() a shader once you are done with it.
 *
 * @post #bound_ is set to \c true
 *
 * @throws ShaderException on debug builds in case there was an error binding the program.
 *
 * @see UniformState::applyState()
 */
void ShaderProgram::bind(UniformState *state) {
#ifdef DEBUG
    assert(handle_ != 0);
    if (bound_) {
        FNLOGW("Shader program was already bound, please check your code");
    }
    glGetError();                   // clear error state
#endif    
    glUseProgram(handle_);
#ifdef DEBUG
    int userr = glGetError();
#endif
    bound_ = true;
    if (state) state->applyState(this);
#ifdef DEBUG
    GLenum err = glGetError();
    if ((err != GL_NO_ERROR)||(userr != GL_NO_ERROR)) {
        bound_ = false;
        THROW_EXCEPTION_ARGS(ShaderException,"Cannot use shader with handle %d, glerr=0x%x userr=0x%X",handle_,err,userr);
    }
#endif
}

/**
 * @brief Unbind this shader program from the active program slot
 *
 * @param compress Optional argument, if set to \c true will not perform the GL call to unbind
 *                 the shader.
 *
 * @post #bound_ will be set to false
 */
void ShaderProgram::unbind(bool compress) {
#ifdef DEBUG
    if (!bound_) {
        FNLOGW("Shader program was not bound, please check your code");
    }
#endif
    bound_ = false;
    if (!compress) glUseProgram(0);
}


/**
 * @brief Map a uniform location/variable to a symbol
 *
 * @param name Name of the uniform variable to map
 *
 * @param symbol Symbol (integer number >= 0) that the location/variable should be assigned to
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * @return On succes, this function returns the location ID of the uniform variable and it
 *         returns -1 if the location was not found (result is for informational purposes and error
 *         detection)
 *
 * This function performs a lookup of the supplied \p name in the program object. Upon positive
 * result, it will associated the provided \p symbol with that location. In cases where the value
 * of a uniform variable changes rather often, the \p symbol can be used instead of the variable
 * name (which always performs a resolve) to save on GL API calls.
 *
 * The way \e optional variables are handled is that in case a variable is not optional and has
 * not been found, a ShaderException will be thrown. In case the variable was optional, no exception
 * will be thrown.
 *
 * @throws ShaderException if non-optional variable was not found or shader was not linked
 */
GLint ShaderProgram::mapUniformLocation(const char *name, int symbol, bool optional) {
    GLint loc = resolveLocation(name, true);
    if (loc == -1) {
        if (optional) return -1;
        THROW_EXCEPTION_ARGS(ShaderException,"Location %s cannot be mapped",name);
    }
    symbolMap_[symbol]=loc;
    return loc;
}


/**
 * @brief Set a symbol-mapped uniform variable (single integer)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified value.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false or the shader was not linked
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformValue(int symbol, GLint value, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformValue(it->second,value);
}


/**
 * @brief Set a symbol-mapped uniform variable (single float)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified value.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformValue(int symbol, GLfloat value, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformValue(it->second,value);
}


/**
 * @brief Set a symbol-mapped uniform variable (integer vec2)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2st component of 2-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified vector
 * components. In this case, the uniform should be an \c ivec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false or the shader was not linked
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec2(int symbol, GLint v0, GLint v1, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec2(it->second,v0,v1);
}


/**
 * @brief Set a symbol-mapped uniform variable (floating-point vec2)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2st component of 2-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified vector
 * components. In this case, the uniform should be an \c vec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec2(int symbol, GLfloat v0, GLfloat v1, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec2(it->second,v0,v1);
}


/**
 * @brief Set a symbol-mapped uniform variable (integer vec3)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2st component of 3-vec to set
 * @param v2 3st component of 3-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified vector
 * components. In this case, the uniform should be an \c ivec3 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec3(int symbol, GLint v0, GLint v1, GLint v2, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec3(it->second,v0,v1,v2);
}


/**
 * @brief Set a symbol-mapped uniform variable (floating-point vec3)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2st component of 3-vec to set
 * @param v2 3st component of 3-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified vector
 * components. In this case, the uniform should be an \c vec3 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec3(int symbol, GLfloat v0, GLfloat v1, GLfloat v2, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec3(it->second,v0,v1,v2);
}


/**
 * @brief Set a symbol-mapped uniform variable (floating-point vec4)
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param v0 1st component of 4-vec to set
 * @param v1 2st component of 4-vec to set
 * @param v2 3st component of 4-vec to set
 * @param v3 4st component of 4-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified vector
 * components. In this case, the uniform should be an \c vec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec4(int symbol, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec4(it->second,v0,v1,v2,v3);
}



/**
 * @brief Set a symbol-mapped uniform vec4 array
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num4Entries Number of vec4 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the symbol to the specified vector
 * components. In this case, the uniform array should be of \c vec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec4Array(int symbol, const GLfloat *data, int num4Entries, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec4Array(it->second,data,num4Entries);
}


/**
 * @brief Set a symbol-mapped uniform uvec4 array
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param data Pointer to unsigned integer data which should be set to the uniform array
 *
 * @param num4Entries Number of uvec4 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the symbol to the specified vector
 * components. In this case, the uniform array should be of \c uvec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformVec4Array(int symbol, const GLuint *data, int num4Entries, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformVec4Array(it->second,data,num4Entries);
}


/**
 * @brief Set a symbol-mapped single mat4 uniform
 *
 * @param symbol Symbol that references the uniform to set
 *
 * @param matrix Pointer to floating-point data which should be set to the matrix (16 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform matrix referenced by the symbol to the specified
 * components.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false, or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformMat4(int symbol, const GLfloat *matrix, bool transpose, bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformMat4(it->second,matrix,transpose);
}


/**
 * @brief Set a symbol-mapped array of mat4 uniforms
 *
 * @param symbol Symbol that references the uniform matrix array to set
 *
 * @param matrices Pointer to floating-point data which should be set to the matrix array
 *
 * @param numMatrices Number of matrices in the array
 *
 * @param transpose Set to \c true if the matrices are supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform matrix array referenced by the symbol to the specified
 * components.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false, or the shader was not linked.
 *
 * @see mapUniformLocation()
 */
void ShaderProgram::setMappedUniformMat4Array(int symbol, const GLfloat *matrices, int numMatrices, bool transpose,bool optional) {
    auto it = symbolMap_.find(symbol);
    if (it == symbolMap_.end()) {
        if (optional) return;
        THROW_EXCEPTION_ARGS(ShaderException,"Symbol %d is unknown symbol",symbol);
    }
    setUniformMat4Array(it->second, matrices ,numMatrices, transpose);
}




/**
 * @brief Set a symbol-mapped uniform variable (single integer)
 *
 * @param name Name of uniform variable in shader
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the symbol to the specified value.
 *
 * @throws ShaderException in case the variable was not found / used in the shader and the \p optional
 *         flag was set to \c false
 */
void ShaderProgram::setUniformValue(const char *name, GLint value, bool optional) {
    setUniformValue(resolveLocation(name,optional),value);
}


/**
 * @brief Set a symbol-mapped uniform variable (single integer)
 *
 * @param location Actual (GL) location of the uniform in the shader
 *
 * @param value Value to set to the uniform
 *
 * This function sets the uniform variable referenced by the symbol to the specified value.
 *
 * @throws ShaderException in case the shader was not linked
 */
void ShaderProgram::setUniformValue(GLint location, GLint value) {
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform1i(location,value);
    }
}


/**
 * @brief Set a uniform variable by ist name (single float)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param value Value to set to the uniform
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform variable referenced by the name to the specified value.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformValue(const char *name, GLfloat value, bool optional) {
    setUniformValue(resolveLocation(name,optional),value);
}



/**
 * @brief Set a uniform variable by its location (single float)
 *
 * @param location Location ID in GL shader
 *
 * @param value Value to set to the uniform
 *
 * This function sets the uniform variable referenced by its location to the specified value.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformValue(GLint location, GLfloat value) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform1f(location,value);
    }
}


/**
 * @brief Set a uniform variable by its name (integer vec2)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2nd component of 2-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformVec2(const char *name, GLint v0, GLint v1, bool optional) {
    setUniformVec2(resolveLocation(name,optional),v0,v1);
}


/**
 * @brief Set a uniform variable by its location ID (integer vec2)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2nd component of 2-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec2(GLint location, GLint v0, GLint v1) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException, "Shader program not linked");
        glUniform2i(location, v0, v1);
    }
}


/**
 * @brief Set a uniform variable by its name (floating-point vec2)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2nd component of 2-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformVec2(const char *name, GLfloat v0, GLfloat v1, bool optional) {
    setUniformVec2(resolveLocation(name,optional), v0, v1);
}

/**
 * @brief Set a uniform variable by its location ID (floating-point vec2)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 2-vec to set
 * @param v1 2nd component of 2-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec2(GLint location, GLfloat v0, GLfloat v1) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform2f(location, v0, v1);
    }
}


/**
 * @brief Set a uniform variable by its name (integer vec3)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2nd component of 3-vec to set
 * @param v2 3rd component of 3-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec3(const char *name, GLint v0, GLint v1, GLint v2, bool optional) {
    setUniformVec3(resolveLocation(name, optional), v0, v1, v2);
}


/**
 * @brief Set a uniform variable by its location ID (integer vec3)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2nd component of 3-vec to set
 * @param v2 3rd component of 3-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec3(GLint location, GLint v0, GLint v1, GLint v2) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform3i(location, v0, v1, v2);
    }
}


/**
 * @brief Set a uniform variable by its name (float vec3)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2nd component of 3-vec to set
 * @param v2 3rd component of 3-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformVec3(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, bool optional) {
    setUniformVec3(resolveLocation(name, optional), v0, v1, v2);
}


/**
 * @brief Set a uniform variable by its location ID (float vec3)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 3-vec to set
 * @param v1 2nd component of 3-vec to set
 * @param v2 3rd component of 3-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec3(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform3f(location, v0, v1, v2);
    }
}


/**
 * @brief Set a uniform variable by its name (integer vec4)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 4-vec to set
 * @param v1 2nd component of 4-vec to set
 * @param v2 3rd component of 4-vec to set
 * @param v3 4th component of 4-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformVec4(const char *name, GLint v0, GLint v1, GLint v2, GLint v3, bool optional) {
    setUniformVec4(resolveLocation(name,optional), v0, v1, v2, v3);
}


/**
 * @brief Set a uniform variable by its location ID (integer vec4)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 4-vec to set
 * @param v1 2nd component of 4-vec to set
 * @param v2 3rd component of 4-vec to set
 * @param v3 4th component of 4-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec4(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform4i(location, v0, v1, v2, v3);
    }
}


/**
 * @brief Set a uniform variable by its name (floating-point vec4)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param v0 1st component of 4-vec to set
 * @param v1 2nd component of 4-vec to set
 * @param v2 3rd component of 4-vec to set
 * @param v3 4th component of 4-vec to set
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the vector components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformVec4(const char *name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3, bool optional) {
    setUniformVec4(resolveLocation(name,optional), v0, v1, v2, v3);
}


/**
 * @brief Set a uniform vec variable by its location ID (float vec4)
 *
 * @param location Name of the uniform variable in the shader
 *
 * @param v0 1st component of 4-vec to set
 * @param v1 2nd component of 4-vec to set
 * @param v2 3rd component of 4-vec to set
 * @param v3 4th component of 4-vec to set
 *
 * This function sets the vector components of the uniform variable referenced by the location
 * to the specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformVec4(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform4f(location,v0,v1,v2,v3);
    }
}


/**
 * @brief Set a uniform matrix by its name (floating-point 3x3 matrix)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param matrix Pointer to float matrix data (9 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the matrix components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformMat3(const char *name,const GLfloat *matrix, bool transpose, bool optional) {
    setUniformMat3(resolveLocation(name,optional),matrix,transpose);
}


/**
 * @brief Set a uniform matrix by its location (floating-point 3x3 matrix)
 *
 * @param location GLSL ID/location of uniform variable
 *
 * @param matrix Pointer to float matrix data (9 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * This function sets the matrix components of the uniform variable referenced by the location to the
 * specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformMat3(GLint location, const GLfloat *matrix, bool transpose) {
    UNIFORM_BOUND_CHECK
    if (!matrix) THROW_EXCEPTION_ARGS(ShaderException,"Illegal matrix pointer %p supplied",matrix);
    if (location != -1) glUniformMatrix3fv(location, 1, transpose, matrix);
}


/**
 * @brief Set a uniform matrix by its name (floating-point 4x4 matrix)
 *
 * @param name Name of the uniform variable in the shader
 *
 * @param matrix Pointer to float matrix data (16 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the matrix components of the uniform variable referenced by the name to the
 * specified values.
 *
 * @throws ShaderException in case the variable name was not found in the linked shader and the
 *         \p optional flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformMat4(const char *name, const GLfloat *matrix, bool transpose, bool optional) {
    setUniformMat4(resolveLocation(name, optional), matrix, transpose);
}


/**
 * @brief Set a uniform matrix by its location (floating-point 4x4 matrix)
 *
 * @param location GLSL ID/location of uniform variable
 *
 * @param matrix Pointer to float matrix data (16 entries)
 *
 * @param transpose Set to \c true if the matrix is supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * This function sets the matrix components of the uniform variable referenced by the location to the
 * specified values.
 *
 * @throws ShaderException in case the shader was not linked.
 */
void ShaderProgram::setUniformMat4(GLint location, const GLfloat *matrix, bool transpose) {
    UNIFORM_BOUND_CHECK
    if (!matrix) THROW_EXCEPTION_ARGS(ShaderException,"Illegal matrix pointer %p supplied", matrix);
    if (location != -1) glUniformMatrix4fv(location, 1, transpose, matrix);
}


/**
 * @brief Set an array of mat4 uniforms by name (floating-point 4x4)
 *
 * @param name Name of the uniform array
 *
 * @param matrices Pointer to floating-point data which should be set to the matrix array
 *
 * @param numMatrices Number of matrices in the array
 *
 * @param transpose Set to \c true if the matrices are supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform matrix array referenced by the name to the specified
 * components.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformMat4Array(const char * name, const GLfloat *matrices, int numMatrices, bool transpose, bool optional) {
    setUniformMat4Array(resolveLocation(name, optional), matrices, numMatrices, transpose);
}


/**
 * @brief Set an array of mat4 uniforms by location (floating-point 4x4)
 *
 * @param location GLSL ID/location of the uniform array
 *
 * @param matrices Pointer to floating-point data which should be set to the matrix array
 *
 * @param numMatrices Number of matrices in the array
 *
 * @param transpose Set to \c true if the matrices are supplied in transposed form, which for
 *                  OpenGL means in row-major form. If the matrix is supplied in column-major
 *                  form, leave this at the default \c false value
 *
 * This function sets the uniform matrix array referenced by the location to the specified
 * components.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false, or the shader was not linked.
 */
void ShaderProgram::setUniformMat4Array(GLint location, const GLfloat *matrices, int numMatrices, bool transpose) {
    UNIFORM_BOUND_CHECK
    if (!matrices) THROW_EXCEPTION_ARGS(ShaderException,"Illegal matrix pointer %p supplied", matrices);
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniformMatrix4fv(location, numMatrices, transpose, matrices);
    }
}


/**
 * @brief Set a uniform vec4 array by name (floating-point)
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num4Entries Number of vec4 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec4Array(const char *name, const GLfloat *data, int num4Entries, bool optional) {
    setUniformVec4Array(resolveLocation(name, optional), data, num4Entries);
}


/**
 * @brief Set a uniform vec4 array by GLSL location (floating-point)
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num4Entries Number of vec4 entries in the \p data array
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec4Array(GLint location, const GLfloat *data, int num4Entries) {
    UNIFORM_BOUND_CHECK
    if (!data) THROW_EXCEPTION_ARGS(ShaderException,"Illegal data pointer %p supplied",data);
    if (location != -1) glUniform4fv(location, num4Entries, data);
}


/**
 * @brief Set a uniform vec4 array by name (unsigned 32-bit integer)
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to unsigned 32-bit integer data which should be set to the uniform array
 *
 * @param num4Entries Number of uvec4 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c uvec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec4Array(const char *name, const GLuint *data, int num4Entries, bool optional) {
    setUniformVec4Array(resolveLocation(name, optional) ,data, num4Entries);
}


/**
 * @brief Set a uniform vec4 array by GLSL location (unsigned 32-bit integer)
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to unsigned integer data which should be set to the uniform array
 *
 * @param num4Entries Number of uvec4 entries in the \p data array
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c uvec4 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec4Array(GLint location, const GLuint *data, int num4Entries) {
    UNIFORM_BOUND_CHECK
    if (!data) THROW_EXCEPTION_ARGS(ShaderException,"Illegal data pointer %p supplied",data);
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform4uiv(location, num4Entries, data);
    }
}


/**
 * @brief Set a uniform vec3 array by name (floating-point)
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to floating-point which should be set to the uniform array
 *
 * @param num3Entries Number of vec3 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec3 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec3Array(const char *name, const GLfloat *data, int num3Entries, bool optional) {
    setUniformVec3Array(resolveLocation(name, optional), data, num3Entries);
}


/**
 * @brief Set a uniform vec3 array by GLSL location (floating-point)
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param num3Entries Number of vec3 entries in the \p data array
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec3 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec3Array(GLint location, const GLfloat *data, int num3Entries) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform3fv(location, num3Entries, data);
    }
}


/**
 * @brief Set a uniform vec2 array by name (signed 32-bit integer)
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit integer data which should be set to the uniform array
 *
 * @param num2Entries Number of ivec2 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c ivec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec2Array(const char *name, const GLint *data, int num2Entries, bool optional) {
    setUniformVec2Array(resolveLocation(name, optional), data, num2Entries);
}


/**
 * @brief Set a uniform vec2 array by GLSL location (signed 32-bit integer)
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to unsigned integer data which should be set to the uniform array
 *
 * @param num2Entries Number of ivec2 entries in the \p data array
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c ivec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec2Array(GLint location, const GLint *data, int num2Entries) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform2iv(location, num2Entries, data);
    }
}

/**
 * @brief Set a uniform vec2 array by name (32-bit floating point)
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating point data which should be set to the uniform array
 *
 * @param num2Entries Number of vec2 entries in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec2Array(const char *name, const GLfloat *data, int num2Entries, bool optional) {
    setUniformVec2Array(resolveLocation(name, optional), data, num2Entries);
}


/**
 * @brief Set a uniform vec2 array by GLSL location (32-bit floating point)
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating point data which should be set to the uniform array
 *
 * @param num2Entries Number of vec2 entries in the \p data array
 *
 * This function sets the uniform array referenced by the \p name to the specified vector
 * components. In this case, the uniform array should be of \c vec2 type.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformVec2Array(GLint location, const GLfloat *data, int num2Entries) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform2fv(location, num2Entries, data);
    }
}

/**
 * @brief Set a uniform float array by name
 *
 * @param name Name of uniform array variable to set data to
 *
 * @param data Pointer to 32-bit floating-point data which should be set to the uniform array
 *
 * @param numEntries Number of floating-point values in the \p data array
 *
 * @param optional Setting this to \c true indicates that the variable we are looking for is
 *                 optional (default is \c false).
 *
 * This function sets the uniform float array referenced by the \p name to the specified values.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformArray(const char *name, const GLfloat *data, int numEntries, bool optional) {
    setUniformArray(resolveLocation(name, optional), data, numEntries);
}


/**
 * @brief Set a uniform float array by GLSL location
 *
 * @param location GLSL ID/location of uniform array variable to set data to
 *
 * @param data Pointer to floating-point data which should be set to the uniform array
 *
 * @param numEntries Number of floats in the \p data array
 *
 * This function sets the uniform float array referenced by the \p name to the specified values.
 *
 * @throws ShaderException in case the variable was not successfully mapped and the \p optional
 *         flag was set to \c false , or the shader was not linked.
 */
void ShaderProgram::setUniformArray(GLint location, const GLfloat *data, int numEntries) {
    UNIFORM_BOUND_CHECK
    if (location != -1) {
        if (!isLinked()) THROW_EXCEPTION_ARGS(ShaderException,"Shader program not linked");
        glUniform1fv(location, numEntries, data);
    }
}


/**
 * @brief Binds an index to a vertex-shader attribute
 *
 * @param name The name of the attribute in the vertex shader
 *
 * @param index The location/index to bind it to
 *
 * @warning Binding an attribute location without (re-)linking the shader \e afterwards is a no-op
 *          (see GL reference manual).
 */
void ShaderProgram::bindAttributeLocation(const char *name, GLuint index) {
    ensureExistence();
    if (!isLinked()) {
        glBindAttribLocation(handle_,index,name);
    }
}


/**
 * @brief Binds a given index to a uniform block
 *
 * @param name Name of the block interface
 *
 * @param bindingIndex Index to bind the block to
 *
 * This function
 *
 * @see https://www.khronos.org/opengl/wiki/Uniform_Buffer_Object
 *
 * @throws ShaderException in case the interface block with the provided \p name was not found or
 *         (in case of debug builds) a general GL error occured.
 */
void ShaderProgram::bindIndexToShaderBuffer(const char *name, int bindingIndex) {
    GLuint block = glGetUniformBlockIndex(handle_, name);
    if (block == GL_INVALID_INDEX) THROW_EXCEPTION_ARGS(ShaderException,"Cannot obtain block index for \"%s\"",name);
#ifdef DEBUG
    glGetError();
#endif
    glUniformBlockBinding(handle_, block, bindingIndex);
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(ShaderException,"Unable to establish block binding (glerr=0x%x)",err);
#endif
}


/**
 * @brief ShaderProgram::compile
 *
 *
 * @throws ShaderException in case the compilation went wrong or there was no program object
 */
void ShaderProgram::compile() {
    if (!isLinkable()) THROW_EXCEPTION_ARGS(ShaderException,"Not enough shader types for linking");
    for (auto ii=shaders_.begin(); ii!=shaders_.end(); ++ii) {
        if (!(*ii)->isCompiled()) (*ii)->compile();
    }
    ensureExistence();
    if (handle_ == 0) THROW_EXCEPTION_ARGS(ShaderException,"Cannot create shader program");
}


/**
 * @brief Link shader program
 *
 * This function first checks if the program is already linked and does nothing in that case.
 * Otherwise it
 *
 * @throws ShaderException in case compilation/linking goes wrong
 */
void ShaderProgram::link() {
    if (isLinked()) return;
    assertContext();
    compile();
    glGetError();
    for (auto ii=shaders_.begin(); ii!=shaders_.end(); ++ii) {
        glAttachShader(handle_, (*ii)->getHandle());
        GLint err = glGetError();
        if (err != GL_NO_ERROR) THROW_EXCEPTION_ARGS(ShaderException,"Unable to attach shader with handle %d, glerr=0x%x",(*ii)->getHandle(),err);
    }
    GLint status=GL_FALSE;
    glLinkProgram(handle_);
    glGetProgramiv(handle_,GL_LINK_STATUS,&status);
    if (status == GL_FALSE) {
#ifdef DEBUG
        FNLOGE("Shader linker error");
        logError();
        FNLOGE("Logging shaders...");
        for (auto ii=shaders_.begin(); ii!=shaders_.end(); ++ii) {
            (*ii)->log();
        }
#endif
        THROW_EXCEPTION_ARGS(ShaderException,"Unable to link shaders to program, status is 0x%x (expected 0x%X)",status,GL_TRUE);
    }
    linked_ = true;
}


/**
 * @brief Lookup uniform variable in shader
 *
 * @param varName Variable name
 * @param silent Flag that controls whether or not an exception is thrown on failure
 *
 * @return ID (GLSL location) of variable with provided \p varName
 *
 * @throws ShaderException in case \p silent was set to \c false and the uniform variable was not
 *         found in the shader program.
 */
GLint ShaderProgram::resolveLocation(const char *varName, bool silent) const {
    assert(handle_);
    GLint loc = glGetUniformLocation(handle_, varName);
    if ((loc < 0) && (!silent)) {
        THROW_EXCEPTION_ARGS(ShaderException,"Cannot resolve location \"%s\" in shader %d", varName, handle_);
    }
    return loc;
}





/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param context Link to GL context to work under
 */
ShaderProgram::ShaderProgram(const fyusenet::GfxContextLink & context) : GfxContextTracker() {
    setContext(context);
    bound_ = false;
    handle_ = 0;
    hasFragment_ = false;
    hasVertex_ = false;
    hasCompute_ = false;
    userFlags_ = 0;
    hash_ = 0;
    linked_ = false;
}


/**
 * @brief Returns a vector of OpenGL shader handles for all shaders in this program
 *
 * @return Vector of GL shader handles used in this program
 */
std::vector<GLuint> ShaderProgram::getShaderHandles() const {
    std::vector<GLuint> result;
    for (auto ii=shaders_.begin(); ii != shaders_.end(); ++ii) {
        if (!(*ii)->isCompiled()) {
            THROW_EXCEPTION_ARGS(ShaderException,"Please compile shaders before extracting handles");
        }
        result.push_back((*ii)->handle_);
    }
    return result;
}


/**
 * @brief Make sure that a program handle exist (create one if not)
 */
void ShaderProgram::ensureExistence() {
    if (handle_ == 0) {
        handle_ = glCreateProgram();
    }
}


/**
 * @brief Log link errors to log facility
 */
void ShaderProgram::logError() const {
    GLint loglen=0;
    glGetProgramiv(handle_,GL_INFO_LOG_LENGTH,&loglen);
    if (loglen>0) {
        char *log = new char[loglen];
        glGetProgramInfoLog(handle_,loglen,&loglen,log);
        char *ptr = log;
        while ((ptr)&&(ptr[0]!=0)) {
            char *nptr = strpbrk(ptr,"\n\r");

            if (nptr) {
                int i=1;
                while ((nptr[i]==10)||(nptr[i]==13)) {
                    nptr[i]=0;
                    nptr++;
                }
                FNLOGI("%s",ptr);

                if (nptr == ptr) ptr=nullptr;
                else ptr=nptr;
            } else {
                FNLOGI("%s",ptr);
                ptr=nullptr;
            }
        }
        delete [] log;
    } else {
        FNLOGI("<no linker log>");
    }
}



} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
