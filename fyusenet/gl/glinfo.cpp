//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Information Object
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <string>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "fbo.h"
#include "glinfo.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::opengl {


GLInfo GLInfo::instance_;
bool GLInfo::initialized_ = false;

//-------------------------------------- Local Definitions -----------------------------------------

#if defined(WIN32) || defined(WIN64)
static bool string_match(const char * haystack, const char *needle) {
    std::string a(haystack);
    std::string b(needle);
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a.find(b) != std::string::npos;
}
#else
static bool string_match(const char *haystack, const char *needle) {
    return (strcasestr(haystack, needle) != nullptr);
}
#endif

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Initialize / instantiate GLInfo singleton object
 *
 * @param chatty If set to \c true, a lot of information is logged into the logging facility
 *
 * This function queries the GL subsystem on the system and stores the results in a GLInfo
 * singleton object that is used by other classes in this abstraction layer.
 */
void GLInfo::init(bool chatty) {
    if (initialized_) return;
    instance_.queryVersion();
    instance_.queryChipset();
    instance_.queryShader();
    instance_.queryExtensions();
    initialized_=true;
    if (instance_.shaderVersion_ >= GLSL_100_ES) {
        instance_.bindingSupport_ = (instance_.shaderVersion_ >= GLSL_310_ES);
    } else {
        instance_.bindingSupport_ = (instance_.shaderVersion_ >= GLSL_430);
    }
    if (chatty) printInfo();
}


/**
 * @brief Check if GL system features a specified GL extnsion
 *
 * @param extension Extension to query
 *
 * @retval true if extension was present
 * @retval false otherwise
 */
bool GLInfo::hasExtension(const char *extension) {
    if (!extension) return false;
    if (!initialized_) THROW_EXCEPTION_ARGS(GLException,"GLInfo not initialized");
    if (instance_.extensions_.empty()) return false;
    const char *ptr = strstr(instance_.extensions_.c_str(),extension);
    return (ptr != nullptr);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Idle constructor
 */
GLInfo::GLInfo() {
}


/**
 * @brief Query all GL extensions present on the system
 *
 * Queries the extension list of the system and stores it in the GLInfo singleton
 */
void GLInfo::queryExtensions() {
    const GLubyte *oomph = glGetString(GL_EXTENSIONS);
    if (oomph) {
        extensions_ = std::string((const char *)glGetString(GL_EXTENSIONS));
    } else {
        GLint extcnt = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &extcnt);
        for (int i = 0; i < extcnt; i++) {
            extensions_.append((const char *) glGetStringi(GL_EXTENSIONS, i)).append(" ");
        }
    }
}

/**
 * @brief
 */
void GLInfo::queryVersion() {
    // not really elegant code, but does the job
    int maj = 0,minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION,&maj);
    glGetIntegerv(GL_MINOR_VERSION,&minor);
    version_ = UNSUPPORTED;
#ifdef FYUSENET_USE_WEBGL
    if (maj > 1) version_ = WEBGL_2_0;
    else version_ = WEBGL_1_0;
#elif defined(FYUSENET_USE_EGL)
    if (maj > 3) version_=GLES_3_3;
    else if (maj == 2) version_ = GLES_2_0;
    else if (maj == 3) {
        switch (minor) {
            case 0:
                version_=GLES_3_0;
                break;
            case 1:
                version_=GLES_3_0;
                break;
            case 2:
                version_=GLES_3_2;
            case 3:
                // intentional fallthrough
            default:
                version_=GLES_3_3;
                break;
        }
    }
#else
    if (maj > 4) version_ = GL_4_6;
    else {
        if (maj == 3) {
            switch (minor) {
                case 0:
                    version_ = GL_3_0;
                    break;
                case 1:
                    version_ = GL_3_1;
                    break;
                case 2:
                    // NOTE (mw) intentional fallthrough
                default:
                    version_ = GL_3_2;
                    break;
            }
        } else if (maj == 4) {
            switch (minor) {
                case 0:
                    version_ = GL_4_0;
                    break;
                case 1:
                    version_ = GL_4_1;
                    break;
                case 2:
                    version_ = GL_4_2;
                    break;
                case 3:
                    version_ = GL_4_3;
                    break;
                case 4:
                    version_ = GL_4_4;
                    break;
                case 5:
                    version_ = GL_4_5;
                    break;
                case 6:
                    // NOTE (mw) intentional fallthrough
                default:
                    version_ = GL_4_6;
                    break;
            }
        }
    }
#endif
}

/**
 * @brief Queries graphics chipset on the current system and stores results in GLInfo singleton
 */
void GLInfo::queryChipset() {
    const GLubyte * vendor = glGetString(GL_VENDOR);
    const GLubyte * renderer = glGetString(GL_RENDERER);
    type_ = GENERIC;
#ifdef ANDROID
    if (strcasestr((const char *)vendor,"NVIDIA") || strcasestr((const char *)renderer,"NVIDIA")) type_ =NVIDIA;
    if (strcasestr((const char *)vendor,"ARM") || strcasestr((const char *)renderer,"ARM")) type_ = ARM_MALI;
    if (strcasestr((const char *)vendor,"Qualcomm") || strcasestr((const char *)renderer,"Adreno")) type_ = QUALCOMM_ADRENO;
    if (strcasestr((const char *)vendor,"Imagination") || strcasestr((const char *)renderer,"PowerVR")) type_ = POWERVR;
#else
    if (string_match((const char *)vendor,"NVIDIA") || string_match((const char *)renderer,"NVIDIA")) type_ = NVIDIA;
    if (string_match((const char *)vendor,"AMD") || string_match((const char *)renderer,"AMD")) type_ = AMD;
    if (string_match((const char *)vendor,"ATI") || string_match((const char *)renderer,"ATI")) type_ = AMD;
    if (string_match((const char *)vendor,"Intel") || string_match((const char *)renderer,"Intel")) type_ = INTEL;
#endif
    renderer_ = std::string((const char *)renderer);
}


/**
 * @brief Queries GL shader capabilities of the system and stores results in GLInfo singleton
 */
void GLInfo::queryShader() {
    const GLubyte *shader = glGetString(GL_SHADING_LANGUAGE_VERSION);
    // somewhat ugly code but does the job
#if defined(FYUSENET_USE_EGL) || defined(FYUSENET_USE_WEBGL)
    shaderVersion_ = GLSL_100_ES;
#else
    shaderVersion_ = GLSL_100;
#endif
    if (shader) {
        const char *version = scanShaderVersion((const char *)shader);
        if (version) {
#if defined(FYUSENET_USE_EGL) || defined(FYUSENET_USE_WEBGL)
            switch (version[0]) {
                case '1':
                    if (version[2] == '0') shaderVersion_ = GLSL_100_ES;
                    else shaderVersion_ = GLSL_110_ES;
                    break;
                case '2':
                    shaderVersion_ = GLSL_200_ES;
                    break;
                case '3':
                    if (version[2] == '0') shaderVersion_ = GLSL_300_ES;
                    else if (version[2] == '1') shaderVersion_ = GLSL_310_ES;
                    else if (version[2] == '2') shaderVersion_ = GLSL_320_ES;
                    else if (version[2] == '3') shaderVersion_ = GLSL_330_ES;
                    else shaderVersion_ = GLSL_330_ES;
                    break;
                default:
                    shaderVersion_ = GLSL_330_ES;
            }
#else
            switch (version[0]) {
                case '1':
                    if (version[2] == '0') shaderVersion_ = GLSL_100;
                    else if (version[2] == '2') shaderVersion_ = GLSL_120;
                    else if (version[2] == '3') shaderVersion_ = GLSL_130;
                    else if (version[2] == '4') shaderVersion_ = GLSL_140;
                    else if (version[2] == '5') shaderVersion_ = GLSL_150;
                    else shaderVersion_=GLSL_150;
                    break;
                case '3':
                    if (version[2] == '0') shaderVersion_ = GLSL_300;
                    else if (version[2] == '1') shaderVersion_ = GLSL_310;
                    else if (version[2] == '2') shaderVersion_ = GLSL_320;
                    else if (version[2] == '3') shaderVersion_ = GLSL_330;
                    else shaderVersion_=GLSL_330;
                    break;
                case '4':
                    if (version[2] == '0') shaderVersion_ = GLSL_400;
                    else if (version[2] == '1') shaderVersion_ = GLSL_410;
                    else if (version[2] == '2') shaderVersion_ = GLSL_420;
                    else if (version[2] == '3') shaderVersion_ = GLSL_430;
                    else if (version[2] == '4') shaderVersion_ = GLSL_440;
                    else if (version[2] == '5') shaderVersion_ = GLSL_450;
                    else shaderVersion_ = GLSL_460;
                    break;
                default:
                    shaderVersion_ = GLSL_450;
            }
#endif
        }
    }
}


/**
 * @brief Scan string for GLSL shader version
 *
 * @param string String to scan
 *
 * @return Pointer inside the supplied \p string that features the shader version string (major.minor)
 */
const char * GLInfo::scanShaderVersion(const char *string) {
    // this could be done way more elegant
    const char *ptr = string;
    while (ptr[0] != 0) {
        if ((ptr[0] == '.') && (ptr > string)) {
            if ((ptr[-1] >= '0') && (ptr[-1] <= '9')) {
                ptr--;
                while (ptr > string) {
                    if ((ptr[-1] >= '0') && (ptr[-1] <= '9')) ptr--;
                    else return ptr;
                }
                return ptr;
            } else ptr++;
        } else ptr++;
    }
    return nullptr;
}


/**
 * @brief Retrieve maximum number of allowed UBOs for vertex shaders
 *
 * @return Max number of uniform blocks for a vertex shader
 */
int GLInfo::getMaxVertexUBOs() {
    int ver = GLInfo::getVersion();
    if ((ver >= GLES_3_0) || ((ver < GLES_2_0) && (ver >= GL_3_1))) {
        GLint data=0;
        glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS,&data);
        return data;
    } else return 0;
}


/**
 * @brief Retrieve maximum number of allowed UBOs for fragment shaders
 *
 * @return Max number of uniform blocks for a fragment shader
 */
int GLInfo::getMaxFragmentUBOs() {
    int ver = GLInfo::getVersion();
    if ((ver >= GLES_3_0) || ((ver < GLES_2_0) && (ver >= GL_3_1))) {
        GLint data=0;
        glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS,&data);
        return data;
    } else return 0;
}

/**
 * @brief Retrieve maximum number of uniform vectors for a given shader type
 *
 * @param type Shader type to query
 *
 * @return Max number of uniform vectors (vec4) for the provided \p type
 */
int GLInfo::getMaxUniformVectors(shadertype type) {
    GLint data=0;
    int ver = GLInfo::getVersion();
    if ((ver >= GLES_2_0) || ((ver < GLES_2_0) && (ver >= GL_3_0))) {
        switch (type) {
            case GEOMETRY:
                return 0;
            case VERTEX:
                glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS,&data);
                return data;
            case FRAGMENT:
                glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS,&data);
                return data;
            default:
                // TODO (mw) support other shader types here
                return 0;
        }
    }
    return 0;
}


/**
 * @brief
 *
 * @return
 */
unsigned int GLInfo::getMaxUBOSize() {
    int ver = GLInfo::getVersion();
    if ((ver >= GLES_3_0) || ((ver < GLES_2_0) && (ver >= GL_3_1))) {
        GLint data=0;
        glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE,&data);
        return (unsigned int)data;
    } else return 0;
}


/**
 * @brief Retrieve maximum number of drawing buffers for multiple render targets
 *
 * @return Max number of drawing buffers that can be used for MRT operation
 */
int GLInfo::getMaximumDrawBuffers() {
    GLint drawb=0, colbuffer=0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS ,&drawb);
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &colbuffer);
    if (colbuffer < drawb) return colbuffer;
    else return drawb;
}


/**
 * @brief Retrieve recommended maximum number of drawing buffers for multiple render targets
 *
 * @return Max. recommended number of drawing buffers to use for MRT operation
 *
 * On some systems it is not a good choice to use the full number of drawing-buffers for
 * multiple render targets. This function returns the recommended maximum number, based on GPU
 * specifics.
 */
int GLInfo::getMaximumRecommendedDrawBuffers() {
    if (instance_.recommendedDrawBuffers_ != -1) return instance_.recommendedDrawBuffers_;
    int maxrt = getMaximumDrawBuffers();
    if (getGPUType() == GLInfo::ARM_MALI) {
        std::string rend = getRendererString();
        if (strstr(rend.c_str(),"-T")) {
            maxrt = std::min((int)MAX_MALI_T_SERIES_RENDER_TARGETS,maxrt);
        }
    }
    maxrt = std::min((int)FBO::MAX_DRAWBUFFERS,maxrt);
    instance_.recommendedDrawBuffers_ = maxrt;
    return maxrt;
}

/**
 * @brief Get maximum size (along any dimension) of textures on this system
 *
 * @return Maximum texture size/extent
 */
int GLInfo::getMaximumTextureSize() {
    int res = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &res);
    return res;
}

/**
 * @brief Get maximum depth for a 2D texture array on this system
 *
 * @return Maximum depth for a 2D texture array
 */
int GLInfo::getMaximumTexArrayDepth() {
    int res = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &res);
    return res;
}

/**
 * @brief Get maximum number of varying vectors that can be passed from vertex to fragment shader
 *
 * @return Max. number of varying vectors (vec4)
 */
int GLInfo::getMaxVaryingVectors() {
    int res = 0;
    glGetIntegerv(GL_MAX_VARYING_VECTORS, &res);
    return res;
}

/**
 * @brief Get maximum number of texture units supported by the system
 *
 * @return Maximum number of texture units
 */
int GLInfo::getMaximumTextureUnits() {
    int res = 0;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &res);
    return res;
}


/**
 * @brief Log info about GL system to logging facility
 */
void GLInfo::printInfo() {
    // ugly
    GLint data = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &data);
    int maj=0, minor=0;
    glGetIntegerv(GL_MAJOR_VERSION, &maj);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    FNLOGI("GL version: %d.%d",maj,minor);
    const char *shader = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
    const char * glsl = scanShaderVersion(shader);
    if (glsl) FNLOGI("GLSL version: %s",glsl);
    const GLubyte * vendor = glGetString(GL_VENDOR);
    const GLubyte * renderer = glGetString(GL_RENDERER);
    FNLOGI("GPU vendor: %s",vendor);
    FNLOGI("GPU renderer: %s",renderer);
    FNLOGI("Caps:")
    FNLOGI("  GL_MAX_TEXTURE_SIZE: %d",data);
    data=0;glGetIntegerv(GL_MAX_VERTEX_ATTRIBS,&data);
    FNLOGI("  GL_MAX_VERTEX_ATTRIBS: %d",data);
    data=0;glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS,&data);
    FNLOGI("  GL_MAX_VERTEX_UNIFORM_VECTORS: %d",data);
    data=0;glGetIntegerv(GL_MAX_VARYING_VECTORS,&data);
    FNLOGI("  GL_MAX_VARYING_VECTORS: %d",data);
    data=0;glGetIntegerv(GL_MAX_VERTEX_OUTPUT_COMPONENTS,&data);
    FNLOGI("  GL_MAX_VERTEX_OUTPUT_COMPONENTS: %d",data);
    data=0;glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,&data);
    FNLOGI("  GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: %d",data);
    data=0;glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,&data);
    FNLOGI("  GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: %d",data);
    data=0;glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,&data);
    FNLOGI("  GL_MAX_TEXTURE_IMAGE_UNITS: %d",data);
    data=0;glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS,&data);
    FNLOGI("  GL_MAX_FRAGMENT_UNIFORM_VECTORS: %d",data);
    data=0;glGetIntegerv(GL_MAX_FRAGMENT_INPUT_COMPONENTS,&data);
    FNLOGI("  GL_MAX_FRAGMENT_INPUT_COMPONENTS: %d",data);
    data=0;glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS,&data);
    FNLOGI("  GL_MAX_COLOR_ATTACHMENTS: %d",data);
    data=0;glGetIntegerv(GL_MAX_DRAW_BUFFERS,&data);
    FNLOGI("  GL_MAX_DRAW_BUFFERS: %d",data);
    data=0;glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,&data);
    FNLOGI("  GL_MAX_FRAGMENT_UNIFORM_COMPONENTS: %d",data);
    data=0;glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS,&data);
    FNLOGI("  GL_MAX_VERTEX_UNIFORM_COMPONENTS: %d",data);
    int range[2],precision=0;
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_LOW_INT,range,&precision);
    FNLOGI("  (I) GL_FRAGMENT_LOW: [%d %d]",range[0],range[1]);
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_MEDIUM_INT,range,&precision);
    FNLOGI("  (I) GL_FRAGMENT_MEDIUM: [%d %d]",range[0],range[1]);
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_HIGH_INT,range,&precision);
    FNLOGI("  (I) GL_FRAGMENT_HIGH: [%d %d]",range[0],range[1]);
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_LOW_FLOAT,range,&precision);
    FNLOGI("  (F) GL_FRAGMENT_LOW: [%d %d] %d",range[0],range[1],precision);
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_MEDIUM_FLOAT,range,&precision);
    FNLOGI("  (F) GL_FRAGMENT_MEDIUM: [%d %d] %d",range[0],range[1],precision);
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER,GL_HIGH_FLOAT,range,&precision);
    FNLOGI("  (F) GL_FRAGMENT_HIGH: [%d %d] %d",range[0],range[1],precision);
    if ((GLInfo::getVersion() >= GLES_3_0) ||
        (GLInfo::getVersion() >= WEBGL_1_0) ||
        ((GLInfo::getVersion() < GLES_2_0)&&(GLInfo::getVersion() >= GL_3_1))) {
        glGetError();
        data=0;glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_UNIFORM_BUFFER_BINDINGS: %d",data);
        data=0;glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_VERTEX_UNIFORM_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_FRAGMENT_UNIFORM_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_COMBINED_UNIFORM_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_UNIFORM_BLOCK_SIZE: %d",data);
        data=0;glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT: %d",data);
        data=0;glGetIntegerv(GL_MIN_PROGRAM_TEXEL_OFFSET,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MIN_PROGRAM_TEXEL_OFFSET: %d",data);
        data=0;glGetIntegerv(GL_MAX_PROGRAM_TEXEL_OFFSET,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_PROGRAM_TEXEL_OFFSET: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_COMBINED_VERTEX_UNIFORM_COPONENTS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS,&data);if (glGetError()!=GL_NO_ERROR) data=0;
        FNLOGI("  GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS: %d",data);
    }
#if defined(GL_VERSION_4_3) || defined(GL_ES_VERSION_3_1) || defined(GL_ES_VERSION_3_2)
    if (supportsComputeShader()) {
        GLint data2=0,data3=0;
        data=0;
        glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS,&data);
        FNLOGI("  GL_MAX_COMPUTE_IMAGE_UNIFORMS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&data);
        FNLOGI("  GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,&data);
        FNLOGI("  GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_COUNT,&data);
        FNLOGI("  GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS,&data);
        FNLOGI("  GL_MAX_COMPUTE_UNIFORM_BLOCKS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_COMPONENTS,&data);
        FNLOGI("  GL_MAX_COMPUTE_UNIFORM_COMPONENTS: %d",data);
        data=0;glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&data);
        FNLOGI("  GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: %d",data);
        data=0;data2=0,data3=0;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&data);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,1,&data2);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,2,&data3);
        FNLOGI("  GL_MAX_COMPUTE_WORK_GROUP_SIZE: %d %d %d",data,data2,data3);
        data=0;data2=0,data3=0;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&data);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,1,&data2);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,2,&data3);
        FNLOGI("  GL_MAX_COMPUTE_WORK_GROUP_COUNT: %d %d %d",data,data2,data3);
    } else {
        FNLOGI("  (NO COMPUTE SHADER SUPPORT)");
    }
#endif
    FNLOGI("Extensions:");
    char buf[256];
    const char *eptr = instance_.extensions_.c_str();
    while (eptr[0]!=0) {
        const char *nextptr = strpbrk(eptr," ");
        if (!nextptr) break;
        else {
            int len = nextptr-eptr;
            if (len < (int)(sizeof(buf)-1)) {
                strncpy(buf,eptr,len);
                buf[len]=0;
            }
            FNLOGI("  - %s",buf);
            eptr=nextptr+1;
        }
    }
}


} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
