//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network Example Main
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdio>
#include <iostream>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/stylenet3x3.h"
#include "../samplenetworks/stylenet9x9.h"
#include "../helpers/jpegio.h"
#include "cxxopts.hpp"

#ifdef FYUSENET_USE_GLFW
#include <fyusenet/gl/glcontext.h>
#endif


//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


static float * readImage(const std::string& imageFile, int & width, int & height) {
    if (!JPEGIO::isJPEG(imageFile)) {
        std::cerr<<"File "<<imageFile<<" is not a JPEG file\n";
        return nullptr;
    }
    uint8_t * rgb = JPEGIO::loadRGBImage(imageFile, width, height);
    if (!rgb) {
        std::cerr<<"Cannot read "<<imageFile<<" make sure it is an RGB image\n";
        return nullptr;
    }
    float * image = new float[width * height * 3];
    for (int i=0; i < width*height*3; i++) {
        image[i] = ((float)rgb[i]) / 255.f;
    }
    delete [] rgb;
    return image;
}

static void writeImage(const float *rgba, int width, int height, const std::string& fileName) {
    assert(rgba);
    uint8_t *rgb = new uint8_t[width*height*3];
    // FIXME (mw) this code silently assumes that when we download 3-channel data it will always be extended to 4-channel data
    for (int y=0; y < height; y++) {
        for (int x=0; x < width; x++) {
            rgb[(y*width+x)*3] = (uint8_t)(rgba[(y*width+x)*4]*255.f);
            rgb[(y*width+x)*3+1] = (uint8_t)(rgba[(y*width+x)*4+1]*255.f);
            rgb[(y*width+x)*3+2] = (uint8_t)(rgba[(y*width+x)*4+2]*255.f);
        }
    }
    JPEGIO::saveRGBImage(rgb, width, height, fileName);
    delete [] rgb;
}


int main(int argc, char **argv) {
    cxxopts::Options options(argv[0],"Sample style-transfer network");
    options.add_options()("h,help","Get program help")
                         ("k,kernel", "Kernel size for the convolution layers, either 3 for 3x3 or 9 for 9x9", cxxopts::value<int>()->default_value("3"))
                         ("w,weights", "Use supplied filename as weight file (mandatory)", cxxopts::value<std::string>())
#ifdef DEBUG
                         ("l,log", "Log layer outputs to supplied directory", cxxopts::value<std::string>())
#endif
                         ("input", "Input JPEG file", cxxopts::value<std::string>())
                         ("output", "Output JPEG file", cxxopts::value<std::string>());
    options.parse_positional({"input","output"});
    auto opts = options.parse(argc, argv);

    if ((opts.count("help") > 0) || (opts.count("input") == 0) || (opts.count("output") == 0) || (opts.count("weights") == 0)) {
        std::cout<<options.help()<<std::endl;
        std::cout<<"\nNote that this sample only accepts JPEG images as input and output of which the dimensions\n(width and height) are a multiple of 4."<<std::endl;
        return 0;
    }

    // -------------------------------------------------------
    // Read JPEG image that is to be processed
    // -------------------------------------------------------
    int width, height;
    float * rgb = readImage(opts["input"].as<std::string>(), width, height);
    if (!rgb) return 1;
    if ((width % 4) || (height % 4)) {
        std::cerr<<"Input image must have dimensions that are a multiple of 4 (width and height)\n";
        return 1;
    }
    // -------------------------------------------------------
    // Setup GL context and thread/PBO pool. If we use GLFW,
    // set mouse-button callbacks and wait for an initial
    // MB press, followed by a couple of empty render calls
    // -------------------------------------------------------
    auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
    if (!glmgr) {
        std::cerr<<"Cannot setup GL context\n";
        delete [] rgb;
        return 1;
    }
    fyusion::fyusenet::GfxContextLink ctx = glmgr->createMainContext();
#ifdef FYUSENET_MULTITHREADING
    fyusion::opengl::AsyncPool::setMaxGLThreads(4);
#endif
    glmgr->setupPBOPools(2, 2);

    // NOTE (mw) this is ugly
#ifdef FYUSENET_USE_GLFW
    static bool buttonup = false;
    auto * glctx = dynamic_cast<const fyusion::opengl::GLContext *>(ctx.interface());
    auto mousecb = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) buttonup = true;
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecb);
    while (!buttonup) {
        glfwWaitEventsTimeout(0.1);
    }
    for (int i=0; i<4; i++) {
        glctx->sync();
    }
#endif
    // -------------------------------------------------------
    // Instantiate network
    // -------------------------------------------------------
    StyleNetBase * net = nullptr;
    StyleNetProvider * params = nullptr;
    switch (opts["kernel"].as<int>()) {
        case 3:
            net = new StyleNet3x3(width, height, true, true, ctx);
            params = new StyleNet3x3Provider(opts["weights"].as<std::string>());
            break;
        case 9:
            net = new StyleNet9x9(width, height, true, true, ctx);
            params = new StyleNet9x9Provider(opts["weights"].as<std::string>());
            break;
        default:
            std::cerr<<"Kernel size "<<opts["kernel"].as<int>()<<" not supported.\n";
    }
    if (!net) {
        delete [] rgb;
        return 1;
    }
    // -------------------------------------------------------
    // Load weights, setup and run network...
    // -------------------------------------------------------
    net->setParameters(params);
    net->setup();
#ifdef DEBUG
    if (opts.count("log")) ((StyleNetBase *)net)->enableDebugOutput(opts["log"].as<std::string>());
#endif
    net->setInputBuffer(rgb);
    auto state = net->forward();
    assert(state.status == fyusion::fyusenet::NeuralNetwork::state::EXEC_DONE);
    // -------------------------------------------------------
    // Save result
    // -------------------------------------------------------
    auto * downbuf = net->getOutputBuffer();
    if (downbuf) {        
        downbuf->with<float>([downbuf, opts](const float *ptr) {
            writeImage(ptr, downbuf->shape().width(), downbuf->shape().height(), opts["output"].as<std::string>());
        });
    }

    // -------------------------------------------------------
    // If we use GLFW, wait for another MB click before
    // terminating
    // -------------------------------------------------------
#ifdef FYUSENET_USE_GLFW
    static bool buttondown = false;
    glctx->sync();
    auto mousecbout = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) {
            buttondown = true;
        }
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecbout);
    while (!buttondown) {
        glfwWaitEventsTimeout(0.1);
    }
#endif

    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    net->cleanup();
    delete net;
    delete [] rgb;
    ctx.reset();
    glmgr->tearDown();

    return 0;
}


// vim: set expandtab ts=4 sw=4:

