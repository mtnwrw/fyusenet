//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network w/ 3x3 Convolutions
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "stylenet3x3.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------



/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct style-transfer network
 *
 * @param width Width of image data to process
 * @param height Height of image data to process
 * @param upload Create an upload layer in the network (will be named "upload")
 * @param download Create a download layer in the network (will be named "download")
 * @param ctx Link to GL context that the network should use
 */
StyleNet3x3::StyleNet3x3(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx) :
    StyleNetBase(width, height, upload, download, ctx) {
    for (int i=0; i < ASYNC_BUFFERS; i++) inBuffers_[i] = nullptr;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc NeuralNetwork::buildLayers
 */
fyusion::fyusenet::CompiledLayers StyleNet3x3::buildLayers() {
    using namespace fyusion::fyusenet;
    std::shared_ptr<LayerFactory> factory = getLayerFactory();
    if (upload_) {
        auto * up = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
        up->shape(3, height_, width_, 3).context(context()).number(layer_ids::UPLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) up->async().callback(std::bind(&StyleNet3x3::internalULCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        up->push(factory);
    } else if (oesInput_) {
#ifdef FYUSENET_USE_EGL
        gpu::GPULayerBuilder * oes = new gpu::GPULayerBuilder("oes");
        oes->shape(3,height_,width_,3).type(LayerType::OESCONV).context(context()).number(layer_ids::UNPACK);
        oes->push(factory);
#else
        assert(false);
#endif
    }

    auto * conv1 = new gpu::ConvLayerBuilder(3,"conv1");
    conv1->shape(12,height_,width_,3).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(layer_ids::CONV1);
    conv1->push(factory);

    auto * conv2 = new gpu::ConvLayerBuilder(3,"conv2");
    conv2->shape(20,height_,width_,12).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(layer_ids::CONV2);
    conv2->push(factory);

    auto * conv3 = new gpu::ConvLayerBuilder(3,"conv3");
    conv3->shape(40,height_/2,width_/2,20).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(layer_ids::CONV3);
    conv3->push(factory);

    auto * res11 = new gpu::ConvLayerBuilder(3,"res1_1");
    res11->shape(40,height_/4,width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(layer_ids::RES1_1);
    res11->push(factory);

    auto * res12 = new gpu::ConvLayerBuilder(3,"res1_2");
    res12->shape(40,height_/4,width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual(ActType::RELU).context(context()).number(layer_ids::RES1_2);
    res12->push(factory);

    auto * res21 = new gpu::ConvLayerBuilder(3,"res2_1");
    res21->shape(40,height_/4,width_/4,40).type(LayerType::CONVOLUTION2D).context(context()).number(layer_ids::RES2_1);
    res21->push(factory);

    auto * res22 = new gpu::ConvLayerBuilder(3,"res2_2");
    res22->shape(40,height_/4,width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(layer_ids::RES2_2);
    res22->push(factory);

    auto * deconv1 = new gpu::ConvLayerBuilder(3,"deconv1");
    deconv1->shape(20,height_/4,width_/4,40).type(LayerType::FRACCONVOLUTION2D).downsample(2).sourceStep(0.5f).context(context()).number(layer_ids::DECONV1);
    deconv1->push(factory);

    auto * deconv2 = new gpu::ConvLayerBuilder(3,"deconv2");
    deconv2->shape(12,height_/4,width_/4,20).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.25f).downsample(2).prefixAct(ActType::RELU).context(context()).number(layer_ids::DECONV2);
    deconv2->push(factory);

    auto * deconv3 = new gpu::ConvLayerBuilder(3,"deconv3");
    deconv3->shape(3,height_/2,width_/2,12).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.5f).prefixAct(ActType::RELU).context(context()).number(layer_ids::DECONV3);
    deconv3->push(factory);

    auto * sigmoid = new gpu::GPULayerBuilder("sigmoid");
    sigmoid->shape(3,height_,width_,3).type(LayerType::SIGMOID).context(context()).number(layer_ids::SIGMOID);
    sigmoid->push(factory);

    if (download_) {
        auto * down = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
        down->shape(3, height_, width_, 3).context(context()).number(layer_ids::DOWNLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) down->async().callback(std::bind(&StyleNet3x3::internalDLCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        down->push(factory);
    }

    return factory->compileLayers();
}


/**
 * @copydoc NeuralNetwork::connectLayers
 */
void StyleNet3x3::connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) {
    using namespace fyusion::fyusenet;
    if (oesInput_) buffers->connectLayers(layers[layer_ids::UNPACK], layers[layer_ids::CONV1], 0);
    else if (upload_) buffers->connectLayers(layers[layer_ids::UPLOAD], layers[layer_ids::CONV1], 0);
    buffers->connectLayers(layers[layer_ids::CONV1],   layers[layer_ids::CONV2], 0);
    buffers->connectLayers(layers[layer_ids::CONV2],   layers[layer_ids::CONV3], 0);
    buffers->connectLayers(layers[layer_ids::CONV3],   layers[layer_ids::RES1_1], 0);
    buffers->connectLayers(layers[layer_ids::CONV3],   layers[layer_ids::RES1_2], 1);
    buffers->connectLayers(layers[layer_ids::RES1_1],  layers[layer_ids::RES1_2], 0);
    buffers->connectLayers(layers[layer_ids::RES1_2],  layers[layer_ids::RES2_1], 0);
    buffers->connectLayers(layers[layer_ids::RES1_2],  layers[layer_ids::RES2_2], 1);
    buffers->connectLayers(layers[layer_ids::RES2_1],  layers[layer_ids::RES2_2], 0);
    buffers->connectLayers(layers[layer_ids::RES2_2],  layers[layer_ids::DECONV1], 0);
    buffers->connectLayers(layers[layer_ids::DECONV1], layers[layer_ids::DECONV2], 0);
    buffers->connectLayers(layers[layer_ids::DECONV2], layers[layer_ids::DECONV3], 0);
    buffers->connectLayers(layers[layer_ids::DECONV3], layers[layer_ids::SIGMOID], 0);
    if (download_) {
        buffers->connectLayers(layers[layer_ids::SIGMOID], layers[layer_ids::DOWNLOAD], 0);
#ifdef FYUSENET_MULTITHREADING
        if (async_) {
            auto * down = (gpu::DownloadLayer *)layers[layer_ids::DOWNLOAD];
            assert(down);
            std::vector<BufferSpec> specs = down->getRequiredOutputBuffers();
            assert(specs.size() == 1);
            BufferShape shape(specs[0].height_, specs[0].width_, specs[0].channels_, 0,
                                 BufferShape::type::FLOAT32);
            asyncDLBuffers_[0] = shape.createCPUBuffer();
            asyncDLBuffers_[1] = shape.createCPUBuffer();
            down->addCPUOutputBuffer(asyncDLBuffers_[0]);
            down->addOutputConnection(0, nullptr, 0);
        } else {
            buffers->createCPUOutput(layers[layer_ids::DOWNLOAD], true);
        }
#else
        buffers->createCPUOutput(layers[layer_ids::DOWNLOAD], true);
#endif
    }
    else {
        buffers->createGPUOutput((gpu::GPULayerBase *)layers[layer_ids::SIGMOID]);
    }
}

