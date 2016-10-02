#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include <TH/TH.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

extern "C"
{
void* loadCaffeNet(const char* param_file, const char* model_file, const char* phase);
void releaseCaffeNet(void* net_);
void saveCaffeNet(void* net_, const char* weight_file);

void writecaffeconvlayer(void* net, const char* layername, thfloattensor* weights, thfloattensor* bias);
void writecaffelinearlayer(void* net, const char* layername, thfloattensor* weights, thfloattensor* bias);
void writecaffebnlayer(void* net, const char* layername,
                       thfloattensor* weights, thfloattensor* bias,
                       thfloattensor* mean, thfloattensor* var);
}

typedef float Dtype;

using namespace caffe;  // NOLINT(build/namespaces)

void* loadCaffeNet(const char* param_file, const char* model_file, const char* phase_name) {
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
    phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
    phase = TEST;
  } else {
    THError("Unknown phase.");
  }

  Net<Dtype>* net = new Net<Dtype>(string(param_file), phase);
  if(model_file != NULL)
    net->CopyTrainedLayersFrom(string(model_file));

  return net;
}

void releaseCaffeNet(void* net_) {
    Net<Dtype>* net = (Net<Dtype>*)net_;
    if ( net != NULL) {
        delete net;
    }
}

void saveCaffeNet(void* net_, const char* weight_file) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    NetParameter net_param;
    net->ToProto(&net_param);

    WriteProtoToBinaryFile(net_param, std::string(weight_file));
}

void writeCaffeBNLayer(void* net_, const char* layerName,
                       THFloatTensor* weights, THFloatTensor* bias,
                       THFloatTensor* mean, THFloatTensor* var) {
     Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();


}

void writeCaffeConvLayer(void* net, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking output layer is conv, so parameter's blob size is 2
    if ( blobs.size() != 2) {
        std::ostringstream oss;
        oss << "Can't write into layer :" << layerName ;
        THError(oss.str().c_str());
    }

    // Checking size
    unsigned int th_weights_size = weights->size[0] * weights->size[1] * weights->size[2] * weights->size[3];
    CHECK_EQ(th_weights_size, blobs[0]->count());

    unsigned int th_bias_size = bias->size[0] * bias->size[1] * bias->size[2] * bias->size[3];
    CHECK_EQ(th_bias_size, blobs[1]->count());

    // Copying data
    const float* data_ptr = THFloatTensor_data(weights);
    caffe_copy(blobs[0]->count(), data_ptr, blobs[0]->mutable_cpu_data());

    data_ptr = THFloatTensor_data(bias);
    caffe_copy(blobs[1]->count(), data_ptr, blobs[1]->mutable_cpu_data());
}

void writeCaffeLinearLayer(void* net_, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {
    Net<Dtype>* net = (Net<Dtype>*)net_;

    const boost::shared_ptr<caffe::Layer<Dtype> > inLayer = net->layer_by_name(std::string(layerName));
    vector<shared_ptr<Blob<Dtype> > > blobs = inLayer->blobs();

    // Checking output layer is conv, so parameter's blob size is 2
    if ( blobs.size() != 2) {
        std::ostringstream oss;
        oss << "Can't write into layer :" << layerName ;
        THError(oss.str().c_str());
    }

    // Checking size
    unsigned int th_weights_size = weights->size[0] * weights->size[1] * weights->size[2] * weights->size[3];
    CHECK_EQ(th_weights_size, blobs[0]->count());

    unsigned int th_bias_size = bias->size[0] * bias->size[1] * bias->size[2] * bias->size[3];
    CHECK_EQ(th_bias_size, blobs[1]->count());

    // Copying data
    const float* data_ptr = THFloatTensor_data(weights);
    caffe_copy(blobs[0]->count(), data_ptr, blobs[0]->mutable_cpu_data());

    data_ptr = THFloatTensor_data(bias);
    caffe_copy(blobs[1]->count(), data_ptr, blobs[1]->mutable_cpu_data());
}

