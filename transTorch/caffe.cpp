#include <string>
#include <vector>

#include <TH/TH.h>
#include "caffe/caffe.hpp"

extern "C"
{
void* loadCaffeNet(const char* param_file, const char* model_file, const char* phase);
void releaseCaffeNet(void* net_);
}

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

  Net<float>* net = new Net<float>(string(param_file), phase);
  if(model_file != NULL)
    net->CopyTrainedLayersFrom(string(model_file));

  return net;
}

void releaseCaffeNet(void* net_) {
    Net<float>* net = (Net<float>*)net_;

    if ( net != NULL) {
        delete net;
    }
}

void writeCaffeLinearLayer(void* net, const char* layerName, THFloatTensor* weights, THFloatTensor* bias) {

}

