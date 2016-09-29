#include <string>
#include <vector>

#include <TH/TH.h>
#include "caffe/caffe.hpp"

extern "C"
{
void loadCaffe(void* handle[1], const char* param_file, const char* model_file, const char* phase);
}

using namespace caffe;  // NOLINT(build/namespaces)

void loadCaffe(void** handle, const char* param_file, const char* model_file, const char* phase_name) {
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
    phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
    phase = TEST;
  } else {
    THError("Unknown phase.");
  }
  Net<float>* net_ = new Net<float>(string(param_file), phase);
  if(model_file != NULL)
    net_->CopyTrainedLayersFrom(string(model_file));

  *handle = net_;
}

