local ffi = require 'ffi'

ffi.cdef[[
void* loadCaffeNet(const char* param_file, const char* model_file);
void releaseCaffeNet(void* net);
void saveCaffeNet(void* net_, const char* weight_file);

void writeCaffeConvLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias); 
void writeCaffeLinearLayer(void* net, const char* layername, THFloatTensor* weights, THFloatTensor* bias); 
void writeCaffeBNlayer(void* net, const char* layername, 
                       THFloatTensor* weights, THFloatTensor* bias, 
                       THFloatTensor* mean, THFloatTensor* var);

]]

transTorch._C = ffi.load(package.searchpath('libtrans_torch', package.cpath))


