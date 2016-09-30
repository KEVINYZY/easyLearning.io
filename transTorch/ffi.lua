local ffi = require 'ffi'

ffi.cdef[[
void* loadCaffeNet(const char* param_file, const char* model_file, const char* phase_name);
void releaseCaffeNet(void* handle);
void writeCaffeLinearLayer(void* net, const char* layerName, THFloatTensor* weights, THFloatTensor* bias); 
]]

transTorch._C = ffi.load(package.searchpath('libtrans_torch', package.cpath))
