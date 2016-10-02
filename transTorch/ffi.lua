local ffi = require 'ffi'

ffi.cdef[[
void* loadCaffeNet(const char* param_file, const char* model_file, const char* phase_name);
void releaseCaffeNet(void* net);
void saveCaffeNet(void* net_, const char* weight_file);

void writecaffeconvlayer(void* net, const char* layername, thfloattensor* weights, thfloattensor* bias); 
void writecaffelinearlayer(void* net, const char* layername, thfloattensor* weights, thfloattensor* bias); 
void writecaffebnlayer(void* net, const char* layername, 
                       thfloattensor* weights, thfloattensor* bias, 
                       thfloattensor* mean, thfloattensor* var);
 
                       
]]

transTorch._C = ffi.load(package.searchpath('libtrans_torch', package.cpath))


