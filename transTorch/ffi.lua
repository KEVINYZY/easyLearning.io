local ffi = require 'ffi'

ffi.cdef[[
void loadCaffe(void* handle[1], const char* param_file, const char* model_file, const char* phase_name);
]]

transTorch._C = ffi.load(package.searchpath('libtrans_torch', package.cpath))
