#!/bin/bash

function aten {
    rm -rf aten/build 
    mkdir -p aten/build
    mkdir -p install
    cd aten/build
    cmake .. -DNO_CUDA=TRUE -DCMAKE_INSTALL_PREFIX=../../install
}

function rpclib {
    rm -rf rpclib/build
    mkdir -p rpclib/build
    cd rpclib/build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../install
    make install
}

function auto {
    python auto/gen_all.py ./aten/build/src/ATen/ATen/Declarations.yaml ./express/generated
}



$1
