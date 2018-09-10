#!/bin/bash

function aten {
    rm -rf aten/build install
    mkdir -p aten/build
    mkdir -p install
    cd aten/build
    cmake .. -DNO_CUDA=TRUE -DCMAKE_INSTALL_PREFIX=../../install
}

function auto {
    python auto/gen_all.py ./aten/build/src/ATen/ATen/Declarations.yaml ./express/generated
}

$1
