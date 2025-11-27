#!/bin/sh
echo "Change directory to MNN_SOURCE_ROOT/project/ios before running this script"
echo "Current PWD: ${PWD}"

rm -rf MNN-visionOS-CPU-GPU
mkdir MNN-visionOS-CPU-GPU
cd MNN-visionOS-CPU-GPU
# Static Begin
mkdir Static 
cd Static

rm -rf visionos_64
mkdir visionos_64
cd visionos_64
cmake ../../../ -DCMAKE_BUILD_TYPE=Release -DMNN_METAL=ON -DARCHS="arm64" -DENABLE_BITCODE=0 -DMNN_AAPL_FMWK=1 -DMNN_SEP_BUILD=0 -DMNN_ARM82=true -DMNN_BUILD_SHARED_LIBS=false -DMNN_USE_THREAD_POOL=OFF -DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_SYSTEM_PROCESSOR=arm64 -DMNN_AAPL_FMWK=ON -DMNN_SUPPORT_RENDER=true -DMNN_ARM82=true -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true $*
echo "Building AArch64"
make MNN -j16
echo "End Building AArch64"
cd ../

mv visionos_64/MNN.framework MNN.framework

rm -rf visionos_64
