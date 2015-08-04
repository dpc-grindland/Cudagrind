#!/bin/bash

# Run script for Cudagrind. Please set approriate values for both
#  LD_LIBRARY_PATH and LD_PRELOAD on your system. Defaults are 
#  meant to be used on the HLRS Laki cluster.

export LD_LIBRARY_PATH=/usr/lib64/valgrind:$LD_LIBRARY_PATH
export LD_PRELOAD=`pwd`/../libcudaWrap.so:/opt/cuda/driver-5.0/lib64/libcuda.so:$LD_PRELOAD

# Start program given in @1 if provided to script, else run ./main
valgrind --suppressions=../cuda.supp --gen-suppressions=all "$@"
