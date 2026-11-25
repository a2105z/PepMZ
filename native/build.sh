#!/bin/sh
# Compile the precursor kernel into native/libpepmz_cuda.so.
# Requires nvcc from the CUDA toolkit and an NVIDIA driver.
set -eu
cd "$(dirname "$0")"
nvcc -O3 -shared -Xcompiler -fPIC -o libpepmz_cuda.so precursor_kernel.cu
echo "built $(pwd)/libpepmz_cuda.so"
