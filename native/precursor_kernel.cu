// Batched theoretical precursor m/z for proteomics.
// One CUDA thread owns one peptide: sum residue masses, then [M+zH]z+.
//
// Build on a machine with the CUDA toolkit and an NVIDIA GPU:
//   ./native/build.sh
//
// Residue masses and the proton mass are passed in from pyOpenMS. The kernel
// does not invent a second mass table.

#include <cuda_runtime.h>

__constant__ double c_residue[27];

__global__ void precursor_kernel(
    const int* __restrict__ sequences,
    const int* __restrict__ lengths,
    int n,
    int width,
    double water,
    double proton,
    const int* __restrict__ charges,
    int n_charges,
    double* __restrict__ mono,
    double* __restrict__ mz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int* row = sequences + static_cast<long long>(i) * width;
    int length = lengths[i];
    double mass = water;
    for (int j = 0; j < length; ++j) {
        int aa = row[j];
        if (aa > 0 && aa < 27) {
            mass += c_residue[aa];
        }
    }
    mono[i] = mass;
    for (int c = 0; c < n_charges; ++c) {
        double z = static_cast<double>(charges[c]);
        mz[static_cast<long long>(i) * n_charges + c] = (mass + z * proton) / z;
    }
}

extern "C" int pepmz_device_count() {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n < 0) {
        return 0;
    }
    return n;
}

extern "C" int pepmz_precursor_launch(
    const int* h_seq,
    const int* h_len,
    int n,
    int width,
    const double* h_mass27,
    double water,
    double proton,
    const int* h_charges,
    int n_charges,
    double* h_mono,
    double* h_mz) {
    if (n <= 0 || width <= 0 || n_charges <= 0 || h_mass27 == nullptr) {
        return 1;
    }

    cudaError_t err = cudaMemcpyToSymbol(c_residue, h_mass27, 27 * sizeof(double));
    if (err != cudaSuccess) {
        return 2;
    }

    int* d_seq = nullptr;
    int* d_len = nullptr;
    int* d_charges = nullptr;
    double* d_mono = nullptr;
    double* d_mz = nullptr;

    size_t seq_n = static_cast<size_t>(n) * static_cast<size_t>(width);
    size_t mz_n = static_cast<size_t>(n) * static_cast<size_t>(n_charges);

    err = cudaMalloc(&d_seq, seq_n * sizeof(int));
    if (err != cudaSuccess) {
        return 3;
    }
    err = cudaMalloc(&d_len, static_cast<size_t>(n) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_seq);
        return 3;
    }
    err = cudaMalloc(&d_charges, static_cast<size_t>(n_charges) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_seq);
        cudaFree(d_len);
        return 3;
    }
    err = cudaMalloc(&d_mono, static_cast<size_t>(n) * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_seq);
        cudaFree(d_len);
        cudaFree(d_charges);
        return 3;
    }
    err = cudaMalloc(&d_mz, mz_n * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_seq);
        cudaFree(d_len);
        cudaFree(d_charges);
        cudaFree(d_mono);
        return 3;
    }

    cudaMemcpy(d_seq, h_seq, seq_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charges, h_charges, static_cast<size_t>(n_charges) * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    precursor_kernel<<<blocks, threads>>>(
        d_seq, d_len, n, width, water, proton, d_charges, n_charges, d_mono, d_mz);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(h_mono, d_mono, static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mz, d_mz, mz_n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_seq);
    cudaFree(d_len);
    cudaFree(d_charges);
    cudaFree(d_mono);
    cudaFree(d_mz);
    return err == cudaSuccess ? 0 : 4;
}
