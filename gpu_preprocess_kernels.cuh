#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

#define MAX_CHANNELS 4          // Maximum number of channels supported
#define MAX_TILE_SIZE 64        // Maximum tile size for shared memory

// Constant memory for normalization parameters
// Declare constant memory variables as extern
extern __constant__ float mean_[MAX_CHANNELS];
extern __constant__ float std_[MAX_CHANNELS];

// Kernel declarations (e.g., resizeAndNormalizeKernel) remain unchanged
__global__ void resizeAndNormalizeKernel(
    const uint8_t* src,
    float* dst,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int channels);