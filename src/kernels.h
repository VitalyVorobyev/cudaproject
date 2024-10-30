#pragma once

#include <npp.h>

// Kernel to find local extrema in each column
__global__ void findLocalExtrema(const Npp32f* src, Npp8u* mask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int idx = width + x;
    for (int y = 1; y < height - 1; ++y) {
        mask[idx] = 255 * (
            (src[idx] > src[idx - width] && src[idx] > src[idx + width]) || 
            (src[idx] < src[idx - width] && src[idx] < src[idx + width]));
        idx += width;
    }
}

__global__ void findLocalExtremaEachPixel(const Npp32f* src, Npp8u* mask, int width, int height, int srcstep, int maskstep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y < 1 || y >= height - 1) return;
    int idx = y * srcstep + x;
    int maskidx = y * maskstep + x;
    mask[maskidx] = 255 * (
        (src[idx] > src[idx - srcstep] && src[idx] > src[idx + srcstep]) || 
        (src[idx] < src[idx - srcstep] && src[idx] < src[idx + srcstep]));
}
