/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "kernels.h"
#include "utils.h"

void find_extrema(const npp::ImageNPP_32f_C1& data, npp::ImageNPP_8u_C1& mask) {
    int width = data.width();
    int height = data.height();

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    findLocalExtremaEachPixel<<<gridSize, blockSize>>>(data.data(), mask.data(), width, height);
}

bool save_image(const npp::ImageNPP_8u_C1& img_d, const std::string& fname) {
    npp::ImageCPU_8u_C1 img_h(img_d.size());
    img_d.copyTo(img_h.data(), img_h.pitch());
    return saveImage(fname, img_h);
}

int main(int argc, const char** argv) {
    cudaDeviceInit();
    if (!printfNPPinfo()) return 0;

    const std::string ifname("../data/3.2.25.png");
    npp::ImageCPU_8u_C1 host_src;
    if (!loadImage(ifname, host_src)) return 0;
    npp::ImageNPP_8u_C1 device_src(host_src);

    const int cols = device_src.width();
    const int rows = device_src.height();
    const NppiSize roi{cols, rows};

    #if 0
    save_image(device_src, "originalcopy.png");
    #endif

    npp::ImageCPU_32f_C1 device_src_32f(device_src.size());
    NppStatus status = nppiConvert_8u32f_C1R(
        device_src.data(), device_src.pitch(),
        device_src_32f.data(), device_src_32f.pitch(), roi);

    if (status != NPP_SUCCESS) {
        std::cerr << "Error converting image to 32-bit float: " << status << std::endl;
        return 0;
    }

    std::cout << "Image size: " << cols << " x " << rows << std::endl;

    npp::ImageNPP_32f_C1 device_blured(device_src_32f.size());
    status = nppiFilterGauss_32f_C1R(
        device_src_32f.data(), device_src_32f.pitch(),
        device_blured.data(), device_blured.pitch(), roi, NPP_MASK_SIZE_5_X_5);

    if (status != NPP_SUCCESS) {
        std::cerr << "Gaussian blur failed: " << status << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        return 0;
    }

    #if 0
    save_image(oDeviceBlured, "blured.png");
    #endif

    npp::ImageNPP_32f_C1 device_gradient(device_blured.size());
    Npp32f gradkernel[] = {-1.0f, 0.0f, 1.0f};
    int kernelSize = sizeof(gradkernel) / sizeof(gradkernel[0]);
    int anchor = kernelSize / 2;

    status = nppiFilterColumn_32f_C1R(
        device_blured.data(), device_blured.pitch(),
        device_gradient.data(), device_gradient.pitch(),
        roi, gradkernel, kernelSize, anchor);
    
    if (status != NPP_SUCCESS) {
        std::cerr << "Gradient filter failed: " << status << std::endl;
        return 0;
    }

    npp::ImageNPP_8u_C1 device_edges(device_gradient.size());
    find_extrema(device_gradient, device_edges);

    if (!save_image(device_edges, "egdes.png")) {
        std::cerr << "Failed to save output image" << std::endl;
    }

    nppiFree(device_edges.data());
    nppiFree(device_gradient.data());
    nppiFree(device_src.data());
    nppiFree(device_src_32f.data());
    nppiFree(device_blured.data());

    return 0;
}
