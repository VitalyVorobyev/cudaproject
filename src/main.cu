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

// std
#include <iostream>
#include <string>

// opencv
#include <opencv2/imgcodecs.hpp>

#include <npp.h>
#include <nppi.h>

#include "helper_cuda.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"

void cudaDeviceInit() {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(-1);
    }

    int dev = 0;  // findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));
}

bool printfNPPinfo() {
    const NppLibraryVersion *libVer = nppGetLibVersion();

    std::cout << "NPP Library Version " <<
        libVer->major << '.' << libVer->minor << '.' << libVer->build << '\n';

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::cout << "  CUDA Driver  Version: "
        << driverVersion / 1000 << '.' << (driverVersion % 100) / 10 << ",\n"
        << "  CUDA Runtime Version: "
        << runtimeVersion / 1000 << '.' << (runtimeVersion % 100) / 10 << '\n';

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

bool loadImage(const std::string& filename, npp::ImageCPU_8u_C1& oHostSrc) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }

    oHostSrc = npp::ImageCPU_8u_C1(img.cols, img.rows);
    std::memcpy(oHostSrc.data(), img.data, img.cols * img.rows * sizeof(unsigned char));
    return true;
}

bool saveImage(const std::string& filename, const npp::ImageCPU_8u_C1& oHostSrc) {
    #if 1
    cv::Mat1b img(oHostSrc.height(), oHostSrc.width());
    std::cout << "Image size: " << oHostSrc.width() << " x " << oHostSrc.height() << std::endl;
    std::memcpy(img.data, oHostSrc.data(), oHostSrc.width() * oHostSrc.height() * sizeof(unsigned char));
    #else
    cv::Mat1b img((int)oHostSrc.height(), (int)oHostSrc.width(), (void*)oHostSrc.data());
    #endif
    return cv::imwrite(filename, img);
}

int main(int argc, const char** argv) {
    cudaDeviceInit();
    if (!printfNPPinfo()) return 0;

    const std::string ifname("../data/3.2.25.png");
    npp::ImageCPU_8u_C1 oHostSrc;
    if (!loadImage(ifname, oHostSrc)) return 0;
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    #if 0
    npp::ImageCPU_8u_C1 oHostSrcCopy(oDeviceSrc.size());
    oDeviceSrc.copyTo(oHostSrcCopy.data(), oHostSrcCopy.pitch());
    const std::string ofnameoriginal("originalcopy.png");
    saveImage(ofnameoriginal, oHostSrcCopy);
    #endif

    const int cols = oDeviceSrc.width();
    const int rows = oDeviceSrc.height();
    const int step = oDeviceSrc.pitch();  // cols * sizeof(Npp8u);
    const NppiSize roi{cols, rows};
    std::cout << "Image size: " << cols << " x " << rows << ", step: " << step << std::endl;

    npp::ImageNPP_8u_C1 oDeviceBlured(cols, rows);
    NppStatus blurstatus = nppiFilterGauss_8u_C1R(
        oDeviceSrc.data(), step,
        oDeviceBlured.data(), step, roi, NPP_MASK_SIZE_7_X_7);

    if (blurstatus != NPP_SUCCESS) {
        std::cerr << "Gaussian blue failed" << std::endl;
        return 0;
    }

    npp::ImageCPU_8u_C1 oHostBlured(oDeviceBlured.size());
    oDeviceBlured.copyTo(oHostBlured.data(), oHostBlured.pitch());
    const std::string ofnameblured("blured.png");
    saveImage(ofnameblured, oHostBlured);

    #if 0
    npp::ImageNPP_8u_C1 oDeviceDst(cols, rows);
    nppi:NppStatus scolfil = nppiFilterColumn_8u_C1R();
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    const std::string ofname("egdes.png");
    saveImage(ofname, oHostDst);
    nppiFree(oDeviceDst.data());

    #endif

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceBlured.data());

    return 0;
}
