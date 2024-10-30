#pragma once

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

    if (!deviceCount) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(-1);
    }

    int dev = 0;  // findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

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
