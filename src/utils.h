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

#include "kernels.h"

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
    std::memcpy(img.data, oHostSrc.data(), oHostSrc.width() * oHostSrc.height() * sizeof(unsigned char));
    #else
    cv::Mat1b img((int)oHostSrc.height(), (int)oHostSrc.width(), (void*)oHostSrc.data());
    #endif
    return cv::imwrite(filename, img);
}

bool find_extrema(const npp::ImageNPP_32f_C1& data, npp::ImageNPP_8u_C1& mask) {
    int width = data.width();
    int height = data.height();
    int step = data.pitch() / sizeof(float);
    int maskstep = mask.pitch() / sizeof(unsigned char);

    std::cout << data.width() << " vs. " << mask.width() << std::endl;
    std::cout << data.height() << " vs. " << mask.height() << std::endl;
    std::cout << step << " vs. " << maskstep << std::endl;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    findLocalExtremaEachPixel<<<gridSize, blockSize>>>(
        data.data(), mask.data(), width, height, step, maskstep);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

bool save_image(const npp::ImageNPP_8u_C1& img_d, const std::string& fname) {
    try {
        npp::ImageCPU_8u_C1 img_h(img_d.size());
        img_d.copyTo(img_h.data(), img_h.pitch());
        if (!saveImage(fname, img_h)) {
            std::cerr << "Failed to save " << fname << std::endl;
            return false;
        }
    } catch (npp::Exception& ex) {
        std::cerr << "Exception during save image: " << ex.message() << std::endl;
        return false;
    }

    return true;
}

bool convert_to_float(const npp::ImageNPP_8u_C1& src, npp::ImageNPP_32f_C1& dst) {
    const int cols = src.width();
    const int rows = src.height();
    const NppiSize roi{cols, rows};

    try {
        NppStatus status = nppiConvert_8u32f_C1R(
            src.data(), src.pitch(), dst.data(), dst.pitch(), roi);
        
        if (status != NPP_SUCCESS) {
            std::cerr << "Error converting image to 32-bit float: " << status << std::endl;
            return false;
        }
    } catch (npp::Exception& ex) {
        std::cerr << "Exception during converting image to 32-bit: " << ex.message() << std::endl;
        return false;
    }

    return true;
}

bool gaussian_blur(const npp::ImageNPP_8u_C1& src, npp::ImageNPP_8u_C1& dst) {
    const int cols = src.width();
    const int rows = src.height();
    const int step = src.pitch();
    const NppiSize roi{cols, rows};

    try {
        NppStatus status = nppiFilterGauss_8u_C1R(
            src.data(), step, dst.data(), step, roi, NPP_MASK_SIZE_7_X_7);

        if (status != NPP_SUCCESS) {
            std::cerr << "Gaussian blur (8u) failed: " << status << std::endl;
            return false;
        }
    } catch (npp::Exception& ex) {
        std::cerr << "Exception during Gaussian blur (8u): " << ex.message() << std::endl;
        return false;
    }

    return true;
}

bool image_gradient(const npp::ImageNPP_32f_C1& src, npp::ImageNPP_32f_C1& dst) {
    const int cols = src.width();
    const int rows = src.height();
    const int step = src.pitch();
    const NppiSize roi{cols, rows};

    Npp32f gradkernel[] = {-1.0f, 0.0f, 1.0f};
    int kernelSize = sizeof(gradkernel) / sizeof(gradkernel[0]);
    int anchor = kernelSize / 2;

    try {
        NppStatus status = nppiFilterColumn_32f_C1R(
            src.data(), step, dst.data(), step, roi, gradkernel, kernelSize, anchor);

        if (status != NPP_SUCCESS) {
            std::cerr << "Gradient filter failed: " << status << std::endl;
            return false;
        }
    } catch (npp::Exception& ex) {
        std::cerr << "Exception during Gradient filter: " << ex.message() << std::endl;
        return false;
    }

    return true;
}
