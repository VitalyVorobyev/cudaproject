// std
#include <iostream>
#include <string>

#include "helper_cuda.h"

int main(int argc, char** argv) {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    std::cout << deviceCount << " devices found" << std::endl;

    return 0;
}
