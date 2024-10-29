// std
#include <string>

#include "common/helper_cuda.h"

int main(int argc, char** argv) {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    return 0;
}
