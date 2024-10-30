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

#include "utils.h"

int main(int argc, const char** argv) {
    cudaDeviceInit();
    if (!printfNPPinfo()) return 0;

    const std::string ifname("../data/3.2.25.png");
    npp::ImageCPU_8u_C1 host_src;
    if (!loadImage(ifname, host_src)) return 0;
    npp::ImageNPP_8u_C1 device_src(host_src);

    std::cout << "Image size: " << device_src.width() << " x " << device_src.height()
        << "\npitch " << device_src.pitch() << std::endl;

    if (!save_image(device_src, "originalcopy.png")) return 0;
    
    npp::ImageNPP_32f_C1 device_src_32f(device_src.size());
    if (!convert_to_float(device_src, device_src_32f)) return 0;

    npp::ImageNPP_8u_C1 device_blured(device_src.size());
    if (!gaussian_blur(device_src, device_blured)) return 0;

    save_image(device_blured, "blured.png");

    npp::ImageNPP_32f_C1 device_blured_32f(device_blured.size());
    if (!convert_to_float(device_blured, device_blured_32f)) return 0;

    npp::ImageNPP_32f_C1 device_gradient(device_blured_32f.size());
    if (!image_gradient(device_blured_32f, device_gradient)) return 0;

    npp::ImageNPP_8u_C1 device_edges(device_gradient.size());
    if (!find_extrema(device_gradient, device_edges)) return 0;

    std::cout << "Yeah! " << device_edges.width() << std::endl;
    if (!save_image(device_edges, "egdes.png")) return 0;

    nppiFree(device_edges.data());
    nppiFree(device_gradient.data());
    nppiFree(device_src.data());
    nppiFree(device_src_32f.data());
    nppiFree(device_blured.data());

    return 0;
}
