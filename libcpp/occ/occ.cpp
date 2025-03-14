//  Copyright (c) Honda Research Institute USA, Inc.
//  Modified from https://github.com/tarashakhurana/4d-occ-forecasting/tree/main/lib/dvr 
//  which is provided under the MIT license
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include <vector>
#include <torch/extension.h>

/* CUDA forward declarations */

torch::Tensor pcld2occ3d_cuda(torch::Tensor points, const std::vector<int> grid);

std::vector<torch::Tensor> render_demo_cuda(const torch::Tensor& occ,
                                            const torch::Tensor& pixel,
                                            const torch::Tensor& rotation,
                                            const torch::Tensor& translation,
                                            const torch::Tensor& scale);

std::vector<torch::Tensor> render_cuda(torch::Tensor sigma,
                                       torch::Tensor origin,
                                       torch::Tensor points);

torch::Tensor pcldmask_cuda(const torch::Tensor points, 
                            const torch::Tensor bboxes);

/* Checkers Define */

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

/* C++ interface */

torch::Tensor pcld2occ3d(torch::Tensor points, const std::vector<int> grid) {
    CHECK_INPUT(points);
    return pcld2occ3d_cuda(points, grid);
}

std::vector<torch::Tensor> render_demo(const torch::Tensor& occ,
                                       const torch::Tensor& pixel,
                                       const torch::Tensor& rotation,
                                       const torch::Tensor& translation,
                                       const torch::Tensor& scale) {
    CHECK_INPUT(occ);
    CHECK_INPUT(pixel);
    CHECK_INPUT(rotation);
    CHECK_INPUT(translation);
    CHECK_INPUT(scale);
    return render_demo_cuda(occ, pixel, rotation, translation, scale);
}

std::vector<torch::Tensor> render(torch::Tensor sigma,
                                  torch::Tensor origin,
                                  torch::Tensor points){
    CHECK_INPUT(sigma);
    CHECK_INPUT(origin);
    CHECK_INPUT(points);
    return render_cuda(sigma, origin, points);
}

torch::Tensor pcldmask(torch::Tensor points, torch::Tensor bboxes) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(points);
    return pcldmask_cuda(points, bboxes);
}

/* Pybind11 Module */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pcld2occ3d", &pcld2occ3d, "Convert point cloud to 3d opacity grids (Initialization)");
    m.def("render_demo", &render_demo, "Render opacity field demo (Forward pass only, mainly used for visualization)");
    m.def("render", &render, "Render ray distance on opacity grids and calculate gradients");
    m.def("pcldmask", &pcldmask, "Generate points mask from 3d bbox");
}
