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
#include <torch/extension.h>
#include <vector>

/*
 * CUDA forward declarations
 */

std::vector<torch::Tensor> render_forward_cuda(torch::Tensor sigma,
                                               torch::Tensor origin,
                                               torch::Tensor points,
                                               torch::Tensor tindex,
                                               const std::vector<int> grid,
                                               std::string phase_name);

std::vector<torch::Tensor>
render_cuda(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
            torch::Tensor tindex, torch::Tensor pcldmask, std::string loss_name);

torch::Tensor init_cuda(torch::Tensor points, torch::Tensor tindex,
                        const std::vector<int> grid);


/*
 * C++ interface
 */

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
render_forward(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
               torch::Tensor tindex, const std::vector<int> grid,
               std::string phase_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_forward_cuda(sigma, origin, points, tindex, grid, phase_name);
}


std::vector<torch::Tensor> render(torch::Tensor sigma, torch::Tensor origin,
                                  torch::Tensor points, torch::Tensor tindex, torch::Tensor pcldmask,
                                  std::string loss_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(pcldmask);
  CHECK_INPUT(tindex);
  return render_cuda(sigma, origin, points, tindex, pcldmask, loss_name);
}

torch::Tensor init(torch::Tensor points, torch::Tensor tindex,
                   const std::vector<int> grid) {
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return init_cuda(points, tindex, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize");
  m.def("render", &render, "Render");
  m.def("render_forward", &render_forward, "Render (forward pass only)");
}
