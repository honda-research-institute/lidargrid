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