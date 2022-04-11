#include <torch/extension.h>

auto d_sigmoid(const torch::Tensor& z) -> torch::Tensor {
  const auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &d_sigmoid, "my_sigmoid");
}