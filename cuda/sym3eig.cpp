#include <torch/extension.h>

#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> sym3eig_fw_cuda(at::Tensor x);
at::Tensor sym3eig_bw_cuda(at::Tensor eig_vec_grad, at::Tensor eig_vec, at::Tensor eig_val_grad,
                             at::Tensor eig_val);

std::tuple<at::Tensor, at::Tensor> sym3eig_fw(at::Tensor x) {
  CHECK_INPUT(x);
  return sym3eig_fw_cuda(x);
}

at::Tensor sym3eig_bw(at::Tensor eig_vec_grad, at::Tensor eig_vec, at::Tensor eig_val_grad,
                             at::Tensor eig_val) {
  CHECK_INPUT(eig_vec_grad);
  CHECK_INPUT(eig_vec);
  CHECK_INPUT(eig_val_grad);
  CHECK_INPUT(eig_val);
  return sym3eig_bw_cuda(eig_vec_grad, eig_vec, eig_val_grad, eig_val);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sym3eig_fw", &sym3eig_fw, "Sym3Eig Forward (GPU)");
  m.def("sym3eig_bw", &sym3eig_bw, "Sym3Eig Backward (GPU)");
}
