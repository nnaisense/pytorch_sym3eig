#include <torch/extension.h>
#include <cmath>
#include <iostream>


#define SYM3EIG_FORWARD(MATRIX)                                                \
  [&]() -> std::tuple<at::Tensor, at::Tensor> {                                \
    auto num_matrices = MATRIX.size(0);                                        \
    auto eig_val = at::empty({num_matrices, 3}, MATRIX.options());             \
    auto eig_vec = at::empty_like(MATRIX);                                     \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(MATRIX.type(), "sym3eig_fw", [&] {       \
      auto matrix_data = MATRIX.data<scalar_t>();                              \
      auto eig_vec_data = eig_vec.data<scalar_t>();                            \
      auto eig_val_data = eig_val.data<scalar_t>();                            \
      for (ptrdiff_t i = 0; i < num_matrices; i++) {                           \
        const int m_idx = 9 * i;                                               \
        const int v_idx = 3 * i;                                               \
                                                                               \
        const scalar_t x11 = matrix_data[m_idx + 0];                           \
        const scalar_t x12 = matrix_data[m_idx + 1];                           \
        const scalar_t x13 = matrix_data[m_idx + 2];                           \
        const scalar_t x21 = matrix_data[m_idx + 3];                           \
        const scalar_t x22 = matrix_data[m_idx + 4];                           \
        const scalar_t x23 = matrix_data[m_idx + 5];                           \
        const scalar_t x31 = matrix_data[m_idx + 6];                           \
        const scalar_t x32 = matrix_data[m_idx + 7];                           \
        const scalar_t x33 = matrix_data[m_idx + 8];                           \
                                                                               \
        const scalar_t p1 = x12 * x12 + x13 * x13 + x23 * x23;                 \
                                                                               \
        if (p1 == 0) {                                                         \
            eig_val[v_idx + 0] = x11;                                          \
            eig_val[v_idx + 1] = x22;                                          \
            eig_val[v_idx + 2] = x33;                                          \
        } else {                                                               \
            const scalar_t q = (x11 + x22 + x33) / 3.0;                        \
            const scalar_t p2 = (x11 - q) * (x11 - q) + (x22 - q) * (x22 - q)  \
                              + (x33 - q) * (x33 - q) + 2 * p1;                \
            const scalar_t p = std::sqrt(p2 / 6.0);                            \
                                                                               \
            const scalar_t b11 = (1.0 / p) * (x11 - q);                        \
            const scalar_t b12 = (1.0 / p) * x12;                              \
            const scalar_t b13 = (1.0 / p) * x13;                              \
            const scalar_t b21 = (1.0 / p) * x21;                              \
            const scalar_t b22 = (1.0 / p) * (x22 - q);                        \
            const scalar_t b23 = (1.0 / p) * x23;                              \
            const scalar_t b31 = (1.0 / p) * x31;                              \
            const scalar_t b32 = (1.0 / p) * x32;                              \
            const scalar_t b33 = (1.0 / p) * (x33 - q);                        \
                                                                               \
            scalar_t r = b11 * b22 * b33 + b12 * b23 * b31 + b13 * b21 * b32 - \
                       b13 * b22 * b31 - b12 * b21 * b33 - b11 * b23 * b32;    \
            r = r / 2.0;                                                       \
                                                                               \
            scalar_t phi;                                                      \
            if (r <= -1) {                                                     \
                phi = M_PI / 3.0;                                              \
            } else if (r >= 1) {                                               \
                phi = 0;                                                       \
            } else {                                                           \
                phi = acos(r) / 3.0;                                           \
            }                                                                  \
                                                                               \
            eig_val_data[v_idx + 0] = q + 2 * p * cos(phi);                    \
            eig_val_data[v_idx + 2] = q + 2 * p * cos(phi + (2 * M_PI / 3));   \
            eig_val_data[v_idx + 1] = 3 * q - eig_val_data[v_idx + 0]          \
                                            - eig_val_data[v_idx + 2];         \
                                                                               \
        for (ptrdiff_t e_it = 0; e_it < 3; e_it++) {                           \
            const int e = 3*i+ e_it;                                           \
            const int m_idx = 9 * i;                                           \
            const scalar_t x11 = matrix_data[m_idx + 0] - eig_val_data[e];     \
            const scalar_t x12 = matrix_data[m_idx + 1];                       \
            const scalar_t x13 = matrix_data[m_idx + 2];                       \
            const scalar_t x21 = matrix_data[m_idx + 3];                       \
            const scalar_t x22 = matrix_data[m_idx + 4] - eig_val_data[e];     \
            const scalar_t x23 = matrix_data[m_idx + 5];                       \
            const scalar_t x31 = matrix_data[m_idx + 6];                       \
            const scalar_t x32 = matrix_data[m_idx + 7];                       \
            const scalar_t x33 = matrix_data[m_idx + 8] - eig_val_data[e];     \
                                                                               \
            const scalar_t r12_1 = x12 * x23 - x13 * x22;                      \
            const scalar_t r12_2 = x13 * x21 - x11 * x23;                      \
            const scalar_t r12_3 = x11 * x22 - x12 * x21;                      \
            const scalar_t r13_1 = x12 * x33 - x13 * x32;                      \
            const scalar_t r13_2 = x13 * x31 - x11 * x33;                      \
            const scalar_t r13_3 = x11 * x32 - x12 * x31;                      \
            const scalar_t r23_1 = x22 * x33 - x23 * x32;                      \
            const scalar_t r23_2 = x23 * x31 - x21 * x33;                      \
            const scalar_t r23_3 = x21 * x32 - x22 * x31;                      \
                                                                               \
            const scalar_t d1 = r12_1 * r12_1 + r12_2 * r12_2 + r12_3 * r12_3; \
            const scalar_t d2 = r13_1 * r13_1 + r13_2 * r13_2 + r13_3 * r13_3; \
            const scalar_t d3 = r23_1 * r23_1 + r23_2 * r23_2 + r23_3 * r23_3; \
                                                                               \
            scalar_t d_max = d1;                                               \
            ptrdiff_t i_max = 0;                                               \
                                                                               \
            if (d2 > d_max) {                                                  \
              d_max = d2;                                                      \
              i_max = 1;                                                       \
            }                                                                  \
                                                                               \
            if (d3 > d_max) {                                                  \
              i_max = 2;                                                       \
            }                                                                  \
                                                                               \
            if (i_max == 0) {                                                  \
              eig_vec_data[m_idx + 3 * 0 + e_it] = r12_1 / std::sqrt(d1);      \
              eig_vec_data[m_idx + 3 * 1 + e_it] = r12_2 / std::sqrt(d1);      \
              eig_vec_data[m_idx + 3 * 2 + e_it] = r12_3 / std::sqrt(d1);      \
            } else if (i_max == 1) {                                           \
              eig_vec_data[m_idx + 3 * 0 + e_it] = r13_1 / std::sqrt(d2);      \
              eig_vec_data[m_idx + 3 * 1 + e_it] = r13_2 / std::sqrt(d2);      \
              eig_vec_data[m_idx + 3 * 2 + e_it] = r13_3 / std::sqrt(d2);      \
            } else {                                                           \
              eig_vec_data[m_idx + 3 * 0 + e_it] = r23_1 / std::sqrt(d3);      \
              eig_vec_data[m_idx + 3 * 1 + e_it] = r23_2 / std::sqrt(d3);      \
              eig_vec_data[m_idx + 3 * 2 + e_it] = r23_3 / std::sqrt(d3);      \
            }                                                                  \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  });                                                                          \
  return std::make_tuple(eig_val, eig_vec);                                    \
}()                                                                            \

std::tuple<at::Tensor, at::Tensor> sym3eig_fw(at::Tensor matrix) {
  return SYM3EIG_FORWARD(matrix);
}



#define SYM3EIG_BACKWARD(EIG_VEC_GRAD, EIG_VEC, EIG_VAL_GRAD, EIG_VAL)         \
  [&]() -> at::Tensor {                                                        \
    auto num_matrices = EIG_VEC_GRAD.size(0);                                  \
    auto X_grad = at::empty_like(EIG_VEC_GRAD);                                \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(EIG_VEC.type(), "sym3eig_bw", [&] {     \
      auto U = EIG_VEC.data<scalar_t>();                                       \
      auto U_grad = EIG_VEC_GRAD.data<scalar_t>();                             \
      auto S = EIG_VAL.data<scalar_t>();                                       \
      auto S_grad = EIG_VAL_GRAD.data<scalar_t>();                             \
      auto X_grad_data = X_grad.data<scalar_t>();                         \
                                                                               \
      for (ptrdiff_t i = 0; i < num_matrices; i++) {                           \
        const int m_idx = 9 * i;                                               \
        const int v_idx = 3 * i;                                               \
        scalar_t u[9] = {.0};                                                  \
        scalar_t eig_val[3] = {.0};                                            \
        scalar_t tmp[9] = {.0};                                                \
        scalar_t x[9] = {.0};                                                  \
                                                                               \
        /* u is actually U^T for convenience*/                                 \
        u[0] = U[m_idx + 0];                                                   \
        u[1] = U[m_idx + 3];                                                   \
        u[2] = U[m_idx + 6];                                                   \
        u[3] = U[m_idx + 1];                                                   \
        u[4] = U[m_idx + 4];                                                   \
        u[5] = U[m_idx + 7];                                                   \
        u[6] = U[m_idx + 2];                                                   \
        u[7] = U[m_idx + 5];                                                   \
        u[8] = U[m_idx + 8];                                                   \
                                                                               \
        x[0] = U_grad[m_idx + 0];                                              \
        x[1] = U_grad[m_idx + 1];                                              \
        x[2] = U_grad[m_idx + 2];                                              \
        x[3] = U_grad[m_idx + 3];                                              \
        x[4] = U_grad[m_idx + 4];                                              \
        x[5] = U_grad[m_idx + 5];                                              \
        x[6] = U_grad[m_idx + 6];                                              \
        x[7] = U_grad[m_idx + 7];                                              \
        x[8] = U_grad[m_idx + 8];                                              \
                                                                               \
        eig_val[0] = S[v_idx + 0];                                             \
        eig_val[1] = S[v_idx + 1];                                             \
        eig_val[2] = S[v_idx + 2];                                             \
                                                                               \
        /* X <- U^T*dL/dU */                                                   \
        tmp[0] = u[0]*x[0] + u[1]*x[3] + u[2]*x[6];                            \
        tmp[1] = u[0]*x[1] + u[1]*x[4] + u[2]*x[7];                            \
        tmp[2] = u[0]*x[2] + u[1]*x[5] + u[2]*x[8];                            \
        tmp[3] = u[3]*x[0] + u[4]*x[3] + u[5]*x[6];                            \
        tmp[4] = u[3]*x[1] + u[4]*x[4] + u[5]*x[7];                            \
        tmp[5] = u[3]*x[2] + u[4]*x[5] + u[5]*x[8];                            \
        tmp[6] = u[6]*x[0] + u[7]*x[3] + u[8]*x[6];                            \
        tmp[7] = u[6]*x[1] + u[7]*x[4] + u[8]*x[7];                            \
        tmp[8] = u[6]*x[2] + u[7]*x[5] + u[8]*x[8];                            \
                                                                               \
        /* X <- (X + X^T)*/                                                    \
        x[0] = tmp[0];                                                 \
        x[1] = tmp[1];                                               \
        x[2] = tmp[2];                                                   \
        x[3] = tmp[3];                                                \
        x[4] = tmp[4];                                                \
        x[5] = tmp[5];                                                \
        x[6] = tmp[6];                                                \
        x[7] = tmp[7];                                                \
        x[8] = tmp[8];                                                \
                                                                               \
        /* X <- K^T o X */                                                     \
        x[0] *= 0.0;                                                          \
        x[1] *= 1./(eig_val[1] - eig_val[0]);                                     \
        x[2] *= 1./(eig_val[2] - eig_val[0]);                                     \
        x[3] *= 1./(eig_val[0] - eig_val[1]);                                                   \
        x[4] *= 0.0;                                                              \
        x[5] *= 1./(eig_val[2] - eig_val[1]);                                     \
        x[6] *= 1./(eig_val[0] - eig_val[2]);                                     \
        x[7] *= 1./(eig_val[1] - eig_val[2]);                                     \
        x[8] *= 0.0;                                                                \
                                                                               \
        /* X <- X + S */                                                       \
        x[0] += S_grad[v_idx + 0];                                             \
        x[4] += S_grad[v_idx + 1];                                             \
        x[8] += S_grad[v_idx + 2];                                             \
                                                                               \
        /* X <- U*X */                                                         \
        tmp[0] = u[0]*x[0] + u[3]*x[3] + u[6]*x[6];                            \
        tmp[1] = u[0]*x[1] + u[3]*x[4] + u[6]*x[7];                            \
        tmp[2] = u[0]*x[2] + u[3]*x[5] + u[6]*x[8];                            \
        tmp[3] = u[1]*x[0] + u[4]*x[3] + u[7]*x[6];                            \
        tmp[4] = u[1]*x[1] + u[4]*x[4] + u[7]*x[7];                            \
        tmp[5] = u[1]*x[2] + u[4]*x[5] + u[7]*x[8];                            \
        tmp[6] = u[2]*x[0] + u[5]*x[3] + u[8]*x[6];                            \
        tmp[7] = u[2]*x[1] + u[5]*x[4] + u[8]*x[7];                            \
        tmp[8] = u[2]*x[2] + u[5]*x[5] + u[8]*x[8];                            \
                                                                               \
        /* X <- X*U^T */                                                       \
        X_grad_data[m_idx + 0] = tmp[0]*u[0] + tmp[1]*u[3] + tmp[2]*u[6];      \
        X_grad_data[m_idx + 1] = tmp[0]*u[1] + tmp[1]*u[4] + tmp[2]*u[7];      \
        X_grad_data[m_idx + 2] = tmp[0]*u[2] + tmp[1]*u[5] + tmp[2]*u[8];      \
        X_grad_data[m_idx + 3] = tmp[3]*u[0] + tmp[4]*u[3] + tmp[5]*u[6];      \
        X_grad_data[m_idx + 4] = tmp[3]*u[1] + tmp[4]*u[4] + tmp[5]*u[7];      \
        X_grad_data[m_idx + 5] = tmp[3]*u[2] + tmp[4]*u[5] + tmp[5]*u[8];      \
        X_grad_data[m_idx + 6] = tmp[6]*u[0] + tmp[7]*u[3] + tmp[8]*u[6];      \
        X_grad_data[m_idx + 7] = tmp[6]*u[1] + tmp[7]*u[4] + tmp[8]*u[7];      \
        X_grad_data[m_idx + 8] = tmp[6]*u[2] + tmp[7]*u[5] + tmp[8]*u[8];      \
      }                                                                        \
    });                                                                        \
    return X_grad;                                                             \
  }()

at::Tensor sym3eig_bw(at::Tensor eig_vec_grad, at::Tensor eig_vec,
                     at::Tensor eig_val_grad, at::Tensor eig_val) {
  return SYM3EIG_BACKWARD(eig_vec_grad, eig_vec, eig_val_grad, eig_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sym3eig_fw", &sym3eig_fw, "Sym3Eig Forward (CPU)");
  m.def("sym3eig_bw", &sym3eig_bw, "Sym3Eig Backward (CPU)");
}
