#include <ATen/ATen.h>

#include <math.h>
#include <iostream>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void eig_val_kernel(const scalar_t *__restrict__ x,
                               scalar_t *__restrict__ eig_val, size_t numel) {
  const ptrdiff_t i = (ptrdiff_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < numel) {
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm
    const ptrdiff_t m_idx = 9 * i;
    const ptrdiff_t v_idx = 3 * i;
    const scalar_t x11 = x[m_idx + 3 * 0 + 0];
    const scalar_t x12 = x[m_idx + 3 * 0 + 1];
    const scalar_t x13 = x[m_idx + 3 * 0 + 2];
    const scalar_t x21 = x[m_idx + 3 * 1 + 0];
    const scalar_t x22 = x[m_idx + 3 * 1 + 1];
    const scalar_t x23 = x[m_idx + 3 * 1 + 2];
    const scalar_t x31 = x[m_idx + 3 * 2 + 0];
    const scalar_t x32 = x[m_idx + 3 * 2 + 1];
    const scalar_t x33 = x[m_idx + 3 * 2 + 2];

    const scalar_t p1 = x12 * x12 + x13 * x13 + x23 * x23;

    if (p1 == 0) {
      eig_val[v_idx + 0] = x11;
      eig_val[v_idx + 1] = x22;
      eig_val[v_idx + 2] = x33;
    } else {
      const scalar_t q = (x11 + x22 + x33) / 3.0;
      const scalar_t p2 = (x11 - q) * (x11 - q) + (x22 - q) * (x22 - q) +
                          (x33 - q) * (x33 - q) + 2 * p1;
      const scalar_t p = sqrt(p2 / 6.0);

      const scalar_t b11 = (1.0 / p) * (x11 - q);
      const scalar_t b12 = (1.0 / p) * x12;
      const scalar_t b13 = (1.0 / p) * x13;
      const scalar_t b21 = (1.0 / p) * x21;
      const scalar_t b22 = (1.0 / p) * (x22 - q);
      const scalar_t b23 = (1.0 / p) * x23;
      const scalar_t b31 = (1.0 / p) * x31;
      const scalar_t b32 = (1.0 / p) * x32;
      const scalar_t b33 = (1.0 / p) * (x33 - q);

      scalar_t r = b11 * b22 * b33 + b12 * b23 * b31 + b13 * b21 * b32 -
                   b13 * b22 * b31 - b12 * b21 * b33 - b11 * b23 * b32;
      r = r / 2.0;

      scalar_t phi;
      if (r <= -1) {
        phi = M_PI / 3.0;
      } else if (r >= 1) {
        phi = 0;
      } else {
        phi = acos(r) / 3.0;
      }

      eig_val[v_idx + 0] = q + 2 * p * cos(phi);
      eig_val[v_idx + 2] = q + 2 * p * cos(phi + (2 * M_PI / 3));
      eig_val[v_idx + 1] = 3 * q - eig_val[v_idx + 0] - eig_val[3 * i + 2];
    }
  }
}

template <typename scalar_t>
__global__ void eig_vec_kernel(const scalar_t *__restrict__ x,
                               const scalar_t *__restrict__ eig_val,
                               scalar_t *__restrict__ eig_vec, size_t numel) {
  const ptrdiff_t e = (ptrdiff_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < numel) {
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    const ptrdiff_t i = e / 3;

    const scalar_t x11 = x[9 * i + 3 * 0 + 0] - eig_val[e];
    const scalar_t x12 = x[9 * i + 3 * 0 + 1];
    const scalar_t x13 = x[9 * i + 3 * 0 + 2];
    const scalar_t x21 = x[9 * i + 3 * 1 + 0];
    const scalar_t x22 = x[9 * i + 3 * 1 + 1] - eig_val[e];
    const scalar_t x23 = x[9 * i + 3 * 1 + 2];
    const scalar_t x31 = x[9 * i + 3 * 2 + 0];
    const scalar_t x32 = x[9 * i + 3 * 2 + 1];
    const scalar_t x33 = x[9 * i + 3 * 2 + 2] - eig_val[e];

    const scalar_t r12_1 = x12 * x23 - x13 * x22;
    const scalar_t r12_2 = x13 * x21 - x11 * x23;
    const scalar_t r12_3 = x11 * x22 - x12 * x21;
    const scalar_t r13_1 = x12 * x33 - x13 * x32;
    const scalar_t r13_2 = x13 * x31 - x11 * x33;
    const scalar_t r13_3 = x11 * x32 - x12 * x31;
    const scalar_t r23_1 = x22 * x33 - x23 * x32;
    const scalar_t r23_2 = x23 * x31 - x21 * x33;
    const scalar_t r23_3 = x21 * x32 - x22 * x31;

    const scalar_t d1 = r12_1 * r12_1 + r12_2 * r12_2 + r12_3 * r12_3;
    const scalar_t d2 = r13_1 * r13_1 + r13_2 * r13_2 + r13_3 * r13_3;
    const scalar_t d3 = r23_1 * r23_1 + r23_2 * r23_2 + r23_3 * r23_3;

    scalar_t d_max = d1;
    ptrdiff_t i_max = 0;

    if (d2 > d_max) {
      d_max = d2;
      i_max = 1;
    }

    if (d3 > d_max) {
      i_max = 2;
    }

    if (i_max == 0) {
      eig_vec[9 * i + 3 * 0 + e % 3] = r12_1 / sqrt(d1);
      eig_vec[9 * i + 3 * 1 + e % 3] = r12_2 / sqrt(d1);
      eig_vec[9 * i + 3 * 2 + e % 3] = r12_3 / sqrt(d1);
    } else if (i_max == 1) {
      eig_vec[9 * i + 3 * 0 + e % 3] = r13_1 / sqrt(d2);
      eig_vec[9 * i + 3 * 1 + e % 3] = r13_2 / sqrt(d2);
      eig_vec[9 * i + 3 * 2 + e % 3] = r13_3 / sqrt(d2);
    } else {
      eig_vec[9 * i + 3 * 0 + e % 3] = r23_1 / sqrt(d3);
      eig_vec[9 * i + 3 * 1 + e % 3] = r23_2 / sqrt(d3);
      eig_vec[9 * i + 3 * 2 + e % 3] = r23_3 / sqrt(d3);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> sym3eig_fw_cuda(at::Tensor x) {
  std::vector<int64_t> size(x.sizes().begin(), x.sizes().end());
  size.pop_back();
  auto eig_val = at::empty(size, x.options());
  auto eig_vec = at::empty_like(x);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "eig_kernel", [&] {
    eig_val_kernel<scalar_t><<<BLOCKS(eig_val.numel() / 3), THREADS>>>(
        x.data<scalar_t>(), eig_val.data<scalar_t>(), eig_val.numel() / 3);
    eig_vec_kernel<scalar_t><<<BLOCKS(eig_val.numel()), THREADS>>>(
        x.data<scalar_t>(), eig_val.data<scalar_t>(), eig_vec.data<scalar_t>(),
        eig_val.numel());
  });

  return std::make_tuple(eig_val, eig_vec);
}


template <typename scalar_t>
__global__ void eig_bw_kernel(const scalar_t *__restrict__ U_grad, // Gradients of eig_vec
                              const scalar_t *__restrict__ U, // eig_vec
                              const scalar_t *__restrict__ S_grad, // Gradients of eig_val
                              const scalar_t *__restrict__ S, // eig_val
                              scalar_t *__restrict__ X_grad, // Output: Gradients of input matrix
                              size_t numel) {
  const ptrdiff_t i = (ptrdiff_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < numel) {
    const ptrdiff_t m_idx = 9 * i;
    const ptrdiff_t v_idx = 3 * i;
    scalar_t ut[9];
    scalar_t eig_val[3];
    scalar_t tmp[9];
    scalar_t x[9];

    //Todo make this coalesced (transpose before)
    ut[0] = U[m_idx + 0];
    ut[1] = U[m_idx + 3];
    ut[2] = U[m_idx + 6];
    ut[3] = U[m_idx + 1];
    ut[4] = U[m_idx + 4];
    ut[5] = U[m_idx + 7];
    ut[6] = U[m_idx + 2];
    ut[7] = U[m_idx + 5];
    ut[8] = U[m_idx + 8];

    x[0] = U_grad[m_idx + 0];
    x[1] = U_grad[m_idx + 1];
    x[2] = U_grad[m_idx + 2];
    x[3] = U_grad[m_idx + 3];
    x[4] = U_grad[m_idx + 4];
    x[5] = U_grad[m_idx + 5];
    x[6] = U_grad[m_idx + 6];
    x[7] = U_grad[m_idx + 7];
    x[8] = U_grad[m_idx + 8];

    eig_val[0] = S[v_idx + 0];
    eig_val[1] = S[v_idx + 1];
    eig_val[2] = S[v_idx + 2];

    // X <- U^T*dL/dU
    tmp[0] = ut[0]*x[0] + ut[1]*x[3] + ut[2]*x[6];
    tmp[1] = ut[0]*x[1] + ut[1]*x[4] + ut[2]*x[7];
    tmp[2] = ut[0]*x[2] + ut[1]*x[5] + ut[2]*x[8];
    tmp[3] = ut[3]*x[0] + ut[4]*x[3] + ut[5]*x[6];
    tmp[4] = ut[3]*x[1] + ut[4]*x[4] + ut[5]*x[7];
    tmp[5] = ut[3]*x[2] + ut[4]*x[5] + ut[5]*x[8];
    tmp[6] = ut[6]*x[0] + ut[7]*x[3] + ut[8]*x[6];
    tmp[7] = ut[6]*x[1] + ut[7]*x[4] + ut[8]*x[7];
    tmp[8] = ut[6]*x[2] + ut[7]*x[5] + ut[8]*x[8];

    /* X <- (X + X^T)*/
    x[0] = tmp[0];
    x[1] = tmp[1];
    x[2] = tmp[2];
    x[3] = tmp[3];
    x[4] = tmp[4];
    x[5] = tmp[5];
    x[6] = tmp[6];
    x[7] = tmp[7];
    x[8] = tmp[8];

    /* X <- K^T o X */
    x[0] *= 0.0;
    x[1] *= 1./(eig_val[1] - eig_val[0]);
    x[2] *= 1./(eig_val[2] - eig_val[0]);
    x[3] *= 1./(eig_val[0] - eig_val[1]);
    x[4] *= 0.0;
    x[5] *= 1./(eig_val[2] - eig_val[1]);
    x[6] *= 1./(eig_val[0] - eig_val[2]);
    x[7] *= 1./(eig_val[1] - eig_val[2]);
    x[8] *= 0.0;

    /* X <- X + S */
    x[0] += S_grad[v_idx + 0];
    x[4] += S_grad[v_idx + 1];
    x[8] += S_grad[v_idx + 2];

    // X <- U*X
    tmp[0] = ut[0]*x[0] + ut[3]*x[3] + ut[6]*x[6];
    tmp[1] = ut[0]*x[1] + ut[3]*x[4] + ut[6]*x[7];
    tmp[2] = ut[0]*x[2] + ut[3]*x[5] + ut[6]*x[8];
    tmp[3] = ut[1]*x[0] + ut[4]*x[3] + ut[7]*x[6];
    tmp[4] = ut[1]*x[1] + ut[4]*x[4] + ut[7]*x[7];
    tmp[5] = ut[1]*x[2] + ut[4]*x[5] + ut[7]*x[8];
    tmp[6] = ut[2]*x[0] + ut[5]*x[3] + ut[8]*x[6];
    tmp[7] = ut[2]*x[1] + ut[5]*x[4] + ut[8]*x[7];
    tmp[8] = ut[2]*x[2] + ut[5]*x[5] + ut[8]*x[8];

    // X <- X*U^T
    X_grad[m_idx + 0] = tmp[0]*ut[0] + tmp[1]*ut[3] + tmp[2]*ut[6];
    X_grad[m_idx + 1] = tmp[0]*ut[1] + tmp[1]*ut[4] + tmp[2]*ut[7];
    X_grad[m_idx + 2] = tmp[0]*ut[2] + tmp[1]*ut[5] + tmp[2]*ut[8];
    X_grad[m_idx + 3] = tmp[3]*ut[0] + tmp[4]*ut[3] + tmp[5]*ut[6];
    X_grad[m_idx + 4] = tmp[3]*ut[1] + tmp[4]*ut[4] + tmp[5]*ut[7];
    X_grad[m_idx + 5] = tmp[3]*ut[2] + tmp[4]*ut[5] + tmp[5]*ut[8];
    X_grad[m_idx + 6] = tmp[6]*ut[0] + tmp[7]*ut[3] + tmp[8]*ut[6];
    X_grad[m_idx + 7] = tmp[6]*ut[1] + tmp[7]*ut[4] + tmp[8]*ut[7];
    X_grad[m_idx + 8] = tmp[6]*ut[2] + tmp[7]*ut[5] + tmp[8]*ut[8];
  }
}


at::Tensor sym3eig_bw_cuda(at::Tensor eig_vec_grad,
                                               at::Tensor eig_vec,
                                               at::Tensor eig_val_grad,
                                               at::Tensor eig_val) {
  auto x_grad = at::empty_like(eig_vec);
  //std::cout << (eig_val.numel() / 3) << BLOCKS(eig_val.numel() / 3) << THREADS << "\n";
  AT_DISPATCH_FLOATING_TYPES(eig_vec.type(), "eig_bw_kernel", [&] {
    eig_bw_kernel<scalar_t><<<BLOCKS(eig_val.numel() / 3), THREADS>>>(
        eig_vec_grad.data<scalar_t>(), eig_vec.data<scalar_t>(),
        eig_val_grad.data<scalar_t>(), eig_val.data<scalar_t>(),
        x_grad.data<scalar_t>(), eig_val.numel() / 3);
  });

  return x_grad;
}

