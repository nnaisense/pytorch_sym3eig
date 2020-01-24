import torch
import sym3eig_cpu
import numpy as np
import time
gpu = False

if torch.cuda.is_available():
    gpu = True
    import sym3eig_cuda


def get_func(name, tensor):
    module = sym3eig_cuda if tensor.is_cuda else sym3eig_cpu
    return getattr(module, name)

class Sym3Eig(object):
    r""" Computes eigenvectors and eigenvalues of symmetric 3x3 matrices in a batch.

    Args:
        x (:class:`Tensor`): Input symmetric 3x3 matrices
            (batch_size x 3 x 3).

    :rtype: :class:`Tensor`
    """

    @staticmethod
    def apply(x):
        eig_val, eig_vec = Sym3Eig_core.apply(x)

        return eig_val, eig_vec




class Sym3Eig_core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrices):
        matrices = (matrices + torch.transpose(matrices, 1, 2))/2
        op = get_func('sym3eig_fw', matrices)
        eig_val, eig_vec = op(matrices)
        ctx.save_for_backward(eig_val, eig_vec)
        return eig_val, eig_vec

    @staticmethod
    def backward(ctx, eig_val_grad, eig_vec_grad):
        eig_val, eig_vec = ctx.saved_tensors
        grad_matrices = None

        if ctx.needs_input_grad[0]:
            op = get_func('sym3eig_bw', eig_val_grad)
            eig_val_grad = eig_val_grad.contiguous()
            eig_vec_grad = eig_vec_grad.contiguous()
            grad_matrices = op(eig_vec_grad, eig_vec, eig_val_grad, eig_val)
            grad_matrices = (grad_matrices + torch.transpose(grad_matrices, 1, 2))/2
            '''
            # Version using pytorch ops
            ut = torch.transpose(eig_vec, 1, 2)
            u = eig_vec
            gu = eig_vec_grad
            s = eig_val
            gs = eig_val_grad
            gs = torch.Tensor([gs[:,0],.0,.0,.0,gs[:,1],.0,.0,.0,gs[:,2]]).view(-1,3,3).double()

            F = 1/(s.unsqueeze(1).expand(-1,3,-1) - s.unsqueeze(2).expand(-1,-1,3))
            F[:,0,0] = F[:,1,1] = F[:,2,2] = 0.0
            X = torch.matmul(ut, gu)
            X = F*X
            grad_matrices = torch.matmul(u, torch.matmul(X, ut))
            val = torch.matmul(u, torch.matmul(gs, ut))
            grad_matrices = grad_matrices + val
            '''
            grad_matrices[torch.isnan(grad_matrices)] = 0.0
        return grad_matrices

