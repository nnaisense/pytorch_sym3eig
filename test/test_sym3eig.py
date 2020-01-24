from itertools import product

from torch.autograd import gradcheck
import pytest
import numpy as np
import torch
from torch_sym3eig import Sym3Eig
from numpy.testing import assert_almost_equal

from .utils import dtypes, devices, tensor

matrices = np.random.rand(100, 3, 3)
matrices = matrices + matrices.swapaxes(1, 2)

eig_val, eig_vec = np.linalg.eig(matrices)
eig_val, eig_vec = torch.from_numpy(eig_val), torch.from_numpy(eig_vec)
eig_val, argsort = eig_val.sort(dim=-1, descending=True)
eig_val = eig_val.numpy()
eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
eig_vec[:, :, 0] = torch.sign(eig_vec[:, 0, 0].view(-1, 1))*eig_vec[:, :, 0]
eig_vec[:, :, 1] = torch.sign(eig_vec[:, 0, 1].view(-1, 1))*eig_vec[:, :, 1]
eig_vec[:, :, 2] = torch.cross(eig_vec[:, :, 0], eig_vec[:, :, 1])
eig_vec = eig_vec.numpy()

tests = [{
    'matrices': matrices,
    'eig_val': eig_val,
    'eig_vec': eig_vec
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_sym3eig_forward(test, dtype, device):
    matrices = tensor(test['matrices'], dtype, device)

    op = Sym3Eig.apply
    eig_val, eig_vec = op(matrices)
    matrices = matrices.cpu()
    if eig_vec.is_cuda:
        eig_val, eig_vec = eig_val.cpu(), eig_vec.cpu()

    z = torch.zeros_like(eig_val[:, 0])
    diag = torch.stack(
        [eig_val[:, 0], z, z, z, eig_val[:, 1], z, z, z, eig_val[:, 2]], dim=1).view(-1, 3, 3)
    decomp = torch.matmul(torch.matmul(eig_vec.double(), diag), torch.transpose(eig_vec.double(), 1, 2))

    shouldbeeye = torch.matmul(eig_vec, torch.transpose(eig_vec, 1, 2))
    eye = np.eye(3).reshape(1, 3, 3)
    eye = np.repeat(eye, eig_vec.size(0), axis=0)
    # Check U*U^T = I
    assert_almost_equal(shouldbeeye.numpy(), eye, decimal=5)
    # Check X = U*S*U^T
    assert_almost_equal(decomp.numpy(), matrices, decimal=5)

    # Check Xv = sv for all (s,v)
    vec1 = eig_vec[:, :, 0]
    vec2 = eig_vec[:, :, 1]
    vec3 = eig_vec[:, :, 2]
    assert_almost_equal(torch.matmul(matrices, vec1.view(-1, 3, 1)).squeeze().numpy(),
                        (vec1*eig_val[:, 0].view(-1, 1)).squeeze().numpy(), decimal=5)
    assert_almost_equal(torch.matmul(matrices, vec2.view(-1, 3, 1)).squeeze().numpy(),
                        (vec2 * eig_val[:, 1].view(-1, 1)).squeeze().numpy(), decimal=5)
    assert_almost_equal(torch.matmul(matrices, vec3.view(-1, 3, 1)).squeeze().numpy(),
                        (vec3 * eig_val[:, 2].view(-1, 1)).squeeze().numpy(), decimal=5)

    # Comparison with numpy.linalg.eig
    # Disambiguate and sort eig_vecs for comparison
    indices = torch.argsort(eig_val, dim=1, descending=True)
    eig_vec = eig_vec.gather(2, indices.view(-1, 1, 3).expand_as(eig_vec))
    eig_val = eig_val.gather(1, indices.view(-1, 3))
    
    eig_vec[:, :, 0] = torch.sign(eig_vec[:, 0, 0].view(-1, 1))*eig_vec[:, :, 0]
    eig_vec[:, :, 1] = torch.sign(eig_vec[:, 0, 1].view(-1, 1))*eig_vec[:, :, 1]
    eig_vec[:, :, 2] = torch.cross(eig_vec[:, :, 0], eig_vec[:, :, 1])
    assert_almost_equal(eig_val.numpy(), test['eig_val'], decimal=5)
    assert_almost_equal(eig_vec.numpy(), test['eig_vec'], decimal=5)


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_sym3eig_backward(test, dtype, device):
    matrices_t = tensor(matrices, dtype, device)
    matrices_t.requires_grad_()
    assert gradcheck(Sym3Eig.apply, matrices_t, eps=1e-6, atol=1e-4) is True
