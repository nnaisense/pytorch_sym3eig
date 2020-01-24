import time
import numpy as np
import torch
from torch_sym3eig import Sym3Eig
import sys

num_matrices = 100000

if len(sys.argv)>1:
    num_matrices = int(sys.argv[1])

matrices = np.random.rand(num_matrices, 3, 3)
matrices = matrices + matrices.swapaxes(1, 2)

matrices = torch.from_numpy(matrices)
matrices.requires_grad_()
print('Computing Forward and Backward for {} matrices'.format(num_matrices))
starttime = time.process_time()
eig_val, eig_vec = Sym3Eig.apply(matrices)
runtime = (time.process_time() - starttime)*1000.0
print('Forward CPU: {} ms'.format(runtime))

starttime = time.process_time()
grad = torch.cat([eig_val.view(-1,3,1), eig_vec], dim=2).sum().backward()
runtime = (time.process_time() - starttime)*1000.0
print('Backward CPU: {} ms'.format(runtime))

if torch.cuda.is_available():
    matrices = np.random.rand(num_matrices, 3, 3)
    matrices = matrices + matrices.swapaxes(1, 2)

    matrices = torch.from_numpy(matrices).cuda()
    matrices.requires_grad_()
    starttime = time.process_time()
    eig_val, eig_vec = Sym3Eig.apply(matrices)
    torch.cuda.synchronize()
    runtime = (time.process_time() - starttime)*1000.0
    print('Forward GPU: {} ms'.format(runtime))

    starttime = time.process_time()
    grad = torch.cat([eig_val.view(-1,3,1), eig_vec], dim=2).sum()
    grad.backward()
    torch.cuda.synchronize()
    runtime = (time.process_time() - starttime)*1000.0
    print('Backward GPU: {} ms'.format(runtime))
