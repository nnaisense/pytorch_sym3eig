# Pytorch extension: Batch-wise eigencomputation for symmetric 3x3 matrices


--------------------------------------------------------------------------------

The operator works on 32 and 64 bit floating point data types and is implemented both for CPU and GPU with custom kernels.
Implementations include forward and backward steps.

The full code for our work [Deep Iterative Surface Normal Estimation](https://arxiv.org/abs/1904.07172) can be found here: https://github.com/nnaisense/deep-iterative-surface-normal-estimation.

## Installation

Ensure that at least PyTorch >= 1.0.0 is installed, checkout repository and run:

```
python setup.py install
```

## Running tests

```
python setup.py test
```
Requires pytest and numpy.

## Usage

```python
import torch
from torch_sym3eig import Sym3Eig

matrices = torch.rand(100, 3, 3)
# Matrices need to be symmetric
matrices = (matrices + torch.transpose(matrices, 1, 2))/2.0

eig_vals, eig_vecs = Sym3Eig.apply(matrices)
```

## Benchmarks

Runtimes for 100k matrices on an i7-7700K @ 4.20GHz and a GTX Titan X, respectively:

|          | CPU      | GPU     |
|----------|----------|---------|
| Forward  | 23.59 ms | 1.93 ms |
| Backward | 20.47 ms | 1.31 ms |


Running benchmarks:
```
python test/benchmark.py 100000
```

### Parameters

* **x** *(Tensor)* - Input matrices (symmetric 3x3) `(number_of_matrices x 3 x 3)`.


### Returns

* **eig_vals** *(Tensor)* - Eigenvalues of the input matrices in no pre-defined order`(number_of_matrices x 3)`.
* **eig_vecs** *(Tensor)* - Corresponding eigenvectors in columns`(number_of_matrices x 3 x 3)`.

### Information
The module was initially created for and used in our [Deep Iterative Surface Normal Estimation](https://arxiv.org/abs/1904.07172) paper, which can be cited as reference:
```
@InProceedings{Lenssen:2020,
	author = {Lenssen, Jan Eric and Osendorfer, Christian and Masci, Jonathan},
	title = {Deep Iterative Surface Normal Estimation},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2020}
}
```


The implementation follows the derivations from:
* Mike B. Giles. Collected matrix derivative results for forward and reverse mode algorithmic differentiation. In Advances in Automatic Differentiation, pages 35–44. Springer Berlin Heidelberg, 2008, and
* https://en.wikipedia.org/wiki/Eigenvalue_algorithm
