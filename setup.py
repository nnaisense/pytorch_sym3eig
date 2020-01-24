from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

ext_modules = [
    CppExtension('sym3eig_cpu', ['cpu/sym3eig.cpp']),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('sym3eig_cuda',
                      ['cuda/sym3eig.cpp', 'cuda/sym3eig_kernel.cu'])
    ]

__version__ = '1.0.0'
#url = 'https://github.com/mrjel/pytorch_sym3eig'

install_requires = ['torchvision']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'numpy']

setup(
    name='torch_sym3eig',
    version=__version__,
    description='Implementation of batch-wise eigenvector/value computation for symmetric 3x3 matrices'
    'Batchwise symmetric 3x3 eigencomputation in PyTorch',
    author='Jan Eric Lenssen',
    author_email='janeric.lenssen@tu-dortmund.de',
    #url=url,
    #download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch', 'eigenvector', 'eigenvalue', 'batchwise-sym3eig', 'geometric-deep-learning', 'neural-networks'
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
