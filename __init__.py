"""
Linear Algebra Library for AI/ML/DL
====================================

A comprehensive linear algebra library built from scratch in Python,
designed specifically for understanding the mathematical foundations
of AI, Machine Learning, and Deep Learning.

Modules:
--------
- vector: Vector operations and computations
- matrix: Matrix operations and computations
- decomposition: Matrix decompositions (LU, QR, Cholesky, SVD)
- eigenvalues: Eigenvalue and eigenvector computations
- transformations: Linear transformations
- projections: Projections and orthogonalization
- tensor: Tensor operations for deep learning
"""

from .vector import Vector
from .matrix import Matrix
from .tensor import Tensor
from . import decomposition
from . import transformations
from . import projections

__version__ = "0.1.0"
__all__ = [
    "Vector",
    "Matrix",
    "Tensor",
    "decomposition",
    "transformations",
    "projections"
]
