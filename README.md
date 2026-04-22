# LinearAlgebra

A pure-Python linear algebra library built from scratch, designed for understanding the mathematical foundations of AI, Machine Learning, and Deep Learning.

> **Educational project** — Phase 0 of an ML research roadmap. Pure Python, zero external dependencies.

## Modules

| Module | Description |
|---|---|
| `vector.py` | Vector class — norms, dot/cross/outer products, projections, cosine similarity |
| `matrix.py` | Matrix class — arithmetic, inverse, determinant, rank, LU decomposition |
| `tensor.py` | Tensor class — activations (ReLU, sigmoid, softmax), loss functions, dropout |
| `decomposition.py` | QR, Cholesky, SVD, eigendecomposition, PCA, pseudoinverse |
| `transformations.py` | 2D/3D rotations, reflections, shearing, Householder, Givens |
| `projections.py` | Gram-Schmidt, modified Gram-Schmidt, least squares, projection matrices |

## Quick Start

```python
from LinearAlgebra import Vector, Matrix, Tensor
from LinearAlgebra import decomposition

# Vectors
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
print(v1.dot(v2))           # 32.0
print(v1.cosine_similarity(v2))

# Matrices
A = Matrix([[1, 2], [3, 4]])
print(A.determinant())      # -2.0
print(A.inverse())

# Decompositions
Q, R = decomposition.qr_decomposition(A)
U, Sigma, V = decomposition.svd(A)

# Tensors / Deep Learning
t = Tensor([[1.0, 2.0, 3.0]])
print(t.relu())
print(t.softmax())
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Version

`0.1.0` — active development, educational use.
