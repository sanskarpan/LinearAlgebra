"""
Matrix Decompositions for AI/ML - Tutorial
===========================================

Demonstrates advanced matrix decompositions used in machine learning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from LinearAlgebra import Matrix, Vector, decomposition


def lu_decomposition_demo():
    """Demonstrate LU decomposition."""
    print("=" * 60)
    print("LU DECOMPOSITION")
    print("=" * 60)

    print("\nLU decomposition: PA = LU")
    print("Used for solving linear systems efficiently\n")

    A = Matrix([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    print(f"Matrix A:\n{A}")

    L, U, P = A.lu_decomposition()

    print(f"\nLower triangular L:\n{L}")
    print(f"\nUpper triangular U:\n{U}")
    print(f"\nPermutation vector P: {P}")


def qr_decomposition_demo():
    """Demonstrate QR decomposition."""
    print("\n" + "=" * 60)
    print("QR DECOMPOSITION")
    print("=" * 60)

    print("\nQR decomposition: A = QR")
    print("Q is orthogonal, R is upper triangular")
    print("Used in least squares, eigenvalue algorithms\n")

    A = Matrix([[1, 2], [3, 4], [5, 6]])
    print(f"Matrix A:\n{A}")

    Q, R = decomposition.qr_decomposition(A)

    print(f"\nOrthogonal Q:\n{Q}")
    print(f"\nUpper triangular R:\n{R}")

    # Verify orthogonality
    QtQ = Q.T @ Q
    print(f"\nQ^T @ Q (should be identity):\n{QtQ}")


def svd_demo():
    """Demonstrate Singular Value Decomposition."""
    print("\n" + "=" * 60)
    print("SINGULAR VALUE DECOMPOSITION (SVD)")
    print("=" * 60)

    print("\nSVD: A = U Σ V^T")
    print("Most important decomposition in ML!")
    print("Used in: PCA, dimensionality reduction, compression\n")

    A = Matrix([[3, 1], [1, 3], [1, 1]])
    print(f"Matrix A:\n{A}")

    U, Sigma, V = decomposition.svd(A)

    print(f"\nLeft singular vectors U:\n{U}")
    print(f"\nSingular values Σ:\n{Sigma}")
    print(f"\nRight singular vectors V:\n{V}")

    # Extract singular values
    print("\nSingular values (diagonal of Σ):")
    for i in range(min(Sigma.rows, Sigma.cols)):
        print(f"  σ_{i+1} = {Sigma[i, i]:.4f}")


def pca_demo():
    """Demonstrate Principal Component Analysis."""
    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 60)

    print("\nPCA: Dimensionality reduction technique")
    print("Finds directions of maximum variance\n")

    # Create sample data (4 samples, 3 features)
    X = Matrix([
        [2.5, 2.4, 3.1],
        [0.5, 0.7, 1.2],
        [2.2, 2.9, 3.5],
        [1.9, 2.2, 2.8]
    ])

    print(f"Original data X (4 samples, 3 features):\n{X}")

    # Apply PCA, keep 2 components
    X_transformed, components, explained_var = decomposition.pca(X, n_components=2)

    print(f"\nTransformed data (4 samples, 2 components):\n{X_transformed}")
    print(f"\nPrincipal components:\n{components}")

    print("\nExplained variance by each component:")
    total_var = sum(explained_var)
    for i, var in enumerate(explained_var):
        print(f"  PC{i+1}: {var:.4f} ({100*var/total_var:.2f}%)")


def cholesky_demo():
    """Demonstrate Cholesky decomposition."""
    print("\n" + "=" * 60)
    print("CHOLESKY DECOMPOSITION")
    print("=" * 60)

    print("\nCholesky: A = LL^T")
    print("For symmetric positive definite matrices")
    print("Used in: optimization, sampling from Gaussians\n")

    # Symmetric positive definite matrix
    A = Matrix([[4, 2, 1], [2, 3, 1], [1, 1, 2]])
    print(f"Symmetric positive definite A:\n{A}")

    L = decomposition.cholesky_decomposition(A)

    print(f"\nLower triangular L:\n{L}")

    # Verify A = L L^T
    LLt = L @ L.T
    print(f"\nL @ L^T (should equal A):\n{LLt}")


def eigendecomposition_demo():
    """Demonstrate eigenvalue decomposition."""
    print("\n" + "=" * 60)
    print("EIGENVALUE DECOMPOSITION")
    print("=" * 60)

    print("\nEigendecomposition: A = QΛQ^T (for symmetric matrices)")
    print("Used in: PCA, spectral clustering, stability analysis\n")

    # Symmetric matrix
    A = Matrix([[2, 1], [1, 2]])
    print(f"Symmetric matrix A:\n{A}")

    eigenvalues, eigenvectors = decomposition.eigendecomposition(A)

    print("\nEigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {val:.4f}")

    print(f"\nEigenvector matrix:\n{eigenvectors}")


def main():
    """Run all decomposition demos."""
    lu_decomposition_demo()
    qr_decomposition_demo()
    svd_demo()
    pca_demo()
    cholesky_demo()
    eigendecomposition_demo()

    print("\n" + "=" * 60)
    print("Decompositions Tutorial Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
