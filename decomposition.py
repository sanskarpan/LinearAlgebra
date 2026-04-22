"""
Matrix Decomposition Module
============================

Implements advanced matrix decompositions essential for AI/ML/DL:
- QR Decomposition (Gram-Schmidt)
- Cholesky Decomposition
- Singular Value Decomposition (SVD)
- Eigenvalue Decomposition

These decompositions are fundamental for:
- Solving least squares problems
- Principal Component Analysis (PCA)
- Dimensionality reduction
- Numerical stability in computations
"""

from typing import Tuple, Optional
import math
from .vector import Vector
from .matrix import Matrix


def qr_decomposition(A: Matrix, method: str = 'gram_schmidt') -> Tuple[Matrix, Matrix]:
    """
    QR Decomposition: A = QR where Q is orthogonal and R is upper triangular.

    Used in:
    - Solving least squares problems
    - Eigenvalue algorithms
    - Orthogonalization

    Args:
        A: Matrix to decompose
        method: 'gram_schmidt' or 'householder'

    Returns:
        Tuple (Q, R) where Q is orthogonal and R is upper triangular

    Raises:
        ValueError: If matrix has linearly dependent columns
    """
    if method == 'gram_schmidt':
        return _qr_gram_schmidt(A)
    elif method == 'householder':
        return _qr_householder(A)
    else:
        raise ValueError(f"Unknown method: {method}")


def _qr_gram_schmidt(A: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Gram-Schmidt orthogonalization."""
    m, n = A.rows, A.cols

    # Initialize Q and R
    Q_vectors = []
    R_data = [[0.0] * n for _ in range(n)]

    for j in range(n):
        # Get column j
        v = A.get_column(j)

        # Orthogonalize against previous Q vectors
        for i in range(j):
            R_data[i][j] = Q_vectors[i].dot(v)
            v = v - R_data[i][j] * Q_vectors[i]

        # Normalize
        R_data[j][j] = v.magnitude()
        if R_data[j][j] < 1e-10:
            raise ValueError("Matrix has linearly dependent columns")

        q = v.normalize()
        Q_vectors.append(q)

    # Convert Q_vectors to matrix
    Q = Matrix.from_vectors(Q_vectors, as_rows=False)
    R = Matrix(R_data)

    return Q, R


def _qr_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    """QR decomposition using Householder reflections (more numerically stable)."""
    m, n = A.rows, A.cols
    Q = Matrix.identity(m)
    R = A.copy()

    for k in range(min(m - 1, n)):
        # Extract column below diagonal
        x = [R.data[i][k] for i in range(k, m)]
        x_vec = Vector(x)

        # Compute Householder vector
        alpha = -math.copysign(x_vec.magnitude(), x[0])
        e1 = Vector.zero(len(x))
        e1[0] = 1.0

        u = x_vec - alpha * e1
        u_norm = u.magnitude()

        if u_norm < 1e-10:
            continue

        v = u.normalize()

        # Apply Householder reflection to R
        for j in range(k, n):
            col = Vector([R.data[i][j] for i in range(k, m)])
            reflection = col - 2 * v.dot(col) * v
            for i in range(k, m):
                R.data[i][j] = reflection[i - k]

        # Update Q
        for i in range(m):
            row = Vector([Q.data[i][j] for j in range(k, m)])
            reflection = row - 2 * row.dot(v) * v
            for j in range(k, m):
                Q.data[i][j] = reflection[j - k]

    return Q, R


def cholesky_decomposition(A: Matrix) -> Matrix:
    """
    Cholesky Decomposition: A = LL^T where L is lower triangular.

    Only works for symmetric positive definite matrices.

    Used in:
    - Solving symmetric positive definite systems
    - Sampling from multivariate Gaussians
    - Optimization algorithms

    Args:
        A: Symmetric positive definite matrix

    Returns:
        Lower triangular matrix L

    Raises:
        ValueError: If matrix is not symmetric positive definite
    """
    if not A.is_square():
        raise ValueError("Cholesky decomposition requires a square matrix")

    if not A.is_symmetric():
        raise ValueError("Cholesky decomposition requires a symmetric matrix")

    n = A.rows
    L_data = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal element
                sum_sq = sum(L_data[i][k] ** 2 for k in range(j))
                val = A.data[i][i] - sum_sq

                if val <= 1e-10:
                    raise ValueError(
                        "Matrix is not positive definite (encountered non-positive diagonal)"
                    )

                L_data[i][j] = math.sqrt(val)
            else:
                # Off-diagonal element
                sum_prod = sum(L_data[i][k] * L_data[j][k] for k in range(j))
                L_data[i][j] = (A.data[i][j] - sum_prod) / L_data[j][j]

    return Matrix(L_data)


def svd(A: Matrix, max_iterations: int = 1000) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Singular Value Decomposition: A = U Σ V^T

    This is one of the most important decompositions in ML/DL.

    Used in:
    - Principal Component Analysis (PCA)
    - Dimensionality reduction
    - Recommender systems
    - Image compression
    - Pseudoinverse computation

    Args:
        A: Matrix to decompose
        max_iterations: Maximum iterations for iterative algorithm

    Returns:
        Tuple (U, Sigma, V) where:
        - U: Left singular vectors (m x m orthogonal)
        - Sigma: Diagonal matrix of singular values (m x n)
        - V: Right singular vectors (n x n orthogonal)

    Note:
        This implementation uses the Jacobi method for small matrices
        and is educational. For production use, consider NumPy's SVD.
    """
    m, n = A.rows, A.cols

    # Compute A^T A for right singular vectors
    AtA = A.T @ A

    # Find eigenvalues and eigenvectors of A^T A
    eigenvalues, V = _symmetric_eigendecomposition(AtA, max_iterations)

    # Sort by eigenvalues (descending)
    pairs = sorted(zip(eigenvalues, range(len(eigenvalues))), reverse=True)
    sorted_indices = [idx for _, idx in pairs]

    # Reorder V columns
    V_sorted = Matrix.from_vectors(
        [V.get_column(i) for i in sorted_indices],
        as_rows=False
    )

    # Compute singular values (sqrt of eigenvalues)
    singular_values = [math.sqrt(max(0, eigenvalues[i])) for i in sorted_indices]

    # Compute U = A V Σ^(-1)
    U_vectors = []
    for i in range(min(m, n)):
        if singular_values[i] > 1e-10:
            u = A @ V_sorted.get_column(i)
            u = u / singular_values[i]
            U_vectors.append(u)
        else:
            # For zero singular values, use arbitrary orthogonal vector
            U_vectors.append(Vector.basis(m, i))

    # Extend U to be m x m orthogonal if needed
    if len(U_vectors) < m:
        # Add orthogonal vectors using Gram-Schmidt
        for i in range(len(U_vectors), m):
            # Start with basis vector
            v = Vector.basis(m, i)
            # Orthogonalize against existing U vectors
            for u in U_vectors:
                v = v - v.dot(u) * u
            if v.magnitude() > 1e-10:
                U_vectors.append(v.normalize())
            else:
                # Try different basis vectors
                for j in range(m):
                    v = Vector.basis(m, j)
                    for u in U_vectors:
                        v = v - v.dot(u) * u
                    if v.magnitude() > 1e-10:
                        U_vectors.append(v.normalize())
                        break

    U = Matrix.from_vectors(U_vectors, as_rows=False)

    # Create Sigma matrix
    Sigma_data = [[0.0] * n for _ in range(m)]
    for i in range(min(m, n)):
        if i < len(singular_values):
            Sigma_data[i][i] = singular_values[i]

    Sigma = Matrix(Sigma_data)

    return U, Sigma, V_sorted


def _symmetric_eigendecomposition(A: Matrix, max_iterations: int = 1000) -> Tuple[list, Matrix]:
    """
    Compute eigenvalues and eigenvectors of symmetric matrix using Jacobi method.

    Args:
        A: Symmetric matrix
        max_iterations: Maximum iterations

    Returns:
        Tuple of (eigenvalues, eigenvectors_matrix)
    """
    if not A.is_symmetric():
        raise ValueError("This method only works for symmetric matrices")

    n = A.rows
    V = Matrix.identity(n)
    A_current = A.copy()

    for iteration in range(max_iterations):
        # Find largest off-diagonal element
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_current.data[i][j]) > max_val:
                    max_val = abs(A_current.data[i][j])
                    p, q = i, j

        # Check convergence
        if max_val < 1e-10:
            break

        # Compute rotation angle
        if abs(A_current.data[p][p] - A_current.data[q][q]) < 1e-10:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(
                2 * A_current.data[p][q],
                A_current.data[p][p] - A_current.data[q][q]
            )

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply Jacobi rotation
        _apply_jacobi_rotation(A_current, V, p, q, c, s)

    # Extract eigenvalues from diagonal
    eigenvalues = [A_current.data[i][i] for i in range(n)]

    return eigenvalues, V


def _apply_jacobi_rotation(A: Matrix, V: Matrix, p: int, q: int, c: float, s: float):
    """Apply Jacobi rotation to matrices A and V."""
    n = A.rows

    # Update A
    for i in range(n):
        if i != p and i != q:
            a_ip = c * A.data[i][p] - s * A.data[i][q]
            a_iq = s * A.data[i][p] + c * A.data[i][q]
            A.data[i][p] = A.data[p][i] = a_ip
            A.data[i][q] = A.data[q][i] = a_iq

    a_pp = c * c * A.data[p][p] - 2 * s * c * A.data[p][q] + s * s * A.data[q][q]
    a_qq = s * s * A.data[p][p] + 2 * s * c * A.data[p][q] + c * c * A.data[q][q]
    a_pq = 0.0

    A.data[p][p] = a_pp
    A.data[q][q] = a_qq
    A.data[p][q] = A.data[q][p] = a_pq

    # Update V
    for i in range(n):
        v_ip = c * V.data[i][p] - s * V.data[i][q]
        v_iq = s * V.data[i][p] + c * V.data[i][q]
        V.data[i][p] = v_ip
        V.data[i][q] = v_iq


def eigendecomposition(A: Matrix, max_iterations: int = 1000) -> Tuple[list, Matrix]:
    """
    Compute eigenvalues and eigenvectors.

    For symmetric matrices, uses Jacobi method.
    For non-symmetric matrices, uses QR algorithm.

    Eigendecomposition is used in:
    - PCA (Principal Component Analysis)
    - Spectral clustering
    - Markov chains
    - Stability analysis

    Args:
        A: Square matrix
        max_iterations: Maximum iterations

    Returns:
        Tuple of (eigenvalues, eigenvectors_matrix)

    Raises:
        ValueError: If matrix is not square
    """
    if not A.is_square():
        raise ValueError("Eigendecomposition requires a square matrix")

    if A.is_symmetric():
        return _symmetric_eigendecomposition(A, max_iterations)
    else:
        return _qr_algorithm(A, max_iterations)


def _qr_algorithm(A: Matrix, max_iterations: int = 1000) -> Tuple[list, Matrix]:
    """
    QR algorithm for eigenvalue computation.

    Args:
        A: Square matrix
        max_iterations: Maximum iterations

    Returns:
        Tuple of (eigenvalues, eigenvectors_matrix)
    """
    n = A.rows
    A_k = A.copy()
    Q_total = Matrix.identity(n)

    for _ in range(max_iterations):
        # QR decomposition
        try:
            Q, R = qr_decomposition(A_k)
        except ValueError:
            # If decomposition fails, return current estimates
            break

        # Update A_k = RQ
        A_k = R @ Q

        # Accumulate eigenvectors
        Q_total = Q_total @ Q

        # Check convergence (off-diagonal elements should approach zero)
        off_diag_sum = sum(
            abs(A_k.data[i][j])
            for i in range(n)
            for j in range(n)
            if i != j
        )

        if off_diag_sum < 1e-10:
            break

    # Extract eigenvalues from diagonal
    eigenvalues = [A_k.data[i][i] for i in range(n)]

    return eigenvalues, Q_total


def pca(X: Matrix, n_components: Optional[int] = None) -> Tuple[Matrix, Matrix, list]:
    """
    Principal Component Analysis using SVD.

    PCA is fundamental in:
    - Dimensionality reduction
    - Feature extraction
    - Data visualization
    - Noise reduction

    Args:
        X: Data matrix (rows are samples, columns are features)
        n_components: Number of components to keep (None = all)

    Returns:
        Tuple of:
        - Transformed data (projected onto principal components)
        - Principal components (eigenvectors)
        - Explained variance (eigenvalues)
    """
    m, n = X.rows, X.cols

    # Center the data (subtract mean)
    means = [sum(X.data[i][j] for i in range(m)) / m for j in range(n)]
    X_centered = Matrix([
        [X.data[i][j] - means[j] for j in range(n)]
        for i in range(m)
    ])

    # Compute SVD
    U, Sigma, V = svd(X_centered)

    # Extract singular values
    singular_values = [Sigma.data[i][i] for i in range(min(m, n))]

    # Compute explained variance
    explained_variance = [s ** 2 / (m - 1) for s in singular_values]

    # Determine number of components
    if n_components is None:
        n_components = len(explained_variance)
    n_components = min(n_components, len(explained_variance))

    # Transform data
    # X_transformed = U @ Sigma (first n_components columns)
    X_transformed_data = [
        [U.data[i][j] * singular_values[j] for j in range(n_components)]
        for i in range(m)
    ]
    X_transformed = Matrix(X_transformed_data)

    # Principal components (first n_components columns of V)
    components = Matrix.from_vectors(
        [V.get_column(i) for i in range(n_components)],
        as_rows=True
    )

    return X_transformed, components, explained_variance[:n_components]


def moore_penrose_inverse(A: Matrix) -> Matrix:
    """
    Compute the Moore-Penrose pseudoinverse using SVD.

    The pseudoinverse is used for:
    - Solving overdetermined systems (least squares)
    - Solving underdetermined systems (minimum norm)
    - Computing optimal solutions

    Args:
        A: Matrix to invert

    Returns:
        Pseudoinverse A^+

    Note:
        For invertible square matrices, A^+ = A^(-1)
        For rectangular matrices, provides best-fit solutions
    """
    # Compute SVD: A = U Σ V^T
    U, Sigma, V = svd(A)

    m, n = A.rows, A.cols

    # Compute Σ^+ (pseudoinverse of Sigma)
    Sigma_plus_data = [[0.0] * m for _ in range(n)]
    for i in range(min(m, n)):
        sigma_i = Sigma.data[i][i]
        if abs(sigma_i) > 1e-10:
            Sigma_plus_data[i][i] = 1.0 / sigma_i

    Sigma_plus = Matrix(Sigma_plus_data)

    # A^+ = V Σ^+ U^T
    return V @ Sigma_plus @ U.T
