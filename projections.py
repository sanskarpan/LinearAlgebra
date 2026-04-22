"""
Projections and Orthogonalization Module
=========================================

Implements projection operations and orthogonalization algorithms:
- Vector projections
- Gram-Schmidt orthogonalization
- QR factorization via Gram-Schmidt
- Orthogonal complement
- Best approximation problems

These operations are fundamental for:
- Least squares fitting
- Signal processing
- Computer graphics
- Dimensionality reduction
"""

from typing import List, Optional
from .vector import Vector
from .matrix import Matrix


def project_vector_onto_vector(v: Vector, u: Vector) -> Vector:
    """
    Project vector v onto vector u.

    The projection is the component of v in the direction of u.

    Formula: proj_u(v) = (v·u / u·u) * u

    Args:
        v: Vector to project
        u: Vector to project onto

    Returns:
        Projection of v onto u

    Raises:
        ValueError: If u is zero vector
    """
    if u.magnitude() < 1e-10:
        raise ValueError("Cannot project onto zero vector")

    scalar_proj = v.dot(u) / u.dot(u)
    return scalar_proj * u


def project_vector_onto_subspace(v: Vector, basis: List[Vector]) -> Vector:
    """
    Project vector v onto a subspace defined by a basis.

    The projection is the closest point in the subspace to v.

    Args:
        v: Vector to project
        basis: List of basis vectors for the subspace

    Returns:
        Projection of v onto the subspace

    Raises:
        ValueError: If basis is empty or contains zero vectors
    """
    if not basis:
        raise ValueError("Basis cannot be empty")

    # Orthogonalize the basis first
    orthonormal_basis = gram_schmidt(basis, normalize=True)

    # Project onto each basis vector and sum
    projection = Vector.zero(v.dim)
    for u in orthonormal_basis:
        projection = projection + project_vector_onto_vector(v, u)

    return projection


def orthogonal_complement(v: Vector, u: Vector) -> Vector:
    """
    Compute the component of v orthogonal to u.

    This is the rejection of v from u.

    Formula: v - proj_u(v)

    Args:
        v: Vector to decompose
        u: Vector to be orthogonal to

    Returns:
        Component of v orthogonal to u
    """
    return v - project_vector_onto_vector(v, u)


def gram_schmidt(vectors: List[Vector], normalize: bool = True) -> List[Vector]:
    """
    Gram-Schmidt orthogonalization process.

    Converts a set of linearly independent vectors into an orthogonal
    (or orthonormal) set spanning the same subspace.

    This is fundamental for:
    - QR decomposition
    - Constructing orthonormal bases
    - Solving least squares problems
    - Numerical stability

    Args:
        vectors: List of linearly independent vectors
        normalize: If True, return orthonormal vectors; else orthogonal

    Returns:
        List of orthogonal (or orthonormal) vectors

    Raises:
        ValueError: If vectors are linearly dependent
    """
    if not vectors:
        return []

    orthogonal_vectors = []

    for v in vectors:
        # Start with current vector
        u = v.copy()

        # Subtract projections onto all previous orthogonal vectors
        for basis_vec in orthogonal_vectors:
            u = u - project_vector_onto_vector(v, basis_vec)

        # Check if result is zero (linear dependence)
        if u.magnitude() < 1e-10:
            raise ValueError("Vectors are linearly dependent")

        # Normalize if requested
        if normalize:
            u = u.normalize()

        orthogonal_vectors.append(u)

    return orthogonal_vectors


def modified_gram_schmidt(vectors: List[Vector], normalize: bool = True) -> List[Vector]:
    """
    Modified Gram-Schmidt orthogonalization (more numerically stable).

    The modified version orthogonalizes against the most recent
    orthogonal vector in each iteration, reducing numerical errors.

    Args:
        vectors: List of linearly independent vectors
        normalize: If True, return orthonormal vectors

    Returns:
        List of orthogonal (or orthonormal) vectors

    Raises:
        ValueError: If vectors are linearly dependent
    """
    if not vectors:
        return []

    # Make copies to avoid modifying originals
    working_vectors = [v.copy() for v in vectors]
    orthogonal_vectors = []

    for i in range(len(working_vectors)):
        u = working_vectors[i]

        # Orthogonalize against all previously computed vectors
        for basis_vec in orthogonal_vectors:
            # Project and subtract immediately
            proj = u.dot(basis_vec) * basis_vec
            u = u - proj

        # Check for linear dependence
        if u.magnitude() < 1e-10:
            raise ValueError("Vectors are linearly dependent")

        # Normalize if requested
        if normalize:
            u = u.normalize()

        orthogonal_vectors.append(u)

    return orthogonal_vectors


def qr_gram_schmidt(A: Matrix) -> tuple[Matrix, Matrix]:
    """
    QR decomposition using Gram-Schmidt orthogonalization.

    Decomposes A into Q (orthogonal) and R (upper triangular).

    Args:
        A: Matrix to decompose (m x n with m >= n)

    Returns:
        Tuple (Q, R) where A = QR

    Raises:
        ValueError: If matrix has linearly dependent columns
    """
    m, n = A.rows, A.cols

    # Extract columns as vectors
    columns = [A.get_column(j) for j in range(n)]

    # Apply Gram-Schmidt
    Q_vectors = []
    R_data = [[0.0] * n for _ in range(n)]

    for j in range(n):
        v = columns[j]

        # Orthogonalize against previous Q vectors
        for i in range(j):
            R_data[i][j] = Q_vectors[i].dot(v)
            v = v - R_data[i][j] * Q_vectors[i]

        # Compute R[j,j]
        R_data[j][j] = v.magnitude()
        if R_data[j][j] < 1e-10:
            raise ValueError("Matrix has linearly dependent columns")

        # Normalize to get Q vector
        q = v.normalize()
        Q_vectors.append(q)

    Q = Matrix.from_vectors(Q_vectors, as_rows=False)
    R = Matrix(R_data)

    return Q, R


def least_squares(A: Matrix, b: Vector) -> Vector:
    """
    Solve least squares problem: minimize ||Ax - b||^2

    Finds the best-fit solution when the system Ax = b is overdetermined.

    Used in:
    - Linear regression
    - Curve fitting
    - Data fitting
    - Approximation problems

    Args:
        A: Coefficient matrix (m x n with m >= n)
        b: Target vector

    Returns:
        Least squares solution x

    Method:
        Uses the normal equations: A^T A x = A^T b
        Or QR decomposition for better numerical stability
    """
    if A.rows < A.cols:
        raise ValueError("Matrix must have at least as many rows as columns")

    # Use QR decomposition for numerical stability
    try:
        Q, R = qr_gram_schmidt(A)

        # Solve R x = Q^T b
        Qt_b = Q.T @ b

        # Back-substitution on R x = Qt_b
        x = _back_substitution(R, Qt_b)

        return x

    except ValueError:
        # Fallback to normal equations
        AtA = A.T @ A
        Atb = A.T @ b

        # Solve using Gaussian elimination
        return _solve_system(AtA, Atb)


def _back_substitution(R: Matrix, b: Vector) -> Vector:
    """
    Solve upper triangular system Rx = b using back-substitution.

    Args:
        R: Upper triangular matrix
        b: Right-hand side vector

    Returns:
        Solution vector x
    """
    n = R.rows
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        if abs(R.data[i][i]) < 1e-10:
            raise ValueError("Singular matrix")

        sum_val = sum(R.data[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_val) / R.data[i][i]

    return Vector(x)


def _solve_system(A: Matrix, b: Vector) -> Vector:
    """
    Solve linear system Ax = b using Gaussian elimination.

    Args:
        A: Coefficient matrix (square)
        b: Right-hand side vector

    Returns:
        Solution vector x
    """
    if not A.is_square():
        raise ValueError("Matrix must be square")

    n = A.rows

    # Augment matrix
    augmented = [A.data[i] + [b[i]] for i in range(n)]
    aug_matrix = Matrix(augmented)

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug_matrix.data[k][i]) > abs(aug_matrix.data[max_row][i]):
                max_row = k

        # Swap rows
        aug_matrix.data[i], aug_matrix.data[max_row] = \
            aug_matrix.data[max_row], aug_matrix.data[i]

        # Check for singularity
        if abs(aug_matrix.data[i][i]) < 1e-10:
            raise ValueError("Matrix is singular")

        # Eliminate below
        for k in range(i + 1, n):
            factor = aug_matrix.data[k][i] / aug_matrix.data[i][i]
            for j in range(i, n + 1):
                aug_matrix.data[k][j] -= factor * aug_matrix.data[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = sum(aug_matrix.data[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug_matrix.data[i][n] - sum_val) / aug_matrix.data[i][i]

    return Vector(x)


def orthogonal_basis(vectors: List[Vector]) -> List[Vector]:
    """
    Compute an orthogonal basis from a set of vectors.

    Alias for gram_schmidt with normalize=False.

    Args:
        vectors: List of linearly independent vectors

    Returns:
        Orthogonal basis for the same subspace
    """
    return gram_schmidt(vectors, normalize=False)


def orthonormal_basis(vectors: List[Vector]) -> List[Vector]:
    """
    Compute an orthonormal basis from a set of vectors.

    Alias for gram_schmidt with normalize=True.

    Args:
        vectors: List of linearly independent vectors

    Returns:
        Orthonormal basis for the same subspace
    """
    return gram_schmidt(vectors, normalize=True)


def projection_matrix(basis: List[Vector]) -> Matrix:
    """
    Compute the projection matrix onto a subspace.

    The projection matrix P satisfies:
    - P @ v = projection of v onto subspace
    - P @ P = P (idempotent)
    - P^T = P (symmetric)

    Args:
        basis: Basis vectors for the subspace

    Returns:
        Projection matrix

    Formula:
        If Q has orthonormal basis vectors as columns,
        then P = Q @ Q^T
    """
    # Orthonormalize the basis
    orthonormal = gram_schmidt(basis, normalize=True)

    # Create matrix with basis vectors as columns
    Q = Matrix.from_vectors(orthonormal, as_rows=False)

    # P = Q @ Q^T
    P = Q @ Q.T

    return P


def best_approximation(v: Vector, subspace_basis: List[Vector]) -> Vector:
    """
    Find the best approximation of v in a subspace.

    The best approximation is the projection of v onto the subspace,
    which minimizes the distance ||v - approximation||.

    Args:
        v: Vector to approximate
        subspace_basis: Basis vectors for the subspace

    Returns:
        Best approximation of v in the subspace
    """
    return project_vector_onto_subspace(v, subspace_basis)


def distance_to_subspace(v: Vector, subspace_basis: List[Vector]) -> float:
    """
    Compute the distance from a vector to a subspace.

    Args:
        v: Vector
        subspace_basis: Basis vectors for the subspace

    Returns:
        Minimum distance from v to the subspace
    """
    projection = project_vector_onto_subspace(v, subspace_basis)
    difference = v - projection
    return difference.magnitude()


def angle_between_subspaces(basis1: List[Vector], basis2: List[Vector]) -> float:
    """
    Compute the principal angle between two subspaces.

    Args:
        basis1: Basis for first subspace
        basis2: Basis for second subspace

    Returns:
        Principal angle in radians

    Note:
        This computes the smallest principal angle.
    """
    # Orthonormalize both bases
    Q1 = gram_schmidt(basis1, normalize=True)
    Q2 = gram_schmidt(basis2, normalize=True)

    # Compute the matrix Q1^T @ Q2
    m1 = Matrix.from_vectors(Q1, as_rows=False)
    m2 = Matrix.from_vectors(Q2, as_rows=False)

    product = m1.T @ m2

    # Find maximum singular value (or eigenvalue of product^T @ product)
    max_cos = 0.0
    for i in range(min(len(Q1), len(Q2))):
        for j in range(min(len(Q1), len(Q2))):
            cos_val = abs(product.data[i][j]) if i < product.rows and j < product.cols else 0.0
            max_cos = max(max_cos, cos_val)

    # Clamp to [0, 1] to handle numerical errors
    max_cos = min(1.0, max_cos)

    # Principal angle
    import math
    return math.acos(max_cos)


def is_orthogonal_set(vectors: List[Vector], tolerance: float = 1e-10) -> bool:
    """
    Check if a set of vectors is orthogonal.

    Args:
        vectors: List of vectors to check
        tolerance: Numerical tolerance

    Returns:
        True if all pairs are orthogonal
    """
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if abs(vectors[i].dot(vectors[j])) > tolerance:
                return False
    return True


def is_orthonormal_set(vectors: List[Vector], tolerance: float = 1e-10) -> bool:
    """
    Check if a set of vectors is orthonormal.

    Args:
        vectors: List of vectors to check
        tolerance: Numerical tolerance

    Returns:
        True if all vectors are unit length and mutually orthogonal
    """
    # Check unit length
    for v in vectors:
        if abs(v.magnitude() - 1.0) > tolerance:
            return False

    # Check orthogonality
    return is_orthogonal_set(vectors, tolerance)
