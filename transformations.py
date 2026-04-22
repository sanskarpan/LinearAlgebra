"""
Linear Transformations Module
==============================

Implements various linear transformations essential for AI/ML/DL:
- Basic 2D/3D transformations (rotation, scaling, shearing, reflection)
- Projections (orthogonal, perspective)
- Affine transformations
- Coordinate system transformations

Linear transformations are fundamental in:
- Computer graphics
- Data augmentation
- Feature engineering
- Geometric deep learning
"""

from typing import Union, Tuple
import math
from .vector import Vector
from .matrix import Matrix


class LinearTransformation:
    """
    Represents a linear transformation as a matrix.

    A linear transformation T: R^n -> R^m satisfies:
    - T(u + v) = T(u) + T(v)
    - T(c * u) = c * T(u)
    """

    def __init__(self, matrix: Matrix):
        """
        Initialize a linear transformation.

        Args:
            matrix: Transformation matrix
        """
        self.matrix = matrix
        self.input_dim = matrix.cols
        self.output_dim = matrix.rows

    def __call__(self, x: Union[Vector, Matrix]) -> Union[Vector, Matrix]:
        """
        Apply the transformation to a vector or matrix.

        Args:
            x: Vector or matrix to transform

        Returns:
            Transformed vector or matrix
        """
        if isinstance(x, Vector):
            return self.matrix @ x
        elif isinstance(x, Matrix):
            return self.matrix @ x
        else:
            raise TypeError("Input must be Vector or Matrix")

    def __matmul__(self, other: 'LinearTransformation') -> 'LinearTransformation':
        """Compose two transformations."""
        return LinearTransformation(self.matrix @ other.matrix)

    def inverse(self) -> 'LinearTransformation':
        """Compute the inverse transformation."""
        return LinearTransformation(self.matrix.inverse())

    def __repr__(self) -> str:
        return f"LinearTransformation({self.output_dim}x{self.input_dim})"


# ============================================================
# 2D TRANSFORMATIONS
# ============================================================

def rotation_2d(angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a 2D rotation transformation.

    Used in:
    - Data augmentation (rotating images)
    - Feature engineering
    - Geometric transformations

    Args:
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        2D rotation transformation
    """
    if degrees:
        angle = math.radians(angle)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    matrix = Matrix([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    return LinearTransformation(matrix)


def scaling_2d(sx: float, sy: float) -> LinearTransformation:
    """
    Create a 2D scaling transformation.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction

    Returns:
        2D scaling transformation
    """
    matrix = Matrix([
        [sx, 0],
        [0, sy]
    ])
    return LinearTransformation(matrix)


def shearing_2d(shx: float, shy: float) -> LinearTransformation:
    """
    Create a 2D shearing transformation.

    Args:
        shx: Shear factor in x direction
        shy: Shear factor in y direction

    Returns:
        2D shearing transformation
    """
    matrix = Matrix([
        [1, shx],
        [shy, 1]
    ])
    return LinearTransformation(matrix)


def reflection_2d(axis: str = 'x') -> LinearTransformation:
    """
    Create a 2D reflection transformation.

    Args:
        axis: Axis to reflect across ('x', 'y', or 'origin')

    Returns:
        2D reflection transformation
    """
    if axis == 'x':
        matrix = Matrix([[1, 0], [0, -1]])
    elif axis == 'y':
        matrix = Matrix([[-1, 0], [0, 1]])
    elif axis == 'origin':
        matrix = Matrix([[-1, 0], [0, -1]])
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return LinearTransformation(matrix)


# ============================================================
# 3D TRANSFORMATIONS
# ============================================================

def rotation_3d_x(angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a 3D rotation around the x-axis.

    Args:
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        3D rotation transformation
    """
    if degrees:
        angle = math.radians(angle)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    matrix = Matrix([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

    return LinearTransformation(matrix)


def rotation_3d_y(angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a 3D rotation around the y-axis.

    Args:
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        3D rotation transformation
    """
    if degrees:
        angle = math.radians(angle)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    matrix = Matrix([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    return LinearTransformation(matrix)


def rotation_3d_z(angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a 3D rotation around the z-axis.

    Args:
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        3D rotation transformation
    """
    if degrees:
        angle = math.radians(angle)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    matrix = Matrix([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    return LinearTransformation(matrix)


def rotation_3d_axis(axis: Vector, angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a 3D rotation around an arbitrary axis (Rodrigues' rotation formula).

    Used in:
    - 3D graphics
    - Robotics
    - Spatial transformations

    Args:
        axis: Rotation axis (will be normalized)
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        3D rotation transformation
    """
    if axis.dim != 3:
        raise ValueError("Axis must be 3-dimensional")

    if degrees:
        angle = math.radians(angle)

    # Normalize axis
    k = axis.normalize()
    kx, ky, kz = k[0], k[1], k[2]

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    one_minus_cos = 1 - cos_a

    # Rodrigues' rotation formula
    matrix = Matrix([
        [
            cos_a + kx * kx * one_minus_cos,
            kx * ky * one_minus_cos - kz * sin_a,
            kx * kz * one_minus_cos + ky * sin_a
        ],
        [
            ky * kx * one_minus_cos + kz * sin_a,
            cos_a + ky * ky * one_minus_cos,
            ky * kz * one_minus_cos - kx * sin_a
        ],
        [
            kz * kx * one_minus_cos - ky * sin_a,
            kz * ky * one_minus_cos + kx * sin_a,
            cos_a + kz * kz * one_minus_cos
        ]
    ])

    return LinearTransformation(matrix)


def scaling_3d(sx: float, sy: float, sz: float) -> LinearTransformation:
    """
    Create a 3D scaling transformation.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction
        sz: Scale factor in z direction

    Returns:
        3D scaling transformation
    """
    matrix = Matrix([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ])
    return LinearTransformation(matrix)


# ============================================================
# PROJECTIONS
# ============================================================

def orthogonal_projection(subspace_basis: list[Vector]) -> LinearTransformation:
    """
    Create an orthogonal projection onto a subspace.

    Used in:
    - Dimensionality reduction
    - Least squares
    - Feature extraction

    Args:
        subspace_basis: List of orthonormal basis vectors for the subspace

    Returns:
        Projection transformation
    """
    if not subspace_basis:
        raise ValueError("Need at least one basis vector")

    # Ensure basis is orthonormal
    from .projections import gram_schmidt
    orthonormal_basis = gram_schmidt(subspace_basis, normalize=True)

    # Projection matrix: P = sum(v_i @ v_i^T)
    n = orthonormal_basis[0].dim
    P = Matrix.zero(n, n)

    for v in orthonormal_basis:
        # Outer product v @ v^T
        outer = v.outer(v)
        P = P + outer

    return LinearTransformation(P)


def projection_onto_line(direction: Vector) -> LinearTransformation:
    """
    Create a projection onto a line.

    Args:
        direction: Direction vector of the line

    Returns:
        Projection transformation
    """
    # Normalize direction
    u = direction.normalize()

    # P = (u @ u^T)
    P = u.outer(u)

    return LinearTransformation(P)


def projection_onto_plane(normal: Vector) -> LinearTransformation:
    """
    Create a projection onto a plane (3D only).

    Args:
        normal: Normal vector to the plane

    Returns:
        Projection transformation
    """
    if normal.dim != 3:
        raise ValueError("Plane projection only defined in 3D")

    # Normalize normal
    n = normal.normalize()

    # P = I - (n @ n^T)
    I = Matrix.identity(3)
    P = I - n.outer(n)

    return LinearTransformation(P)


# ============================================================
# ADVANCED TRANSFORMATIONS
# ============================================================

def change_of_basis(old_basis: list[Vector], new_basis: list[Vector]) -> LinearTransformation:
    """
    Create a change of basis transformation.

    Used in:
    - Coordinate transformations
    - Diagonalization
    - Principal axes

    Args:
        old_basis: List of old basis vectors (as columns)
        new_basis: List of new basis vectors (as columns)

    Returns:
        Change of basis transformation from old to new

    Note:
        To transform coordinates from old basis to new basis:
        [v]_new = T @ [v]_old
    """
    if len(old_basis) != len(new_basis):
        raise ValueError("Bases must have same number of vectors")

    # Create matrices from basis vectors
    P_old = Matrix.from_vectors(old_basis, as_rows=False)
    P_new = Matrix.from_vectors(new_basis, as_rows=False)

    # Change of basis: P_new^(-1) @ P_old
    T = P_new.inverse() @ P_old

    return LinearTransformation(T)


def householder_reflection(v: Vector) -> LinearTransformation:
    """
    Create a Householder reflection.

    Householder reflections are used in:
    - QR decomposition
    - Orthogonalization
    - Numerical linear algebra

    Args:
        v: Vector defining the reflection hyperplane

    Returns:
        Householder reflection transformation
    """
    # Normalize v
    u = v.normalize()

    # H = I - 2(u @ u^T)
    n = v.dim
    I = Matrix.identity(n)
    H = I - 2 * u.outer(u)

    return LinearTransformation(H)


def givens_rotation(n: int, i: int, j: int, angle: float, degrees: bool = False) -> LinearTransformation:
    """
    Create a Givens rotation matrix.

    Givens rotations are used in:
    - QR decomposition
    - Eigenvalue algorithms
    - Sparse matrix computations

    Args:
        n: Dimension of the space
        i: First coordinate index
        j: Second coordinate index
        angle: Rotation angle
        degrees: If True, angle is in degrees

    Returns:
        Givens rotation transformation
    """
    if degrees:
        angle = math.radians(angle)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Start with identity
    G = Matrix.identity(n)

    # Set rotation entries
    G.data[i][i] = cos_a
    G.data[i][j] = -sin_a
    G.data[j][i] = sin_a
    G.data[j][j] = cos_a

    return LinearTransformation(G)


# ============================================================
# UTILITIES
# ============================================================

def is_linear_transformation(T, input_dim: int, output_dim: int,
                            num_tests: int = 10, tolerance: float = 1e-10) -> bool:
    """
    Test if a function is a linear transformation.

    A function T is linear if:
    1. T(u + v) = T(u) + T(v)
    2. T(c * u) = c * T(u)

    Args:
        T: Function to test
        input_dim: Dimension of input vectors
        output_dim: Dimension of output vectors
        num_tests: Number of random tests to perform
        tolerance: Numerical tolerance

    Returns:
        True if T appears to be linear
    """
    import random

    for _ in range(num_tests):
        # Generate random vectors and scalar
        u = Vector([random.uniform(-10, 10) for _ in range(input_dim)])
        v = Vector([random.uniform(-10, 10) for _ in range(input_dim)])
        c = random.uniform(-10, 10)

        # Test additivity: T(u + v) = T(u) + T(v)
        Tu = T(u)
        Tv = T(v)
        T_u_plus_v = T(u + v)

        if (Tu + Tv - T_u_plus_v).magnitude() > tolerance:
            return False

        # Test homogeneity: T(c * u) = c * T(u)
        T_cu = T(c * u)
        c_Tu = c * Tu

        if (T_cu - c_Tu).magnitude() > tolerance:
            return False

    return True


def kernel_basis(T: Matrix, tolerance: float = 1e-10) -> list[Vector]:
    """
    Compute a basis for the kernel (null space) of a transformation.

    The kernel is the set of vectors that map to zero:
    ker(T) = {v : T(v) = 0}

    Used in:
    - Solving homogeneous systems
    - Finding conserved quantities
    - Constraint analysis

    Args:
        T: Transformation matrix
        tolerance: Numerical tolerance

    Returns:
        List of basis vectors for the kernel
    """
    # Use row reduction to find null space
    # Solve T @ x = 0

    n = T.cols
    augmented = T.copy()

    # Row reduce to RREF
    pivot_cols = []
    pivot_row = 0

    for col in range(n):
        # Find pivot
        max_row = pivot_row
        for row in range(pivot_row + 1, augmented.rows):
            if abs(augmented.data[row][col]) > abs(augmented.data[max_row][col]):
                max_row = row

        if abs(augmented.data[max_row][col]) < tolerance:
            continue

        # Swap rows
        augmented.data[pivot_row], augmented.data[max_row] = \
            augmented.data[max_row], augmented.data[pivot_row]

        # Scale pivot row
        pivot = augmented.data[pivot_row][col]
        for j in range(n):
            augmented.data[pivot_row][j] /= pivot

        # Eliminate column
        for row in range(augmented.rows):
            if row != pivot_row:
                factor = augmented.data[row][col]
                for j in range(n):
                    augmented.data[row][j] -= factor * augmented.data[pivot_row][j]

        pivot_cols.append(col)
        pivot_row += 1

    # Free variables
    free_vars = [i for i in range(n) if i not in pivot_cols]

    if not free_vars:
        return []  # Trivial kernel

    # Construct basis vectors
    basis = []
    for free_var in free_vars:
        x = [0.0] * n
        x[free_var] = 1.0

        # Back-substitute to find values of pivot variables
        for i in range(len(pivot_cols) - 1, -1, -1):
            pivot_col = pivot_cols[i]
            value = -sum(augmented.data[i][j] * x[j] for j in range(pivot_col + 1, n))
            x[pivot_col] = value

        basis.append(Vector(x))

    return basis


def image_basis(T: Matrix) -> list[Vector]:
    """
    Compute a basis for the image (column space) of a transformation.

    The image is the set of all possible outputs:
    im(T) = {T(v) : v in domain}

    Args:
        T: Transformation matrix

    Returns:
        List of basis vectors for the image
    """
    # The image is spanned by linearly independent columns
    from .projections import gram_schmidt

    columns = [T.get_column(i) for i in range(T.cols)]

    # Remove zero columns
    non_zero = [v for v in columns if v.magnitude() > 1e-10]

    if not non_zero:
        return []

    # Use Gram-Schmidt to find independent vectors
    try:
        return gram_schmidt(non_zero, normalize=True)
    except ValueError:
        # If Gram-Schmidt fails, return non-zero columns
        return non_zero
