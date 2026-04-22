"""
Vector Operations Module
=========================

Implements a comprehensive Vector class with all operations needed for AI/ML/DL.

Features:
- Basic arithmetic operations (addition, subtraction, scalar multiplication)
- Vector products (dot product, cross product)
- Norms (L1, L2, L-infinity, p-norm)
- Distance metrics
- Angle calculations
- Unit vectors and normalization
"""

from typing import Union, List, Optional
import math


class Vector:
    """
    A mathematical vector with comprehensive operations for AI/ML/DL.

    Attributes:
        elements (list): The components of the vector
        dim (int): The dimension of the vector
    """

    def __init__(self, elements: List[Union[int, float]]):
        """
        Initialize a vector.

        Args:
            elements: List of numerical values representing vector components

        Raises:
            ValueError: If elements list is empty
        """
        if not elements:
            raise ValueError("Vector must have at least one element")
        self.elements = [float(x) for x in elements]
        self.dim = len(elements)

    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector({self.elements})"

    def __str__(self) -> str:
        """Pretty print the vector."""
        return f"[{', '.join(f'{x:.4f}' for x in self.elements)}]"

    def __len__(self) -> int:
        """Return the dimension of the vector."""
        return self.dim

    def __getitem__(self, index: int) -> float:
        """Access vector element by index."""
        return self.elements[index]

    def __setitem__(self, index: int, value: Union[int, float]):
        """Set vector element by index."""
        self.elements[index] = float(value)

    def __eq__(self, other: 'Vector') -> bool:
        """Check equality with another vector."""
        if not isinstance(other, Vector):
            return False
        if self.dim != other.dim:
            return False
        return all(abs(a - b) < 1e-10 for a, b in zip(self.elements, other.elements))

    # ============================================================
    # BASIC ARITHMETIC OPERATIONS
    # ============================================================

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors element-wise.

        Args:
            other: Vector to add

        Returns:
            New vector representing the sum

        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot add vectors of dimensions {self.dim} and {other.dim}")
        return Vector([a + b for a, b in zip(self.elements, other.elements)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract two vectors element-wise.

        Args:
            other: Vector to subtract

        Returns:
            New vector representing the difference

        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot subtract vectors of dimensions {self.dim} and {other.dim}")
        return Vector([a - b for a, b in zip(self.elements, other.elements)])

    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """
        Multiply vector by a scalar.

        Args:
            scalar: Number to multiply by

        Returns:
            New vector scaled by the scalar
        """
        return Vector([scalar * x for x in self.elements])

    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Right multiplication by scalar (scalar * vector)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """
        Divide vector by a scalar.

        Args:
            scalar: Number to divide by

        Returns:
            New vector divided by the scalar

        Raises:
            ValueError: If scalar is zero
        """
        if scalar == 0:
            raise ValueError("Cannot divide vector by zero")
        return Vector([x / scalar for x in self.elements])

    def __neg__(self) -> 'Vector':
        """Negate the vector."""
        return Vector([-x for x in self.elements])

    # ============================================================
    # VECTOR PRODUCTS
    # ============================================================

    def dot(self, other: 'Vector') -> float:
        """
        Compute the dot product (inner product) with another vector.

        The dot product is fundamental in ML for:
        - Similarity measures
        - Projections
        - Neural network computations

        Args:
            other: Vector to compute dot product with

        Returns:
            Scalar dot product

        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot compute dot product of vectors with dimensions {self.dim} and {other.dim}")
        return sum(a * b for a, b in zip(self.elements, other.elements))

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Compute the cross product with another vector (3D only).

        The cross product is used in:
        - 3D rotations
        - Normal vector calculations
        - Physics simulations

        Args:
            other: Vector to compute cross product with

        Returns:
            Vector perpendicular to both input vectors

        Raises:
            ValueError: If vectors are not 3-dimensional
        """
        if self.dim != 3 or other.dim != 3:
            raise ValueError("Cross product is only defined for 3D vectors")

        a, b, c = self.elements
        d, e, f = other.elements

        return Vector([
            b * f - c * e,
            c * d - a * f,
            a * e - b * d
        ])

    def outer(self, other: 'Vector') -> 'Matrix':
        """
        Compute the outer product with another vector.

        The outer product creates a matrix and is used in:
        - Rank-1 updates
        - Gradient computations
        - Attention mechanisms

        Args:
            other: Vector to compute outer product with

        Returns:
            Matrix of shape (self.dim, other.dim)
        """
        from .matrix import Matrix
        result = [[self.elements[i] * other.elements[j]
                  for j in range(other.dim)]
                  for i in range(self.dim)]
        return Matrix(result)

    # ============================================================
    # NORMS AND DISTANCES
    # ============================================================

    def norm(self, p: Union[int, float, str] = 2) -> float:
        """
        Compute the p-norm of the vector.

        Norms are crucial in ML for:
        - Regularization (L1, L2)
        - Gradient clipping
        - Distance metrics

        Args:
            p: Type of norm to compute
               - 1: L1 norm (Manhattan/taxicab)
               - 2: L2 norm (Euclidean) - default
               - float: General p-norm
               - 'inf': L-infinity norm (maximum absolute value)

        Returns:
            The computed norm
        """
        if p == 1:
            return sum(abs(x) for x in self.elements)
        elif p == 2:
            return math.sqrt(sum(x ** 2 for x in self.elements))
        elif p == 'inf' or p == float('inf'):
            return max(abs(x) for x in self.elements)
        else:
            return sum(abs(x) ** p for x in self.elements) ** (1 / p)

    def l1_norm(self) -> float:
        """L1 norm (sum of absolute values). Used in Lasso regularization."""
        return self.norm(1)

    def l2_norm(self) -> float:
        """L2 norm (Euclidean length). Used in Ridge regularization."""
        return self.norm(2)

    def magnitude(self) -> float:
        """Magnitude of the vector (same as L2 norm)."""
        return self.l2_norm()

    def normalize(self) -> 'Vector':
        """
        Return a unit vector in the same direction.

        Normalization is essential in:
        - Feature scaling
        - Gradient descent
        - Embedding representations

        Returns:
            Unit vector in the same direction

        Raises:
            ValueError: If vector is zero
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return self / mag

    def distance(self, other: 'Vector', metric: str = 'euclidean') -> float:
        """
        Compute distance to another vector.

        Args:
            other: Vector to compute distance to
            metric: Distance metric ('euclidean', 'manhattan', 'chebyshev')

        Returns:
            Distance between vectors

        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(f"Cannot compute distance between vectors of dimensions {self.dim} and {other.dim}")

        diff = self - other

        if metric == 'euclidean' or metric == 'l2':
            return diff.l2_norm()
        elif metric == 'manhattan' or metric == 'l1':
            return diff.l1_norm()
        elif metric == 'chebyshev' or metric == 'linf':
            return diff.norm('inf')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # ============================================================
    # ANGLE AND SIMILARITY
    # ============================================================

    def angle(self, other: 'Vector', degrees: bool = False) -> float:
        """
        Compute the angle between this vector and another.

        Args:
            other: Vector to compute angle with
            degrees: If True, return angle in degrees; otherwise radians

        Returns:
            Angle between vectors

        Raises:
            ValueError: If either vector is zero
        """
        if self.magnitude() == 0 or other.magnitude() == 0:
            raise ValueError("Cannot compute angle with zero vector")

        cos_angle = self.dot(other) / (self.magnitude() * other.magnitude())
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = max(-1, min(1, cos_angle))

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad) if degrees else angle_rad

    def cosine_similarity(self, other: 'Vector') -> float:
        """
        Compute cosine similarity with another vector.

        Cosine similarity is widely used in:
        - Document similarity
        - Recommendation systems
        - Word embeddings

        Args:
            other: Vector to compute similarity with

        Returns:
            Cosine similarity in range [-1, 1]
        """
        if self.magnitude() == 0 or other.magnitude() == 0:
            return 0.0
        return self.dot(other) / (self.magnitude() * other.magnitude())

    # ============================================================
    # PROJECTIONS
    # ============================================================

    def project_onto(self, other: 'Vector') -> 'Vector':
        """
        Project this vector onto another vector.

        Projections are used in:
        - Orthogonalization
        - Dimensionality reduction
        - Least squares

        Args:
            other: Vector to project onto

        Returns:
            Projection of self onto other

        Raises:
            ValueError: If other is zero vector
        """
        if other.magnitude() == 0:
            raise ValueError("Cannot project onto zero vector")

        scalar_proj = self.dot(other) / other.dot(other)
        return scalar_proj * other

    def reject_from(self, other: 'Vector') -> 'Vector':
        """
        Compute the rejection of this vector from another.

        The rejection is the component perpendicular to the other vector.

        Args:
            other: Vector to reject from

        Returns:
            Component of self perpendicular to other
        """
        return self - self.project_onto(other)

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def is_zero(self, tolerance: float = 1e-10) -> bool:
        """Check if vector is zero within tolerance."""
        return all(abs(x) < tolerance for x in self.elements)

    def is_unit(self, tolerance: float = 1e-10) -> bool:
        """Check if vector is a unit vector."""
        return abs(self.magnitude() - 1.0) < tolerance

    def is_orthogonal(self, other: 'Vector', tolerance: float = 1e-10) -> bool:
        """Check if this vector is orthogonal to another."""
        return abs(self.dot(other)) < tolerance

    def copy(self) -> 'Vector':
        """Create a deep copy of the vector."""
        return Vector(self.elements.copy())

    def to_list(self) -> List[float]:
        """Convert vector to Python list."""
        return self.elements.copy()

    @staticmethod
    def zero(dim: int) -> 'Vector':
        """Create a zero vector of given dimension."""
        return Vector([0.0] * dim)

    @staticmethod
    def ones(dim: int) -> 'Vector':
        """Create a vector of ones of given dimension."""
        return Vector([1.0] * dim)

    @staticmethod
    def basis(dim: int, index: int) -> 'Vector':
        """
        Create a standard basis vector.

        Args:
            dim: Dimension of the vector
            index: Index of the 1 (0-indexed)

        Returns:
            Standard basis vector
        """
        if index >= dim or index < 0:
            raise ValueError(f"Index {index} out of range for dimension {dim}")
        elements = [0.0] * dim
        elements[index] = 1.0
        return Vector(elements)

    @staticmethod
    def from_angle(angle: float, degrees: bool = False) -> 'Vector':
        """
        Create a 2D unit vector from an angle.

        Args:
            angle: Angle from positive x-axis
            degrees: If True, angle is in degrees

        Returns:
            2D unit vector
        """
        if degrees:
            angle = math.radians(angle)
        return Vector([math.cos(angle), math.sin(angle)])
