"""
Basic Linear Algebra Operations - Tutorial
===========================================

This script demonstrates basic vector and matrix operations
essential for understanding AI/ML/DL.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from LinearAlgebra import Vector, Matrix
import math


def vector_examples():
    """Demonstrate vector operations."""
    print("=" * 60)
    print("VECTOR OPERATIONS")
    print("=" * 60)

    # Create vectors
    print("\n1. Creating Vectors")
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")

    # Addition
    print("\n2. Vector Addition")
    v3 = v1 + v2
    print(f"v1 + v2 = {v3}")

    # Scalar multiplication
    print("\n3. Scalar Multiplication")
    v4 = 2 * v1
    print(f"2 * v1 = {v4}")

    # Dot product
    print("\n4. Dot Product")
    dot = v1.dot(v2)
    print(f"v1 · v2 = {dot}")

    # Cross product (3D only)
    print("\n5. Cross Product")
    a = Vector([1, 0, 0])
    b = Vector([0, 1, 0])
    c = a.cross(b)
    print(f"{a} × {b} = {c}")

    # Norms
    print("\n6. Vector Norms")
    v = Vector([3, 4])
    print(f"v = {v}")
    print(f"L1 norm (Manhattan): {v.l1_norm()}")
    print(f"L2 norm (Euclidean): {v.l2_norm()}")
    print(f"Magnitude: {v.magnitude()}")

    # Normalization
    print("\n7. Normalization")
    v_normalized = v.normalize()
    print(f"Normalized: {v_normalized}")
    print(f"Magnitude after normalization: {v_normalized.magnitude()}")

    # Angle between vectors
    print("\n8. Angle Between Vectors")
    x = Vector([1, 0])
    y = Vector([1, 1])
    angle_rad = x.angle(y)
    angle_deg = x.angle(y, degrees=True)
    print(f"Angle between {x} and {y}:")
    print(f"  Radians: {angle_rad:.4f}")
    print(f"  Degrees: {angle_deg:.2f}")

    # Projection
    print("\n9. Vector Projection")
    v = Vector([3, 4])
    u = Vector([1, 0])
    proj = v.project_onto(u)
    print(f"Projection of {v} onto {u} = {proj}")


def matrix_examples():
    """Demonstrate matrix operations."""
    print("\n" + "=" * 60)
    print("MATRIX OPERATIONS")
    print("=" * 60)

    # Create matrices
    print("\n1. Creating Matrices")
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    print(f"A =\n{A}")
    print(f"B =\n{B}")

    # Addition
    print("\n2. Matrix Addition")
    C = A + B
    print(f"A + B =\n{C}")

    # Multiplication
    print("\n3. Matrix Multiplication")
    D = A @ B
    print(f"A @ B =\n{D}")

    # Transpose
    print("\n4. Transpose")
    At = A.T
    print(f"A^T =\n{At}")

    # Determinant
    print("\n5. Determinant")
    det_A = A.determinant()
    print(f"det(A) = {det_A}")

    # Inverse
    print("\n6. Matrix Inverse")
    A_inv = A.inverse()
    print(f"A^-1 =\n{A_inv}")

    # Verify A * A^-1 = I
    identity = A @ A_inv
    print(f"A @ A^-1 (should be identity):\n{identity}")

    # Special matrices
    print("\n7. Special Matrices")
    I = Matrix.identity(3)
    Z = Matrix.zero(2, 3)
    D = Matrix.diagonal([1, 2, 3])

    print(f"Identity (3x3):\n{I}")
    print(f"Zero (2x3):\n{Z}")
    print(f"Diagonal:\n{D}")

    # Matrix-vector multiplication
    print("\n8. Matrix-Vector Multiplication")
    M = Matrix([[1, 2], [3, 4]])
    v = Vector([5, 6])
    result = M @ v
    print(f"M =\n{M}")
    print(f"v = {v}")
    print(f"M @ v = {result}")


def ml_applications():
    """Demonstrate ML/DL applications."""
    print("\n" + "=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("=" * 60)

    # Linear transformation
    print("\n1. Linear Transformation (Simple Neural Network Layer)")
    print("   y = Wx + b")

    # Weight matrix (2 inputs, 3 outputs)
    W = Matrix([[0.5, 0.3], [0.2, 0.8], [0.7, 0.1]])
    x = Vector([1.0, 2.0])
    b = Vector([0.1, 0.2, 0.3])

    y = (W @ x) + b

    print(f"Weight matrix W:\n{W}")
    print(f"Input x: {x}")
    print(f"Bias b: {b}")
    print(f"Output y: {y}")

    # Distance metrics (used in k-NN)
    print("\n2. Distance Metrics (used in k-NN, clustering)")
    p1 = Vector([1, 2])
    p2 = Vector([4, 6])

    euclidean = p1.distance(p2, metric='euclidean')
    manhattan = p1.distance(p2, metric='manhattan')

    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Euclidean distance: {euclidean:.4f}")
    print(f"Manhattan distance: {manhattan:.4f}")

    # Cosine similarity (used in NLP, recommender systems)
    print("\n3. Cosine Similarity (NLP, Recommender Systems)")
    doc1 = Vector([1, 2, 3, 0, 0])  # Document vector
    doc2 = Vector([0, 1, 2, 3, 1])  # Another document

    similarity = doc1.cosine_similarity(doc2)

    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Cosine similarity: {similarity:.4f}")

    # Frobenius norm (regularization)
    print("\n4. Frobenius Norm (Weight Regularization)")
    weights = Matrix([[0.5, 0.3], [0.2, 0.8]])
    frob_norm = weights.frobenius_norm()

    print(f"Weight matrix:\n{weights}")
    print(f"Frobenius norm: {frob_norm:.4f}")
    print(f"L2 regularization term (λ=0.01): {0.01 * frob_norm**2:.6f}")


def main():
    """Run all examples."""
    vector_examples()
    matrix_examples()
    ml_applications()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
