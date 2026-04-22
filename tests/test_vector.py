"""
Unit tests for Vector class
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import math
from LinearAlgebra.vector import Vector


class TestVectorBasics(unittest.TestCase):
    """Test basic vector operations."""

    def test_creation(self):
        """Test vector creation."""
        v = Vector([1, 2, 3])
        self.assertEqual(v.dim, 3)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 2.0)
        self.assertEqual(v[2], 3.0)

    def test_addition(self):
        """Test vector addition."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        v3 = v1 + v2
        self.assertEqual(v3.to_list(), [5.0, 7.0, 9.0])

    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector([4, 5, 6])
        v2 = Vector([1, 2, 3])
        v3 = v1 - v2
        self.assertEqual(v3.to_list(), [3.0, 3.0, 3.0])

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector([1, 2, 3])
        v2 = v * 2
        self.assertEqual(v2.to_list(), [2.0, 4.0, 6.0])

        v3 = 3 * v
        self.assertEqual(v3.to_list(), [3.0, 6.0, 9.0])

    def test_dot_product(self):
        """Test dot product."""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        dot = v1.dot(v2)
        self.assertEqual(dot, 32.0)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    def test_cross_product(self):
        """Test cross product."""
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = v1.cross(v2)
        self.assertEqual(v3.to_list(), [0.0, 0.0, 1.0])


class TestVectorNorms(unittest.TestCase):
    """Test vector norms and distances."""

    def test_l1_norm(self):
        """Test L1 norm."""
        v = Vector([3, -4, 5])
        self.assertEqual(v.l1_norm(), 12.0)

    def test_l2_norm(self):
        """Test L2 norm (magnitude)."""
        v = Vector([3, 4])
        self.assertEqual(v.l2_norm(), 5.0)
        self.assertEqual(v.magnitude(), 5.0)

    def test_normalize(self):
        """Test normalization."""
        v = Vector([3, 4])
        v_norm = v.normalize()
        self.assertAlmostEqual(v_norm.magnitude(), 1.0)
        self.assertAlmostEqual(v_norm[0], 0.6)
        self.assertAlmostEqual(v_norm[1], 0.8)

    def test_distance(self):
        """Test distance computation."""
        v1 = Vector([0, 0])
        v2 = Vector([3, 4])
        dist = v1.distance(v2)
        self.assertEqual(dist, 5.0)


class TestVectorAngles(unittest.TestCase):
    """Test angle and similarity computations."""

    def test_angle(self):
        """Test angle computation."""
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        angle = v1.angle(v2)
        self.assertAlmostEqual(angle, math.pi / 2)

        angle_deg = v1.angle(v2, degrees=True)
        self.assertAlmostEqual(angle_deg, 90.0)

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        v1 = Vector([1, 0])
        v2 = Vector([1, 0])
        cos_sim = v1.cosine_similarity(v2)
        self.assertAlmostEqual(cos_sim, 1.0)

        v3 = Vector([0, 1])
        cos_sim2 = v1.cosine_similarity(v3)
        self.assertAlmostEqual(cos_sim2, 0.0)


class TestVectorProjections(unittest.TestCase):
    """Test vector projections."""

    def test_projection(self):
        """Test vector projection."""
        v = Vector([3, 4])
        u = Vector([1, 0])
        proj = v.project_onto(u)
        self.assertEqual(proj.to_list(), [3.0, 0.0])

    def test_rejection(self):
        """Test vector rejection."""
        v = Vector([3, 4])
        u = Vector([1, 0])
        rej = v.reject_from(u)
        self.assertEqual(rej.to_list(), [0.0, 4.0])


class TestVectorUtilities(unittest.TestCase):
    """Test utility methods."""

    def test_is_orthogonal(self):
        """Test orthogonality check."""
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        self.assertTrue(v1.is_orthogonal(v2))

        v3 = Vector([1, 1])
        self.assertFalse(v1.is_orthogonal(v3))

    def test_static_constructors(self):
        """Test static constructor methods."""
        zero = Vector.zero(3)
        self.assertEqual(zero.to_list(), [0.0, 0.0, 0.0])

        ones = Vector.ones(3)
        self.assertEqual(ones.to_list(), [1.0, 1.0, 1.0])

        basis = Vector.basis(3, 1)
        self.assertEqual(basis.to_list(), [0.0, 1.0, 0.0])


if __name__ == '__main__':
    unittest.main()
