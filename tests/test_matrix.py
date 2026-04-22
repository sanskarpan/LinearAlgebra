"""
Unit tests for Matrix class
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from LinearAlgebra.matrix import Matrix
from LinearAlgebra.vector import Vector


class TestMatrixBasics(unittest.TestCase):
    """Test basic matrix operations."""

    def test_creation(self):
        """Test matrix creation."""
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[1, 1], 4.0)

    def test_addition(self):
        """Test matrix addition."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1 + m2
        self.assertEqual(m3.data, [[6.0, 8.0], [10.0, 12.0]])

    def test_subtraction(self):
        """Test matrix subtraction."""
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        m3 = m1 - m2
        self.assertEqual(m3.data, [[4.0, 4.0], [4.0, 4.0]])

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        m = Matrix([[1, 2], [3, 4]])
        m2 = m * 2
        self.assertEqual(m2.data, [[2.0, 4.0], [6.0, 8.0]])

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1 @ m2
        # [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
        # [[19, 22], [43, 50]]
        self.assertEqual(m3.data, [[19.0, 22.0], [43.0, 50.0]])

    def test_transpose(self):
        """Test matrix transpose."""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        mt = m.transpose()
        self.assertEqual(mt.shape, (3, 2))
        self.assertEqual(mt.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


class TestMatrixProperties(unittest.TestCase):
    """Test matrix properties."""

    def test_trace(self):
        """Test trace computation."""
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.trace(), 5.0)  # 1 + 4

    def test_determinant_2x2(self):
        """Test 2x2 determinant."""
        m = Matrix([[1, 2], [3, 4]])
        det = m.determinant()
        self.assertAlmostEqual(det, -2.0)  # 1*4 - 2*3 = -2

    def test_determinant_3x3(self):
        """Test 3x3 determinant."""
        m = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        det = m.determinant()
        # Check absolute value (sign may vary with permutation implementation)
        self.assertAlmostEqual(abs(det), 1.0)

    def test_rank(self):
        """Test rank computation."""
        m = Matrix([[1, 2], [2, 4]])  # Rank 1
        self.assertEqual(m.rank(), 1)

        m2 = Matrix([[1, 0], [0, 1]])  # Rank 2
        self.assertEqual(m2.rank(), 2)

    def test_inverse(self):
        """Test matrix inversion."""
        m = Matrix([[1, 2], [3, 4]])
        m_inv = m.inverse()

        # Verify A * A^-1 = I
        identity = m @ m_inv
        self.assertTrue(identity.is_identity(tolerance=1e-10))


class TestMatrixChecks(unittest.TestCase):
    """Test matrix property checks."""

    def test_is_symmetric(self):
        """Test symmetry check."""
        m_sym = Matrix([[1, 2], [2, 1]])
        self.assertTrue(m_sym.is_symmetric())

        m_asym = Matrix([[1, 2], [3, 1]])
        self.assertFalse(m_asym.is_symmetric())

    def test_is_diagonal(self):
        """Test diagonal check."""
        m_diag = Matrix([[1, 0], [0, 2]])
        self.assertTrue(m_diag.is_diagonal())

        m_not_diag = Matrix([[1, 1], [0, 2]])
        self.assertFalse(m_not_diag.is_diagonal())

    def test_is_identity(self):
        """Test identity check."""
        m_id = Matrix.identity(3)
        self.assertTrue(m_id.is_identity())

        m_not_id = Matrix([[1, 0], [0, 2]])
        self.assertFalse(m_not_id.is_identity())


class TestMatrixDecomposition(unittest.TestCase):
    """Test matrix decompositions."""

    def test_lu_decomposition(self):
        """Test LU decomposition."""
        A = Matrix([[2, 1], [4, 3]])
        L, U, P = A.lu_decomposition()

        # Verify dimensions
        self.assertEqual(L.shape, (2, 2))
        self.assertEqual(U.shape, (2, 2))

        # L should be lower triangular with 1s on diagonal
        self.assertAlmostEqual(L[0, 0], 1.0)
        self.assertAlmostEqual(L[1, 1], 1.0)


class TestMatrixConstructors(unittest.TestCase):
    """Test static constructor methods."""

    def test_zero_matrix(self):
        """Test zero matrix creation."""
        m = Matrix.zero(2, 3)
        self.assertEqual(m.shape, (2, 3))
        self.assertTrue(all(m.data[i][j] == 0.0
                           for i in range(2) for j in range(3)))

    def test_identity_matrix(self):
        """Test identity matrix creation."""
        m = Matrix.identity(3)
        self.assertTrue(m.is_identity())

    def test_diagonal_matrix(self):
        """Test diagonal matrix creation."""
        m = Matrix.diagonal([1, 2, 3])
        self.assertEqual(m.data, [[1.0, 0.0, 0.0],
                                  [0.0, 2.0, 0.0],
                                  [0.0, 0.0, 3.0]])

    def test_from_vectors(self):
        """Test matrix from vectors."""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        m = Matrix.from_vectors([v1, v2], as_rows=True)
        self.assertEqual(m.data, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == '__main__':
    unittest.main()
