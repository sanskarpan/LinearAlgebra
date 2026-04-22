"""
Tensor Operations Module for Deep Learning
===========================================

Implements basic tensor operations essential for understanding deep learning:
- Multi-dimensional arrays (tensors)
- Broadcasting
- Tensor contractions
- Activation functions
- Common operations used in neural networks

Note: This is an educational implementation. For production deep learning,
use frameworks like PyTorch or TensorFlow.
"""

from typing import List, Union, Tuple, Optional, Callable
import math
from .vector import Vector
from .matrix import Matrix


class Tensor:
    """
    A multi-dimensional array (tensor) for deep learning operations.

    In deep learning:
    - 0D tensor: Scalar
    - 1D tensor: Vector
    - 2D tensor: Matrix
    - 3D tensor: Batch of matrices or RGB image
    - 4D tensor: Batch of RGB images
    """

    def __init__(self, data: Union[float, List], shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize a tensor.

        Args:
            data: Nested list structure or flat list with shape
            shape: Optional shape tuple if data is flat list
        """
        if isinstance(data, (int, float)):
            # Scalar
            self.data = float(data)
            self.shape = ()
            self.ndim = 0
        elif shape is not None:
            # Flat list with shape
            self.shape = shape
            self.ndim = len(shape)
            self.data = self._reshape_flat(list(data), shape)
        else:
            # Nested list
            self.data = self._to_nested_list(data)
            self.shape = self._infer_shape(self.data)
            self.ndim = len(self.shape)

        self.size = self._compute_size()

    def _to_nested_list(self, data):
        """Convert data to nested list structure."""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, Vector):
            return data.to_list()
        elif isinstance(data, Matrix):
            return data.to_list()
        elif isinstance(data, list):
            return [self._to_nested_list(item) for item in data]
        else:
            return float(data)

    def _infer_shape(self, data) -> Tuple[int, ...]:
        """Infer shape from nested list structure."""
        if isinstance(data, (int, float)):
            return ()

        shape = [len(data)]
        if data and isinstance(data[0], list):
            shape.extend(self._infer_shape(data[0]))

        return tuple(shape)

    def _reshape_flat(self, flat_data: List[float], shape: Tuple[int, ...]):
        """Reshape flat list into nested structure."""
        if not shape:
            return flat_data[0]

        if len(shape) == 1:
            return flat_data

        size = 1
        for dim in shape[1:]:
            size *= dim

        result = []
        for i in range(shape[0]):
            start = i * size
            end = start + size
            result.append(self._reshape_flat(flat_data[start:end], shape[1:]))

        return result

    def _compute_size(self) -> int:
        """Compute total number of elements."""
        if self.ndim == 0:
            return 1
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"

    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}):\n{self._format_data()}"

    def _format_data(self, indent: int = 0) -> str:
        """Format tensor data for printing."""
        if self.ndim == 0:
            return f"{self.data:.4f}"
        elif self.ndim == 1:
            return "[" + ", ".join(f"{x:.4f}" for x in self.data) + "]"
        else:
            lines = []
            for item in self.data:
                sub_tensor = Tensor(item)
                lines.append("  " * indent + sub_tensor._format_data(indent + 1))
            return "[\n" + "\n".join(lines) + "\n" + "  " * indent + "]"

    # ============================================================
    # ARITHMETIC OPERATIONS
    # ============================================================

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise addition with broadcasting."""
        if isinstance(other, (int, float)):
            return Tensor(self._apply_scalar(self.data, lambda x: x + other))

        # Broadcasting
        return self._elementwise_op(other, lambda a, b: a + b)

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise subtraction with broadcasting."""
        if isinstance(other, (int, float)):
            return Tensor(self._apply_scalar(self.data, lambda x: x - other))

        return self._elementwise_op(other, lambda a, b: a - b)

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication (Hadamard product) with broadcasting."""
        if isinstance(other, (int, float)):
            return Tensor(self._apply_scalar(self.data, lambda x: x * other))

        return self._elementwise_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division with broadcasting."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Division by zero")
            return Tensor(self._apply_scalar(self.data, lambda x: x / other))

        return self._elementwise_op(other, lambda a, b: a / b if b != 0 else float('inf'))

    def __pow__(self, power: float) -> 'Tensor':
        """Element-wise power."""
        return Tensor(self._apply_scalar(self.data, lambda x: x ** power))

    def __neg__(self) -> 'Tensor':
        """Negate tensor."""
        return Tensor(self._apply_scalar(self.data, lambda x: -x))

    def _apply_scalar(self, data, func: Callable):
        """Apply function to all elements."""
        if isinstance(data, (int, float)):
            return func(data)
        return [self._apply_scalar(item, func) for item in data]

    def _elementwise_op(self, other: 'Tensor', op: Callable) -> 'Tensor':
        """Perform element-wise operation with broadcasting."""
        # Simple broadcasting for same-shape tensors
        if self.shape == other.shape:
            return Tensor(self._apply_elementwise(self.data, other.data, op))

        # Scalar broadcasting
        if other.ndim == 0:
            return Tensor(self._apply_scalar(self.data, lambda x: op(x, other.data)))

        if self.ndim == 0:
            return Tensor(self._apply_scalar(other.data, lambda x: op(self.data, x)))

        # General broadcasting (simplified version)
        raise NotImplementedError("Complex broadcasting not yet implemented")

    def _apply_elementwise(self, data1, data2, op: Callable):
        """Apply operation element-wise."""
        if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            return op(data1, data2)

        return [self._apply_elementwise(d1, d2, op) for d1, d2 in zip(data1, data2)]

    # ============================================================
    # TENSOR OPERATIONS
    # ============================================================

    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """
        Reshape tensor to new shape.

        Args:
            new_shape: New shape tuple

        Returns:
            Reshaped tensor

        Raises:
            ValueError: If new shape is incompatible
        """
        new_size = 1
        for dim in new_shape:
            new_size *= dim

        if new_size != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to shape {new_shape}")

        flat = self.flatten().data
        return Tensor(flat, new_shape)

    def flatten(self) -> 'Tensor':
        """Flatten tensor to 1D."""
        flat_data = self._flatten_recursive(self.data)
        return Tensor(flat_data)

    def _flatten_recursive(self, data) -> List[float]:
        """Recursively flatten nested structure."""
        if isinstance(data, (int, float)):
            return [float(data)]

        result = []
        for item in data:
            result.extend(self._flatten_recursive(item))
        return result

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """
        Transpose tensor dimensions.

        For 2D tensors, swaps rows and columns.
        For higher dimensions, permutes axes.

        Args:
            axes: Permutation of axes (None = reverse all axes)

        Returns:
            Transposed tensor
        """
        if self.ndim == 0:
            return Tensor(self.data)

        if self.ndim == 1:
            return Tensor(self.data)

        if self.ndim == 2:
            # Matrix transpose
            transposed = [[self.data[i][j] for i in range(self.shape[0])]
                         for j in range(self.shape[1])]
            return Tensor(transposed)

        # For higher dimensions, use axes permutation
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))

        # Complex transpose implementation
        raise NotImplementedError("High-dimensional transpose with custom axes not yet implemented")

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Union['Tensor', float]:
        """
        Sum tensor elements.

        Args:
            axis: Axis to sum over (None = all elements)
            keepdims: Keep reduced dimensions

        Returns:
            Sum result
        """
        if axis is None:
            # Sum all elements
            total = sum(self.flatten().data)
            return Tensor(total) if keepdims else total

        # Sum over specific axis
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"Axis {axis} out of range for {self.ndim}D tensor")

        # Implementation for specific axes
        if self.ndim == 2 and axis == 0:
            # Sum over rows
            result = [sum(self.data[i][j] for i in range(self.shape[0]))
                     for j in range(self.shape[1])]
            return Tensor(result)
        elif self.ndim == 2 and axis == 1:
            # Sum over columns
            result = [sum(row) for row in self.data]
            return Tensor(result)

        raise NotImplementedError("Sum over arbitrary axes not fully implemented")

    def mean(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """Compute mean of tensor elements."""
        sum_result = self.sum(axis)

        if axis is None:
            return sum_result / self.size

        # Compute size of reduced dimension
        if isinstance(sum_result, Tensor):
            scale = self.shape[axis]
            return sum_result / scale

        return sum_result / self.shape[axis]

    # ============================================================
    # ACTIVATION FUNCTIONS
    # ============================================================

    def relu(self) -> 'Tensor':
        """
        ReLU activation: max(0, x)

        Widely used in deep learning for:
        - Hidden layers
        - Introducing non-linearity
        - Gradient flow
        """
        return Tensor(self._apply_scalar(self.data, lambda x: max(0.0, x)))

    def sigmoid(self) -> 'Tensor':
        """
        Sigmoid activation: 1 / (1 + exp(-x))

        Used in:
        - Binary classification
        - Gate mechanisms (LSTM)
        - Output layers
        """
        def sigmoid_func(x):
            # Numerically stable sigmoid
            if x >= 0:
                z = math.exp(-x)
                return 1 / (1 + z)
            else:
                z = math.exp(x)
                return z / (1 + z)

        return Tensor(self._apply_scalar(self.data, sigmoid_func))

    def tanh(self) -> 'Tensor':
        """
        Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Used in:
        - Hidden layers
        - RNN/LSTM
        - Range: [-1, 1]
        """
        return Tensor(self._apply_scalar(self.data, math.tanh))

    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Softmax activation: exp(x_i) / sum(exp(x_j))

        Used in:
        - Multi-class classification
        - Attention mechanisms
        - Output layers

        Args:
            axis: Axis to apply softmax over

        Returns:
            Softmax probabilities
        """
        if self.ndim == 1:
            # 1D softmax
            max_val = max(self.data)
            exp_vals = [math.exp(x - max_val) for x in self.data]  # Numerical stability
            sum_exp = sum(exp_vals)
            return Tensor([e / sum_exp for e in exp_vals])

        elif self.ndim == 2 and axis == 1:
            # 2D softmax over rows
            result = []
            for row in self.data:
                max_val = max(row)
                exp_vals = [math.exp(x - max_val) for x in row]
                sum_exp = sum(exp_vals)
                result.append([e / sum_exp for e in exp_vals])
            return Tensor(result)

        raise NotImplementedError("Softmax for higher dimensions not fully implemented")

    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        """
        Leaky ReLU: max(alpha * x, x)

        Args:
            alpha: Slope for negative values

        Returns:
            Activated tensor
        """
        return Tensor(self._apply_scalar(self.data, lambda x: x if x > 0 else alpha * x))

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        return Tensor(self._apply_scalar(self.data, math.exp))

    def log(self) -> 'Tensor':
        """Element-wise natural logarithm."""
        return Tensor(self._apply_scalar(self.data, lambda x: math.log(x) if x > 0 else float('-inf')))

    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        return Tensor(self._apply_scalar(self.data, lambda x: math.sqrt(x) if x >= 0 else float('nan')))

    def clip(self, min_val: float, max_val: float) -> 'Tensor':
        """
        Clip tensor values to range [min_val, max_val].

        Used in:
        - Gradient clipping
        - Numerical stability
        """
        return Tensor(self._apply_scalar(
            self.data,
            lambda x: max(min_val, min(max_val, x))
        ))

    def to_matrix(self) -> Matrix:
        """Convert 2D tensor to Matrix."""
        if self.ndim != 2:
            raise ValueError("Can only convert 2D tensors to Matrix")
        return Matrix(self.data)

    def to_vector(self) -> Vector:
        """Convert 1D tensor to Vector."""
        if self.ndim != 1:
            raise ValueError("Can only convert 1D tensors to Vector")
        return Vector(self.data)

    @staticmethod
    def from_matrix(matrix: Matrix) -> 'Tensor':
        """Create tensor from Matrix."""
        return Tensor(matrix.to_list())

    @staticmethod
    def from_vector(vector: Vector) -> 'Tensor':
        """Create tensor from Vector."""
        return Tensor(vector.to_list())

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> 'Tensor':
        """Create tensor of zeros."""
        size = 1
        for dim in shape:
            size *= dim
        return Tensor([0.0] * size, shape)

    @staticmethod
    def ones(shape: Tuple[int, ...]) -> 'Tensor':
        """Create tensor of ones."""
        size = 1
        for dim in shape:
            size *= dim
        return Tensor([1.0] * size, shape)

    @staticmethod
    def random_uniform(shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0) -> 'Tensor':
        """Create tensor with uniform random values."""
        import random
        size = 1
        for dim in shape:
            size *= dim
        data = [random.uniform(low, high) for _ in range(size)]
        return Tensor(data, shape)

    @staticmethod
    def random_normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> 'Tensor':
        """
        Create tensor with normal random values.

        Used for:
        - Weight initialization
        - Xavier/He initialization
        """
        import random
        size = 1
        for dim in shape:
            size *= dim
        data = [random.gauss(mean, std) for _ in range(size)]
        return Tensor(data, shape)


# ============================================================
# COMMON DEEP LEARNING OPERATIONS
# ============================================================

def batch_matrix_multiply(A: Tensor, B: Tensor) -> Tensor:
    """
    Batch matrix multiplication for 3D tensors.

    Used in:
    - Batch processing in neural networks
    - Attention mechanisms

    Args:
        A: Tensor of shape (batch, m, k)
        B: Tensor of shape (batch, k, n)

    Returns:
        Tensor of shape (batch, m, n)
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Batch matrix multiply requires 3D tensors")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Batch sizes must match")

    if A.shape[2] != B.shape[1]:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    batch_size = A.shape[0]
    result = []

    for b in range(batch_size):
        # Extract matrices
        A_matrix = Matrix(A.data[b])
        B_matrix = Matrix(B.data[b])

        # Multiply
        C_matrix = A_matrix @ B_matrix

        result.append(C_matrix.to_list())

    return Tensor(result)


def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> float:
    """
    Compute cross-entropy loss for classification.

    Used in:
    - Training neural networks
    - Multi-class classification

    Args:
        predictions: Predicted probabilities (after softmax)
        targets: True labels (one-hot encoded)

    Returns:
        Cross-entropy loss
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have same shape")

    # -sum(targets * log(predictions))
    epsilon = 1e-10  # For numerical stability

    def compute_loss(pred, target):
        if isinstance(pred, list):
            return sum(compute_loss(p, t) for p, t in zip(pred, target))
        return -target * math.log(pred + epsilon)

    total_loss = compute_loss(predictions.data, targets.data)

    # Average over batch
    if predictions.ndim == 2:
        return total_loss / predictions.shape[0]

    return total_loss


def mse_loss(predictions: Tensor, targets: Tensor) -> float:
    """
    Mean Squared Error loss.

    Used in:
    - Regression problems
    - Autoencoders

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        MSE loss
    """
    diff = predictions - targets
    squared = diff ** 2
    return squared.mean()


def l2_regularization(weights: Tensor, lambda_reg: float = 0.01) -> float:
    """
    Compute L2 regularization term.

    Used for:
    - Preventing overfitting
    - Weight decay

    Args:
        weights: Weight tensor
        lambda_reg: Regularization strength

    Returns:
        L2 penalty
    """
    squared = weights ** 2
    return lambda_reg * squared.sum()


def dropout(x: Tensor, drop_rate: float = 0.5, training: bool = True) -> Tensor:
    """
    Apply dropout for regularization.

    During training, randomly sets elements to zero.
    During inference, returns input unchanged.

    Args:
        x: Input tensor
        drop_rate: Probability of dropping a unit
        training: Whether in training mode

    Returns:
        Tensor with dropout applied
    """
    if not training or drop_rate == 0:
        return x

    import random

    def apply_dropout(data):
        if isinstance(data, (int, float)):
            return 0.0 if random.random() < drop_rate else data / (1 - drop_rate)
        return [apply_dropout(item) for item in data]

    return Tensor(apply_dropout(x.data))
