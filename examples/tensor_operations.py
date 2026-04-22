"""
Tensor Operations for Deep Learning - Tutorial
===============================================

Demonstrates tensor operations essential for understanding deep learning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from LinearAlgebra.tensor import Tensor, cross_entropy_loss, mse_loss


def basic_tensor_ops():
    """Demonstrate basic tensor operations."""
    print("=" * 60)
    print("BASIC TENSOR OPERATIONS")
    print("=" * 60)

    # Creating tensors
    print("\n1. Creating Tensors")
    t1 = Tensor([1, 2, 3, 4])  # 1D tensor (vector)
    t2 = Tensor([[1, 2], [3, 4]])  # 2D tensor (matrix)
    t3 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D tensor

    print(f"1D Tensor: {t1}")
    print(f"2D Tensor: {t2}")
    print(f"3D Tensor: {t3}")

    # Arithmetic operations
    print("\n2. Arithmetic Operations")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b =\n{a + b}")
    print(f"a * 2 =\n{a * 2}")
    print(f"a * b (element-wise) =\n{a * b}")

    # Reshaping
    print("\n3. Reshaping")
    t = Tensor([1, 2, 3, 4, 5, 6])
    print(f"Original shape: {t.shape}")
    reshaped = t.reshape((2, 3))
    print(f"Reshaped to (2, 3):\n{reshaped}")

    # Transpose
    print("\n4. Transpose")
    m = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original:\n{m}")
    transposed = m.transpose()
    print(f"Transposed:\n{transposed}")


def activation_functions():
    """Demonstrate activation functions."""
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTIONS")
    print("=" * 60)

    x = Tensor([-2, -1, 0, 1, 2])

    print(f"\nInput: {x}")

    # ReLU
    print("\n1. ReLU (Rectified Linear Unit)")
    relu_out = x.relu()
    print(f"ReLU(x) = {relu_out}")

    # Sigmoid
    print("\n2. Sigmoid")
    sigmoid_out = x.sigmoid()
    print(f"Sigmoid(x) = {sigmoid_out}")

    # Tanh
    print("\n3. Tanh")
    tanh_out = x.tanh()
    print(f"Tanh(x) = {tanh_out}")

    # Leaky ReLU
    print("\n4. Leaky ReLU")
    leaky_relu_out = x.leaky_relu(alpha=0.1)
    print(f"Leaky ReLU(x, α=0.1) = {leaky_relu_out}")

    # Softmax
    print("\n5. Softmax (for classification)")
    logits = Tensor([1.0, 2.0, 3.0])
    probs = logits.softmax()
    print(f"Logits: {logits}")
    print(f"Softmax(logits) = {probs}")
    print(f"Sum of probabilities: {probs.sum():.4f}")


def neural_network_layer():
    """Simulate a simple neural network layer."""
    print("\n" + "=" * 60)
    print("SIMPLE NEURAL NETWORK LAYER")
    print("=" * 60)

    print("\nSimulating: y = ReLU(Wx + b)")

    # Input (batch_size=2, input_dim=3)
    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])

    # Weights (input_dim=3, output_dim=2)
    W = Tensor([[0.5, 0.3, 0.2],
                [0.1, 0.4, 0.7]])

    # Bias (output_dim=2)
    b = Tensor([0.1, 0.2])

    print(f"Input x (2 samples, 3 features):\n{x}")
    print(f"Weights W (2 outputs, 3 inputs):\n{W}")
    print(f"Bias b: {b}")

    # Forward pass (simplified - proper implementation would use batch operations)
    print("\nForward pass:")
    print("1. Linear transformation: z = Wx + b")
    print("2. Activation: y = ReLU(z)")

    # Convert to matrices for multiplication
    from LinearAlgebra import Matrix, Vector

    x_matrix = Matrix(x.data)
    W_matrix = Matrix(W.data)

    # z = x @ W^T + b
    z_matrix = x_matrix @ W_matrix.T

    # Add bias (broadcasting)
    z_data = [[z_matrix[i, j] + b.data[j] for j in range(2)] for i in range(2)]
    z = Tensor(z_data)

    print(f"After linear transformation z:\n{z}")

    # Apply ReLU
    y = z.relu()
    print(f"After ReLU activation y:\n{y}")


def loss_functions():
    """Demonstrate loss functions."""
    print("\n" + "=" * 60)
    print("LOSS FUNCTIONS")
    print("=" * 60)

    # Mean Squared Error (regression)
    print("\n1. Mean Squared Error (MSE) - for Regression")
    predictions = Tensor([2.5, 3.0, 4.5])
    targets = Tensor([2.0, 3.5, 4.0])

    mse = mse_loss(predictions, targets)

    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    print(f"MSE Loss: {mse:.4f}")

    # Cross-Entropy (classification)
    print("\n2. Cross-Entropy Loss - for Classification")
    # Predicted probabilities (after softmax)
    pred_probs = Tensor([[0.7, 0.2, 0.1],
                         [0.1, 0.8, 0.1]])

    # True labels (one-hot encoded)
    true_labels = Tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])

    ce_loss = cross_entropy_loss(pred_probs, true_labels)

    print(f"Predicted probabilities:\n{pred_probs}")
    print(f"True labels (one-hot):\n{true_labels}")
    print(f"Cross-Entropy Loss: {ce_loss:.4f}")


def weight_initialization():
    """Demonstrate weight initialization strategies."""
    print("\n" + "=" * 60)
    print("WEIGHT INITIALIZATION")
    print("=" * 60)

    print("\n1. Random Normal (Gaussian)")
    W_normal = Tensor.random_normal((3, 4), mean=0.0, std=0.1)
    print(f"Normal initialization (3x4, μ=0, σ=0.1):\n{W_normal}")

    print("\n2. Random Uniform")
    W_uniform = Tensor.random_uniform((3, 4), low=-0.1, high=0.1)
    print(f"Uniform initialization (3x4, range=[-0.1, 0.1]):\n{W_uniform}")

    print("\n3. Xavier/Glorot Initialization (simplified)")
    import math
    fan_in, fan_out = 3, 4
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    W_xavier = Tensor.random_uniform((fan_in, fan_out), low=-limit, high=limit)
    print(f"Xavier initialization (3x4):\n{W_xavier}")


def gradient_clipping():
    """Demonstrate gradient clipping."""
    print("\n" + "=" * 60)
    print("GRADIENT CLIPPING")
    print("=" * 60)

    print("\nPrevents exploding gradients in training")

    gradients = Tensor([[-5.0, 2.0], [8.0, -3.0]])
    print(f"Original gradients:\n{gradients}")

    clipped = gradients.clip(-2.0, 2.0)
    print(f"After clipping to [-2, 2]:\n{clipped}")


def main():
    """Run all tensor operation demos."""
    basic_tensor_ops()
    activation_functions()
    neural_network_layer()
    loss_functions()
    weight_initialization()
    gradient_clipping()

    print("\n" + "=" * 60)
    print("Tensor Operations Tutorial Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Tensors are multi-dimensional arrays")
    print("- Activation functions introduce non-linearity")
    print("- Loss functions measure prediction error")
    print("- Proper initialization is crucial for training")
    print("- These are the building blocks of deep learning!")


if __name__ == '__main__':
    main()
