import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR input and output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

# Learning rate
learning_rate = 0.1

# Training loop
for epoch in range(5000):

    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Compute loss (MSE)
    loss = np.mean((y - y_pred) ** 2)

    # Backpropagation
    d_y_pred = (y_pred - y) * sigmoid_derivative(y_pred)
    dW2 = np.dot(a1.T, d_y_pred)
    db2 = np.sum(d_y_pred, axis=0, keepdims=True)

    d_a1 = np.dot(d_y_pred, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss periodically
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("\nFinal Predictions:")
print(y_pred)
