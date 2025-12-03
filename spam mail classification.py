import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class FeedForward2HL:
    def __init__(self):
        self.W1 = np.array([0.1, 0.5, 0.3])
        self.W2 = np.array([0.3, 0.7, 0.6])


    def forward(self, x):
        # Hidden layer 1
        z1 = np.dot(x, self.W1)
        h1 = relu(z1)

        # Hidden layer 2
        z2 = np.dot(x, self.W2) * h1
        h2 = sigmoid(z2)

        # Output layer
        output = sigmoid(h2)

        return output


# Run
x = np.array([1, 0, 1])
model = FeedForward2HL()
result = model.forward(x)

print("Final Output:", result)

if result >= 0.5:
    print("Spam")
else:
    print("Not Spam")
