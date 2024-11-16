import numpy as np
import matplotlib.pyplot as plt

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    def forward(self, X):
        self.hidden_output = sigmoid(np.dot(X, self.weights_input_hidden))
        return sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        delta_output = (y - output) * sigmoid_derivative(output)
        delta_hidden = delta_output.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, learning_rate=0.1, epochs=10000)

# Make predictions
predictions = nn.predict(X)

# Plot the XOR dataset and predictions
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', label='XOR Data')
plt.scatter(X[:, 0], X[:, 1], c=np.round(predictions).flatten(), cmap='plasma', marker='x', s=200, label='Predictions')
plt.title('XOR Dataset and Predictions')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.show()

# Print predictions and actual values
for i in range(len(X)):
    print(f"Input: {X[i]}, Actual: {y[i][0]}, Predicted: {np.round(predictions[i][0])}")