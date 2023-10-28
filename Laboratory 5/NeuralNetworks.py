import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
import time
import seaborn as sns
import random

# Preparing the Data
start_time = time.time()
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

y_series = pd.Series(y)
print(y_series.value_counts())

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Dataset size: {X.shape[0]} images")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
encoder = OneHotEncoder(sparse=False, categories='auto')
encoder.fit(y.to_numpy().reshape(-1, 1))  # Convert y to numpy array

print(f"Train set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"Test set: {len(X_test)} images")


# Implementing the MLP
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self.weights = []
        self.biases = []

        # Weights and biases for hidden layers
        for i in range(self.num_layers - 1):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]
            output_dim = self.hidden_sizes[i]
            self.weights.append(np.random.randn(input_dim, output_dim))
            self.biases.append(np.zeros(output_dim))

        # Weights and biases for output layer
        self.weights.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        self.biases.append(np.zeros(self.output_size))

    def forward(self, X):
        self.layer_outputs = []
        input_data = X

        # Forward pass through hidden layers
        for i in range(self.num_layers - 1):
            layer_input = np.dot(input_data, self.weights[i]) + self.biases[i]
            layer_output = self.relu(layer_input)
            self.layer_outputs.append(layer_output)
            input_data = layer_output

        # Forward pass through output layer
        output = np.dot(input_data, self.weights[-1]) + self.biases[-1]
        output = self.softmax(output)
        self.layer_outputs.append(output)
        return output

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        one_hot_y = self.one_hot_encode(y)

        deltas = [0] * self.num_layers

        # Backward pass through output layer
        deltas[-1] = self.layer_outputs[-1] - one_hot_y

        # Backward pass through hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * self.relu_derivative(self.layer_outputs[i])

        # Update weights and biases
        for i in range(self.num_layers - 1, -1, -1):
            if i == 0:
                layer_input = X
            else:
                layer_input = self.layer_outputs[i - 1]
            weight_gradients = np.dot(layer_input.T, deltas[i]) / m
            bias_gradients = np.mean(deltas[i], axis=0)
            self.weights[i] -= learning_rate * weight_gradients
            self.biases[i] -= learning_rate * bias_gradients

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def softmax(self, x):
        max_val = np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(x - max_val)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def one_hot_encode(self, y):
        return encoder.transform(y.reshape(-1, 1))


# Training the MLP
print("Training the MLP...")
input_size = X_train.shape[1]
hidden_sizes = [128, 128]  # Number of neurons in each hidden layer
output_size = len(np.unique(y))  # Determine the number of output classes dynamically
learning_rate = 0.001
num_epochs = 315
batch_size = 32

print(f"Number of layers: Input: {input_size}, Hidden: {hidden_sizes}, Output: {output_size}")

mlp = MLP(input_size, hidden_sizes, output_size)

num_batches = X_train.shape[0] // batch_size
val_accuracies = []  # Store validation accuracies for each epoch

for epoch in range(num_epochs):
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        output = mlp.forward(X_batch)
        mlp.backward(X_batch, y_batch, learning_rate)

    # Validate the model after each epoch
    val_output = mlp.forward(X_val)
    val_predictions = np.argmax(val_output, axis=1)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")

# Plot number of epochs vs accuracy
plt.plot(range(1, num_epochs + 1), val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Number of Epochs vs Validation Accuracy')
plt.show()

# Evaluating the MLP
print("Evaluating the MLP on the test set...")
test_output = mlp.forward(X_test)
test_predictions = np.argmax(test_output, axis=1)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Compute the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.show()

# Print examples of mistaken predictions
incorrect_indices = np.nonzero(test_predictions != y_test.to_numpy())[0]
if incorrect_indices.size:
    chosen_index = random.choice(incorrect_indices)
    plt.imshow(X_test[chosen_index].reshape(28, 28), cmap='gray_r')
    plt.title(f"Predicted: {test_predictions[chosen_index]}, Actual: {y_test.to_numpy()[chosen_index]}")
    plt.show()
else:
    print("No mistakes were made in the predictions.")

execution_time = time.time() - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
