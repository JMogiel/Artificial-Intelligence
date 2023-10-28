import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probabilities = self.calculate_class_probabilities(y)
        self.feature_probabilities = self.calculate_feature_probabilities(X, y)

    def calculate_class_probabilities(self, y):
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        return class_probabilities

    def calculate_feature_probabilities(self, X, y):
        feature_probabilities = {}
        for c in self.classes:
            X_class = X[y == c]
            feature_probabilities[c] = {
                'mean': np.mean(X_class, axis=0),
                'std': np.std(X_class, axis=0)
            }
        return feature_probabilities

    def calculate_likelihood(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        likelihood = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
        return likelihood

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = []
            for c in self.classes:
                class_probability = self.class_probabilities[c]
                feature_probabilities = self.feature_probabilities[c]
                likelihoods = self.calculate_likelihood(x, feature_probabilities['mean'], feature_probabilities['std'])
                likelihoods = np.maximum(likelihoods, 1e-9)
                class_score = np.log(class_probability) + np.sum(np.log(likelihoods))
                class_scores.append(class_score)
            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)
        return predictions

def evaluate_model(X, y, method, param_range, num_repeats=20):
    results = []

    for param in param_range:
        accuracy_vals = []

        for _ in range(num_repeats):
            if method == 'split':
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=param, random_state=42)
            elif method == 'cv':
                naive_bayes = NaiveBayesClassifier()
                scores = cross_val_score(naive_bayes, X, y, cv=param, scoring='accuracy')
                accuracy_vals.append(np.mean(scores))
                continue
            else:
                raise ValueError("Invalid method. Use 'split' or 'cv'.")

            naive_bayes = NaiveBayesClassifier()
            naive_bayes.fit(X_train, y_train)
            y_pred_val = naive_bayes.predict(X_val)
            accuracy_val = accuracy_score(y_val, y_pred_val)
            accuracy_vals.append(accuracy_val)

        results.append(np.mean(accuracy_vals))

    return results

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Determine the size of the dataset
num_samples, num_features = X.shape
num_classes = len(data.target_names)

# Print dataset information
print(f"Dataset size: {num_samples} samples, {num_features} features")
print(f"Number of classes: {num_classes}")

# Method 1: Different divisions of training and validation set
split_sizes = [0.15, 0.25, 0.35, 0.4, 0.5, 0.6]  # Specific split sizes
accuracy_val_splits = evaluate_model(X, y, 'split', split_sizes)

# Method 2: Different values of k for cross-validation
k_values = np.arange(2, 11)  # Vary k from 2 to 10
accuracy_cv_means = evaluate_model(X, y, 'cv', k_values)

# Plotting
plt.figure(figsize=(10, 5))

# Method 1: Different divisions of training and validation set
plt.subplot(1, 2, 1)
plt.plot(split_sizes, accuracy_val_splits, marker='o')
plt.title("Accuracy vs. Train-Validation Split Size")
plt.xlabel("Split Size")
plt.ylabel("Accuracy")
plt.grid(True)

# Method 2: Different values of k for cross-validation
plt.subplot(1, 2, 2)
plt.plot(k_values, accuracy_cv_means, marker='o')
plt.fill_between(k_values, np.array(accuracy_cv_means) - np.array(accuracy_cv_stds),
                 np.array(accuracy_cv_means) + np.array(accuracy_cv_stds),
                 alpha=0.2)
plt.title("Accuracy vs. K-fold Cross-Validation")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.show()

# Print average accuracy values
print(f"Average Accuracy (Validation Set): {np.mean(accuracy_val_splits):.4f}")
print(f"Average Accuracy (Cross-validation): {np.mean(accuracy_cv_means):.4f}")
