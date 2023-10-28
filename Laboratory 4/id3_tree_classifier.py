import numpy as np
import pandas as pd
from typing import Dict, Any, Union


class ID3TreeClassifier:
    def __init__(self, max_depth: int = 12):
        """
        Initializes the ID3TreeClassifier with the specified maximum depth.

        :param max_depth: The maximum depth of the decision tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, data: pd.DataFrame) -> float:
        """
        Calculates the entropy of the given data.

        :param data: A DataFrame containing the data with 'class' column.
        :return: The entropy value.
        """
        class_counts = data["class"].value_counts().values
        #print(data["class"])
        #print(f"Class_counts:{class_counts}")
        probabilities = class_counts / np.sum(class_counts)
        # print(f"Probabilities:{probabilities}")
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, data: pd.DataFrame, attribute: str) -> float:
        """
        Calculates the information gain of an attribute with respect to the given data.

        :param data: A DataFrame containing the data with 'class' column.
        :param attribute: The attribute for which the information gain will be calculated.
        :return: The information gain value.
        """
        original_entropy = self._entropy(data)

        values, counts = np.unique(data[attribute], return_counts=True)
        weighted_entropy = np.sum(
            [
                (counts[i] / np.sum(counts))
                * self._entropy(data.where(data[attribute] == v).dropna())
                for i, v in enumerate(values)
            ]
        )

        return original_entropy - weighted_entropy

    def _build_tree(self, data: pd.DataFrame, depth: int) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Recursively builds the decision tree based on the given data and depth.

        :param data: A DataFrame containing the data with 'class' column.
        :param depth: The current depth of the tree.
        :return: A dictionary representing the decision tree.
        """
        if depth >= self.max_depth or len(np.unique(data["class"])) == 1:
            return {"leaf": data["class"].mode()[0]}

        best_attribute = max(data.columns[:-1], key=lambda x: self._information_gain(data, x))
        #print(f"Level {depth}: Best attribute is {best_attribute}")
        tree = {"attribute": best_attribute}
        #print(f"Best attribute:{best_attribute}")
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute] == value].drop(columns=[best_attribute])
            tree[value] = self._build_tree(sub_data, depth + 1)

        #print(f"Tree:{tree}")
        return tree

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the decision tree classifier to the training data.

        Args:
            X: Training data features
            y: Training data target labels
        """
        data = X.copy()
        #print(f"Data:{data}")
        data["class"] = y
        self.most_common_class = y.mode()[0]
        self.tree = self._build_tree(data, 0)

    def _classify(self, row: pd.Series, tree: Dict[str, Union[str, Dict[str, Any]]]) -> Any:
        """
        Classify a single data point using the decision tree.

        Args:
            row: Data point to classify
            tree: Decision tree used for classification

        Returns:
            The predicted class label
        """
        if "leaf" in tree:
            return tree["leaf"]

        attribute_value = row[tree["attribute"]]
        if attribute_value in tree:
            return self._classify(row, tree[attribute_value])
        else:
            return self.most_common_class

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the class attribute for each row in the given feature set.

        :param X: A DataFrame containing the feature set.
        :return: A Series containing the predicted class attribute for each row in X.
        """
        return X.apply(lambda row: self._classify(row, self.tree), axis=1)
