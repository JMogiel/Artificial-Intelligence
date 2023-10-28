import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List


def discretize_dataframe(data: pd.DataFrame, attributes: List[str], bins: int = 6) -> pd.DataFrame:
    """
    Discretizes the continuous attributes of the given DataFrame.

    :param
        df: A DataFrame containing the continuous attributes.
        attributes: A list of the continuous attributes to discretize.
        bins: The number of bins to use for discretization.
    :return: A DataFrame with the continuous attributes discretized.
    """
    data_copy = data.copy()
    #print(f"Data_copy:{data_copy}")
    for attribute in attributes:
        data_copy[attribute] = pd.cut(data[attribute], bins, labels=False)
    return data_copy


def load_and_prepare_data(file_path: str, test_size: float, val_size: float) -> tuple:
    """
    Loads the dataset from the given file path, preprocesses it, and splits it into train, validation, and test sets.

    :param file_path: The path to the dataset file.
    :param test_size: The proportion of the dataset to include in the test split.
    :param val_size: The proportion of the dataset to include in the validation split.
    :return: A tuple containing the train, validation, and test sets (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    data = pd.read_csv(file_path, delimiter=";")
    attributes_to_discretize = ["age", "weight", "height", "ap_hi", "ap_lo"]
    discretized_data = discretize_dataframe(data, attributes_to_discretize)
    X = discretized_data.drop(columns=["id", "cardio"])
    #print(f"X:{X}")
    y = discretized_data["cardio"]
    #print(f"y:{y}")
    #print(discretized_data)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=23, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=23, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
