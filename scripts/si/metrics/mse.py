import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

	Parameters
    ------------
	y_true: The true labels of the dataset.
    y_pred: The predicted labels of the dataset.

    Returns
    ------------
	The mean squared error of the model
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Mean Squared Error (MSE) function.

	Parameters
    ------------
	y_true: The true labels of the dataset.
    y_pred: The predicted labels of the dataset.

    Returns
    ------------
	The derivative of the MSE function.
    """
    return -2 * (y_true - y_pred) / (len(y_true) * 2)
