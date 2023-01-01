import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the cross-entropy loss function.
    It's calculated by the difference of two probabilities, the true values and the predicted ones.
    *Entropy is the information of a message, in this context, of a variable.

	Parameters
    ------------
    y_true: The true labels of the dataset.
    y_pred: The predicted labels of the dataset.

	Returns
    ------------
    The cross-entropy loss function.
    """
    return - np.sum(y_true) * np.log(y_pred) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the derivative of the cross-entropy loss function.

	Parameters
    ------------
	y_true: The true labels of the dataset.
    y_pred: The predicted labels of the dataset.

    Returns
    ------------
	The derivative of the cross-entropy loss function
    """

    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)
