from cmath import sqrt

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculated the Root Mean Squared error.

	Parameters
    ------------
	y_true: An array of true labels.
    y_pred: An array of predicted labels.

    Parameters
    ------------
	The RMSE value.
    """
    rmse = sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
    return rmse
