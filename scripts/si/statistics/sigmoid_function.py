import numpy as np


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.

	Parameters
    ------------
	x: array of samples.

    Returns
    ------------
	Array of sigmoid values
    """
    return 1 / (1 + np.exp(-x))
