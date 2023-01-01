import numpy as np


def euclidean_distance(x: np.array, y: np.ndarray) -> np.ndarray:
    """
    Calculates the euclidean distance between two points.

    distance_y1n = sqrt((x1 - y11)^2 + (x2 - y12)^2) + ... + sqrt((xn - y1n)^2)
    distance_y2n = sqrt((x1 - y21)^2 + (x2 - y22)^2) + ... + sqrt((xn - y2n)^2)

	Parameters
    ------------
	x: Vector of points.
	y: Multiple vectors of points.

	Returns
    ------------
    A np.ndarray of euclidean distance.
    """

    return np.sqrt(np.sum((x - y) ** 2, axis=1))
