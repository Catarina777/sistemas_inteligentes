from ctypes import Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:
    def __init__(self, k: int, distance: euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        Stores the dataset.

		Parameters
    	------------
		dataset: Dataset object.

        Returns
    	------------
		Dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, x: np.ndarray):
        """
        Predicts the class with the highest frequency.

		Parameters
    	------------
		x: Sample.

        Returns
    	------------
		Indexes of the classes with the highest frequency.
        """
        # Calculates the distance between the samples and the dataset
        distances = self.distance(x, self.dataset.x)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]

        # Returns the unique classes and the number of occurrences from the matching classes
        labels, counts = np.unique(knn_labels, return_counts=True)

        # Gets the most frequent class
        high_freq_lab = labels[np.argmax(counts)]  # get the indexes of the classes with the highest frequency/count

        return high_freq_lab

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class with the highest frequency.

		Parameters
    	------------
		dataset: Dataset object.

        Returns
    	------------
		Class with the highest frequency.
        """

        # axis=1 means that we want to apply the distance function to each sample of the dataset
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.

		Parameters
    	------------
		dataset: Dataset object.

    	Returns
    	------------
		Accuracy.
        """
        predictions = self.predict(dataset)

        return accuracy(dataset.y, predictions)  # Returns the number of correct predictions divided
        # by the total number of predictions (accuracy)
        # The correct predictions are calculated by the predictions and the true values from the dataset
