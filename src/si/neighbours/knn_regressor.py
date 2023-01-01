from ctypes import Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
	"""
	The k-Nearst Neighbors Regressor is a machine learning model that estimates the mean value of the k most similar examples.
	"""

	def __init__(self, k: int, distance: euclidean_distance = euclidean_distance):
		"""
		Initialize the KNN Regressor.

		Parameters
		------------
		k: The number of nearest neighbors to use
		distance: The distance function to use

		Attributes
		------------
		dataset: The training data
		"""
		self.k = k
		self.distance = distance
		self.dataset = None

	def fit(self, dataset: Dataset) -> 'KNNRegressor':
		"""
		Stores the dataset.

		Parameters
		------------
		dataset: Dataset object

		Returns
		------------
		The dataset
		"""
		self.dataset = dataset
		return self

	def _get_closet_label(self, x: np.ndarray) -> Union[int, str]:
		"""
		Calculates the class with the highest frequency.

		Parameters
		------------
		x: Array of samples.

		Returns
		------------
		Indexes of the classes with the highest frequency
		"""

		# Calculates the distance between the samples and the dataset
		distances = self.distance(x, self.dataset.x)

		# Sort the distances and get indexes
		# get the first k indexes of the sorted distances array
		knn = np.argsort(distances)[:self.k]
		knn_labels = self.dataset.y[knn]

		# Computes the mean of the matching classes
		match_class_mean = np.mean(knn_labels)

		# Sorts the classes with the highest mean
		high_freq_class = np.argsort(match_class_mean)[:self.k]

		return high_freq_class

	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts the class with the highest frequency

		Returns
		------------
		Class with the highest frequency.
		"""
		return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

	def score(self, dataset: Dataset) -> float:
		"""
		Returns the accuracy of the model.

		Returns
		------------
		Accuracy of the model.
		"""
		predictions = self.predict(dataset)
		return rmse(dataset.y, predictions)
