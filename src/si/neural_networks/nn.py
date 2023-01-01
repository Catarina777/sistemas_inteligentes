from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative


class NN:
	"""
	The NN is the Neural Network model.
	It comprehends the model topology including several neural network layers.
	The algorithm for fitting the model is based on backpropagation.
	"""

	def __init__(self, layers: list):
		"""
		Initialize the NN.

		Parameters
		------------
		layers: List of layers that are part of the neural network
		"""
		# parameters
		self.layers = layers

	def fit(self, dataset: Dataset) -> "NN":
		"""
		Trains the neural network.

		Parameters
		------------
		dataset: dataset to train the neural network

		Returns
		------------
		Trained neural network.
		"""
		# pointer to the input data, a more correct way would be to copy the data (x = dataset.x.copy())
		x = dataset.x
		for layer in self.layers:
			# if we were to use the dataset.x we would be using the original data, but we want to use the data that was already processed by the previous layer
			x = layer.forward(x)

		return self

	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts the classes of the dataset.

	Parameters
		------------
		dataset: dataset to predict the classes

		Returns
		------------
		Predicted classes.
		"""
		x = dataset.x
		# forward propagation
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def score(self, dataset: Dataset, score_func: Callable = accuracy) -> float:
		"""
		Returns the accuracy of the model.

		Parameters
		------------
		dataset: Dataset object.
		score_func: Function to calculate the score.

		Returns
		------------
		Accuracy.
		"""
		predictions = self.predict(dataset)
		return score_func(dataset.y, predictions)
