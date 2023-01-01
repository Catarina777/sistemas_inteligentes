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

	def __init__(self, layers: list, epochs: int = 1000, learning_rate: float = 0.01, loss: callable = mse, loss_derivative: callable = mse_derivative, verbose: bool = False):
		"""
		Initialize the NN.

		Parameters
		------------
		layers: List of layers that are part of the neural network
		epochs: Number of epochs to train the model.
		learning_rate: The learning rate of the model.
		loss:The loss function to use.
		loss_derivative: The derivative of the loss function to use.
		verbose: Whether to print the loss at each epoch.

		Attributes
		------------
		history: The history of the model training.
		"""
		# parameters
		self.layers = layers
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.loss = loss
		self.loss_derivative = loss_derivative
		self.verbose = verbose

		self.history = {}

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
		for epoch in range(1, self.epochs + 1):

			y_pred = dataset.x.copy()
			y_true = np.reshape(dataset.y,(-1,1))

			# forward propagation
			for layer in self.layers:
				y_pred = layer.forward(y_pred)

			# backward propagation
			error = self.loss_derivative(y_true, y_pred)
			for layer in self.layers[::-1]:
				error = layer.backward(error, self.learning_rate)

			# save history
			cost = self.loss(y_true, y_pred)
			self.history[epoch] = cost

			# print loss
			if self.verbose:
				print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')


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

	def cost(self, dataset: Dataset) -> float:
		"""
		It computes the cost of the model on the given dataset

		Parameters
		------------
		dataset: The dataset to compute the cost on
		"""
		y_pred = self.predict(dataset)
		return self.loss(dataset.y, y_pred)

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
