from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression:
	"""
	The Logistic Regression model is a linear model that is used for classification problems.

	Attributes
	------------
	theta: np.ndarray
		The model parameters'
	theta_zero: float
		The model bias
	cost_history: dict
		The cost history of the model
	"""
	def __init__(self, use_adaptive_alpha: bool = False, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
		"""
		Initializes the LogisticRegression object.

		Parameters
		-----------
		use_adaptive_alpha: Is used or not in the gradient descent, which implies the use of different fit methods
		l2_penalty: L2 regularization parameter
		alpha: learning rate
		max_iter: maximum number of iterations

		"""
		# parameters
		self.use_adaptive_alpha = use_adaptive_alpha
		self.l2_penalty = l2_penalty
		self.alpha = alpha
		self.max_iter = max_iter

		# attributes
		self.theta = None
		self.theta_zero = None
		self.cost_history = {}

	def gradient_descent(self, dataset: Dataset, m: int) -> None:
		"""
		Computes the gradient descent of the model

		Parameters
		------------
		dataset: The dataset to compute the gradient descent on.
		m: Number of examples

		Returns
		------------
		The gradient descent of the model
		"""
		# predicted y
		y_pred = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

		# apply sigmoid function
		y_pred = sigmoid_function(y_pred)

		# computed the gradient descent and updates with the learning rate
		gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)

		# computing the penalty
		penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

		# updating the model parameters
		self.theta = self.theta - gradient - penalization_term
		self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

	def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
		"""
		Executes the gradient descent algorithm, that must stop when the value of the cost function doesn't change.
		When the difference between the cost of the previous and the current iteration is less than 0.0001, the Gradient Descent must stop.
		Fits the model without updating the learning rate (alpha).

		Parameters
		------------
		dataset: The dataset to fit the model on

		Returns
		------------
		Self.
		"""
		m, n = dataset.shape()
		# initialize the model parameters
		self.theta = np.zeros(n)
		self.theta_zero = 0

		for i in range(int(self.max_iter)):
			self.gradient_descent(dataset, m) #gradient descent
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)

			if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001:
				break
		return self

	def _adaptive_fit(self, dataset: Dataset) -> 'LogisticRegression':
		"""
		Executes the gradient descent algorithm, that must decrease the alpha value when the value of the cost function doesn't change.
		When the difference between the cost of the previous and the current iteration is less than 0.0001.
		Fits the model updating the learning rate (alpha).
		Returns self.

		Parameters
		------------
		dataset: The dataset to fit the model on

		Returns
		------------
		Self.
		"""
		m, n = dataset.shape()

		# initialize the model parameters
		self.theta = np.zeros(n)
		self.theta_zero = 0

		for i in range(int(self.max_iter)):
			self.gradient_descent(dataset, m) #gradient descent
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)

			if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001:
				self.alpha = self.alpha/2

		return self

	def fit(self, dataset: Dataset) -> 'LogisticRegression':
		"""
		Fits the model to the dataset

		Parameters
		------------
		dataset: The dataset to fit the model on.

		Returns
		------------
		A Logistic Regression object of the fitted model
		"""
		if self.use_adaptive_alpha:
			return self._adaptive_fit(dataset)
		else:
			return self._regular_fit(dataset)

	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts the dataset and converts them to binary.

		Returns
		------------
		A vector of predictions
		"""
		predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)

		# mask for the predictions that are greater than 0.5
		mask = predictions >= 0.5
		predictions[mask] = 1
		predictions[~mask] = 0
		return predictions

	def score(self, dataset: Dataset):
		"""
		Computes the accuracy of the model

		Parameters
		------------
		dataset: The dataset to compute the score function on.

		Returns
		------------
		The accuracy of the model
		"""
		y_pred = self.predict(dataset)
		return accuracy(dataset.y, y_pred)

	def cost(self, dataset: Dataset) -> float:
		"""
		Computes the cost of the model

		Parameters
		------------
		dataset: The dataset to compute the cost function on.

		Returns
		------------
		The cost of the model
		"""
		sample =  dataset.shape()[0]

		predictions = sigmoid_function(np.dot(dataset.x, self.theta) + self.theta_zero)
		cost = (- dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
		cost = np.sum(cost) / sample
		# regularization term
		cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * sample))
		return cost

	def cost_function_plot(self):
		"""
		Plots the cost function history of the model

		Returns
		------------
		None
		"""
		import matplotlib.pyplot as plt

		iter = list(self.cost_history.keys())
		val = list(self.cost_history.values())

		plt.plot(iter, val, '-r')
		plt.xlabel('Iteration')
		plt.ylabel('Cost')
		plt.show()
