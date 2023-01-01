from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

alpha_options = Literal['static_alpha', 'half_alpha']


class RidgeRegression:
	"""
	The RidgeRegression is a linear model using the L2 regularization.
	This model solves the linear regression problem using an adapted Gradient Descent technique.

	Attributes
	------------
	theta: np.array
		The model parameters, namely the coefficients of the linear model.
		For example, x0 * theta[0] + x1 * theta[1] + ...
	theta_zero: float
		The model parameter, namely the intercept of the linear model.
		For example, theta_zero * 1
	cost_history: dict
		The history of the cost function of the model.
	"""

	def __init__(self, use_adaptive_alpha: bool = True, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
		"""
		Initializes the RidgeRegression object.

		Parameters
		------------
		use_adaptive_alpha: Is used or not in the gradient descent, which implies the use of different fit methods
		l2_penalty: L2 regularization parameter
		alpha: Learning rate
		max_iter: Maximum number of iterations
		"""
		# parameters
		self.use_adaptive_alpha = use_adaptive_alpha
		self.l2_penalty = l2_penalty
		# learning rate, set as a low value to not jump over the minimum
		self.alpha = alpha
		self.max_iter = max_iter

		# attributes
		# model coefficient
		self.theta = None
		# f function of a linear model
		self.theta_zero = None
		# history of the cost function
		self.cost_history = {}

	def gradient_descent(self, dataset: Dataset, m):
		"""
		Computes the gradient descent of the model

		Parameters
		------------
		dataset: The dataset to compute the gradient descent on.

		Returns
		------------
		The gradient descent of the model
		"""

		# predicted y
		y_pred = np.dot(dataset.x, self.theta) + self.theta_zero  # corresponds to the classical function of
		# y = mx + b

		# computing and updating the gradient with the learning rate
		gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)  # calculates the
		# gradient of the cost function
		# np.dot sums the colum values of the multiplication arrays
		# learning rate is multiplicated by 1/m to normalize the rate to the dataset size

		# computing the penalty
		penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

		# updating the model parameters
		self.theta = self.theta - gradient - penalization_term
		self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

	def _regular_fit(self, dataset: Dataset) -> 'RidgeRegression':
		"""
		Executes the gradient descent algorithm, that must stop when the value of the cost function doesn't change.
		When the difference between the cost of the previous and the current iteration is less than 1, the Gradient Descent must stop.
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

		for i in range(self.max_iter):
			self.gradient_descent(dataset, m) #gradient descent
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)

			if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 1:
				break

		return self


	def _adaptive_fit(self, dataset: Dataset) -> 'RidgeRegression':
		"""
		Executes the gradient descent algorithm, that must decrease the alpha value when the value of the cost function doesn't change.
		When the difference between the cost of the previous and the current iteration is less than 1.
		Fits the model updating the learning rate (alpha).

		Parameters
		----------
		dataset: The dataset to fit the model to

		Returns
		------------
		Self.
		"""
		m, n = dataset.shape()

		# initialize the model parameters
		self.theta = np.zeros(n)
		self.theta_zero = 0

		for i in range(self.max_iter):
			self.gradient_descent(dataset, m) #gradient descent
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)

			if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 1:
				self.alpha = self.alpha/2
		return self

	def fit(self, dataset: Dataset) -> 'RidgeRegression':
		"""
		Fit the model to the dataset

		Parameters
		------------
		dataset: The dataset to fit the model to

		Returns
		------------
		The fitted model
		"""
		if self.use_adaptive_alpha:
			return self._adaptive_fit(dataset)
		else:
			return self._regular_fit(dataset)

	def predict(self, dataset: Dataset) -> np.array:
		"""
		Predict the output of the dataset

		Parameters
		------------
		dataset: The dataset to predict the output of the dataset

		Returns
		------------
		The predictions of the dataset
		"""
		return np.dot(dataset.x, self.theta) + self.theta_zero

	def score(self, dataset: Dataset) -> float:
		"""
		Compute the mean square error of the model on the dataset

		Parameters
		------------
		dataset: The dataset to compute the MSE on

		Returns
		------------
		Mean Square Error of the model
		"""
		y_pred = self.predict(dataset)
		return mse(dataset.y, y_pred)

	def cost(self, dataset: Dataset) -> float:
		"""
		Compute the cost function (J function) of the model on the dataset using L2 regularization

		Parameters
		------------
		dataset: The dataset to compute the cost function on

		Returns
		------------
		The cost function of the model
		"""
		y_pred = self.predict(dataset)

		cost_function = (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

		return cost_function

