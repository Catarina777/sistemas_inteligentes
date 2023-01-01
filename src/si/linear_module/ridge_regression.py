from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

alpha_options = Literal['static_alpha', 'half_alpha']


class RidgeRegression:
	"""
	The RidgeRegression is a linear model using the L2 regularization.
	This model solves the linear regression problem using an adapted Gradient Descent technique.
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

		Attributes
		------------
		theta: The model parameters, namely the coefficients of the linear model. For example, x0 * theta[0] + x1 * theta[1] + ...
		theta_zero: The model parameter, namely the intercept of the linear model.
		cost_history: The history of the cost function of the model.
		"""
		# parameters
		self.use_adaptive_alpha = use_adaptive_alpha
		self.l2_penalty = l2_penalty
		# learning rate, set as a low value to not jump over the minimum
		self.alpha = alpha
		self.max_iter = max_iter

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

		# predicted y values
		# corresponds to the classical function of y = mx + b
		y_pred = np.dot(dataset.x, self.theta) + self.theta_zero

		# computing and updating the gradient with the learning rate
		# alpha is normalized to be the same size as the dataset
		gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.x)
		# computing the penalty
		# prevents ajustments
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
		# size = number of features
		self.theta = np.zeros(n)
		self.theta_zero = 0

		# gradient descent
		for i in range(self.max_iter):
			self.gradient_descent(dataset, m)
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)
			# checks if there is any difference between the cost of the previous and the current iteration
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

		# gradient descent
		for i in range(self.max_iter):
			self.gradient_descent(dataset, m)
			# creats the dictionary keys
			self.cost_history[i] = self.cost(dataset)

			if i != 0 and self.cost_history[i-1] - self.cost_history[i] < 1:
				# updating the learning rate
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

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")
