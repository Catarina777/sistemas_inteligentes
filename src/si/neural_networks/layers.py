import numpy as np
from si.statistics.sigmoid_function import sigmoid_function


class Dense:
	"""
	A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
	"""
	def __init__(self, input_size: int, output_size: int):
		"""
		Initialize the Dense object.

		Parameters
		------------
		input_size: The number of input nodes
		output_size: The number of output nodes

		Attributes
		------------
		weights: The weights matrix
		bias: A bias vector
		"""
		if input_size < 1:
			raise ValueError("input_size must be an integer greater than 1")

		if output_size < 1:
			raise ValueError("output_size must be an integer greater than 1")

		# parameters
		self.input_size = input_size
		self.output_size = output_size

		self.x = None
		# 0.01 is a hyperparameter to avoid exploding gradients
		self.weights = np.random.randn(input_size, output_size) * 0.01
		# each layer receives a weight that multiplies by the input that are then summed bias initialization, receives a bias to avoid overfitting
		self.bias = np.zeros((1, output_size))

	def forward(self, input_data: np.ndarray) -> np.ndarray:
		"""
		Computes the forward pass of the layer.

		Parameters
		------------
		x: input data value

		Returns
		------------
		Input data multiplied by the weights.
		"""
		self.x = input_data
		# the input_data needs to be a matrix with the same number of columns as the number of features
		# the number os columns of the input_data must be equal to the number of rows of the weights
		return np.dot(input_data, self.weights) + self.bias

	def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
		"""
		Computes the backward pass of the layer

		Parameters
		------------
		error: error value of the loss function
		learning_rate: learning rate

		Returns
		------------
		Error of the previous layer.
		"""
		error_to_propagate = np.dot(error, self.weights.T)
		# updates the weights and bias
		self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
		# sum because the bias has the dimension of nodes
		self.bias = self.bias - learning_rate * np.sum(error, axis = 0)
		return error_to_propagate


class SigmoidActivation:
	"""
	The Sigmoid Activation object implements an activation neural network based on the sigmoid function - sigmoid activation layer
	"""

	def __init__(self):
		# attribute
		self.x = None

	def forward(self, input_data: np.ndarray) -> np.ndarray:
		"""
		Computes the forward pass of the layer.

		Parameters
		------------
		input_data: input data

		Returns
		------------
		Input data multiplied by the weights.
		"""
		self.X = input_data
		return sigmoid_function(input_data)

	def backward(self, error: np.ndarray) -> np.ndarray:
		"""
		Computes the backward pass of the layer.

		Returns
		------------
		Error of the previous layer.
		"""
		# multiplication of each element by the derivative and not by the entire matrix
		sigmoid_derivative = sigmoid_function(self.x) * (1 - sigmoid_function(self.x))
		error_to_propagate = error * sigmoid_derivative
		return error_to_propagate


class SoftMaxActivation:
	"""
	The SoftMax Activation object implements an activation neural network based on the probability of occurrence of each class - softmax activation layer
	This layer is applied to multiclass problems.
	"""
	def __init__(self):
		self.X = None

	def forward(self, input_data: np.ndarray) -> np.ndarray:
		"""
		Computes the probability of each class.

		Parameters
		------------
		input_data: input data

		Returns
		------------
		Probability of each class.
		"""
		self.X = input_data
		exp = np.exp(input_data - np.max(input_data))
		return  exp / np.sum(exp, axis = 1, keepdims = True)

	def backward(self, error: np.ndarray) -> np.ndarray:
		"""
		Computes the backward pass of the layer.

		Returns
		------------
		Error of the previous layer.
		"""
		S = self.forward(self.X)  # softmax

		# calculate the jacobian
		# first matrix by repeating S in rows
		S_vector = S.reshape(S.shape[0], 1)
		# second matrix by repeating S in columns (transposing the first matrix)
		S_matrix = np.tile(S_vector,S.shape[0])

		# calculate the jacobian derivative
		# multiplying them together element-wise
		softmax_derivative = np.diag(S) - (S_matrix * np.transpose(S_matrix))
		error_to_propagate = error * softmax_derivative
		return error_to_propagate


class ReLUActivation:
	"""
	The ReLUActivation object implements an activation neural network based on the rectified linear relationship - relu activation layer
	This layer considers only positive values.
	"""
	def __init__(self):
		self.X = None

	def forward(input_data: np.ndarray) -> np.ndarray:
		"""
		Computes the rectified linear relationship.

		Parameters
		------------
		input_data: input data

		Returns
		------------
		Rectified linear relationship.
		"""
		# maximum between 0 and the input_data, the 0 is to avoid negative values
		self.X = input_data
		return  np.maximum(0, input_data)

	def backward(self, error: np.ndarray) -> np.ndarray:
		"""
		Computes the backwards pass of the rectified linear relationship.

		Returns
		------------
		Error of the previous layer.
		"""
		relu_derivative = np.where(self.x > 0, 1, 0)
		error_to_propagate = error * relu_derivative
		return error_to_propagate


class LinearActivation:
	def __init__(self):
		self.X = None

	def forward(self, input_data: np.ndarray) -> np.ndarray:
		"""
		Computes the linear relationship.

		Parameters
		------------
		input_data: input data

		Returns
		------------
		Linear relationship.
		"""
		self.X = input_data
		return input_data

	def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
		"""
		Computes the backward pass of the layer.

		Parameters
		------------
		error: The error propagated to the layer
		alpha: The learning rate of the model

		Returns
		------------
		Error of the previous layer.
		"""
		identity_derivative = np.ones_like(self.X)
		error_to_propagate = error * identity_derivative
		return error_to_propagate
