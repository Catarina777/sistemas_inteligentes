import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
	"""
	The StackingClassifier is an ensemble model that uses a set of models to generate predictions.
	These predictions are then used to train the final model. The final model can then be used to predict the output variable (Y).
	"""

	def __init__(self, models: list, final_model: callable):
		"""
		Initializes the StackingClassifier.

		Parameters
		----------
		models: List of initialized models of classifiers
		final_model: Final model classifier
		"""
		# parameters
		self.models = models
		self.final_model = final_model

	def fit(self, dataset: Dataset) -> 'StackingClassifier':
		"""
		Fit the models to the dataset.

		Parameters
		------------
		dataset: Dataset object to fit the models to.

		Returns
		------------
		self: StackingClassifier
		"""
		# trains the models
		for model in self.models:
			model.fit(dataset)

		# gets the models predictions
		predictions = []
		for model in self.models:
			predictions.append(model.predict(dataset))

		# the predictions from the previous ensemble models will be used as new_features to fit the final_model
		self.final_model.fit(Dataset(dataset.x, np.array(predictions).T))
		return self

	def predict(self, dataset: Dataset) -> np.array:
		"""
		Computes the prevision of all the models and returns the final model prediction.

		Parameters
		------------
		dataset: Dataset object to predict the labels of.

		Returnss
		------------
		The final model prediction
		"""
		# gets the model predictions
		predictions = []
		for model in self.models:
			predictions.append(model.predict(dataset))

		# gets the final model previsions
		y_pred = self.final_model.predict(Dataset(dataset.x, np.array(predictions).T))

		return y_pred

	def score(self, dataset: Dataset) -> float:
		"""
		Calculates the accuracy of the model.

		Returns
		------------
		Accuracy of the model.
		"""
		y_pred = self.predict(dataset)
		score = accuracy(dataset.y, y_pred)

		return score
