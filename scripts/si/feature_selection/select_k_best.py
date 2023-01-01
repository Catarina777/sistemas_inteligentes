import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest:
	"""
	Select features according to the k highest scores.
	Feature ranking is performed by computing the scores of each feature using a scoring function:
		- f_classification: ANOVA F-value between label/feature for classification tasks.
	"""
	def __init__(self, score_func, k):
		"""
		Initializes the select k best.

		Parameters
		------------
		score_func: Function that calculates the score for each feature.
		k: Number of features to select.

		Attributes
		------------
		f_value: F scores of features.
		p_value: p-values of F-scores.
		"""
		self.score_func = score_func
		self.k = k
		self.f_value = None
		self.p_value = None

	def fit(self, dataset: Dataset):
		"""
		Calculates the score for each feature.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		SelectKBest object.
		"""
		# calculates f_value an p_value according to the function
		self.f_value, self.p_value = self.score_func(dataset)
		return self

	def transform(self, dataset: Dataset) -> Dataset:
		"""
		Selects the k best features.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Dataset object.
		"""
		if self.f_value is None:
			raise Exception("You must fit the select k best before transform the dataset.")

		# get the indexes of the k best features, as the sorting is in ascending order, we get the last k indexes
		indexes = np.argsort(self.f_value)[-self.k:]
		best_features = dataset.x[:, indexes]
		best_features_names = [dataset.features_names[i] for i in indexes]

		return Dataset(best_features, dataset.y, best_features_names, dataset.label_name)

	def fit_transform(self, dataset: Dataset) -> Dataset:
		"""
		Calculates the score for each feature and selects the k best features.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Dataset object.
		"""
		self.fit(dataset)
		return self.transform(dataset)
