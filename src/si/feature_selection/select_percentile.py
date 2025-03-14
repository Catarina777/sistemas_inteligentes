import numpy as np

from si.data.dataset import Dataset


class SelectPercentile:
	"""
	Select features with the highest scores according to the given percentile.
	Feature ranking is performed by computing the scores of each feature using a scoring function:
		- f_classification: ANOVA F-value between label/feature for classification tasks.
	"""

	def __init__(self, score_func, percentile=25):
		"""
		Initializes the select percentile.

		Parameters
		------------
		score_func: Function taking dataset and returning a pair of arrays (scores, p_values)
		percentile: The percentile of features to select.
		"""
		self.score_func = score_func
		self.percentile = percentile
		self.f_value = None
		self.p_value = None

	def fit(self, dataset):
		"""
		Calculates the score for each feature.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		SelectPercentile object.
		"""
		# calculates f_value an p_value according to the function
		self.f_value, self.p_value = self.score_func(dataset)

		return self

	def transform(self, dataset):
		"""
		Selects the percentile best features with the highest score until the percentile is reached.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Dataset object.
		"""
		if self.f_value is None:
			raise Exception("You must fit the select percentile before transform the dataset.")

		# calculates the number of features to be selected based on the percentile value
		features_percentile = int(len(dataset.features_names) * self.percentile / 100)
		indexes = np.argsort(self.f_value)[-features_percentile:]
		best_features = dataset.x[:, indexes]
		best_features_names = [dataset.features_names[i] for i in indexes]

		return Dataset(best_features, dataset.y, best_features_names, dataset.label_name)

	def fit_transform(self, dataset):
		"""
		Calculates the score for each feature and selects the percentile best features.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Dataset object.
		"""
		self.fit(dataset)
		return self.transform(dataset)
