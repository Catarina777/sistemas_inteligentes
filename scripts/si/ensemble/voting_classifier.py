import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
	"""
	The VotingClassifier is an ensemble model that uses voting as a combination function.
	The ensemble sets of models are based on the combination of predictions from several models.
	"""

	def __init__(self, models: list):
		"""
		Initializes the VotingClassifier.

		Parameters
		------------
		models: List of initialized models of classifiers
		"""
		# parameters
		self.models = models  # list of initialized models

	def fit(self, dataset: Dataset) -> 'VotingClassifier':
		"""
		Fit the model to the dataset

		Parameters
		------------
		dataset: Dataset object to fit the model to.

		Returns
		------------
		self: VotingClassifier
		"""
		for model in self.models:
			model.fit(dataset)
		return self

	def predict(self, dataset) -> np.ndarray:
		"""
		Combines the previsions of each model with a voting system.

		Parameters
		------------
		dataset: Dataset object to predict the labels of.

		Returns
		------------
		The most represented class
		"""
		predictions = np.array([model.predict(dataset) for model in self.models])

		def get_most_represented_class(pred: np.ndarray) -> int:
			"""
			Helper function, which returns the majority vote of the given predictions

			Parameters
			------------
			pred: Predictions for a certain sample to get the majority vote of
			"""
			# get the most common label and its counts
			labels, counts = np.unique(pred, return_counts=True)
			return labels[np.argmax(counts)]

		# apply a function that returns the most frequent label for each example
		return np.apply_along_axis(get_most_represented_class, axis=0, arr=predictions)

	def score(self, dataset: Dataset) -> float:
		"""
		Returns the accuracy of the model.

		Returns
		------------
		Accuracy of the model.
		"""
		y_pred = self.predict(dataset)
		score = accuracy(dataset.y, y_pred)

		return score

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))
