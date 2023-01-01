from ctypes import Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:
	"""
	The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
	a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
	looking at the classes of the k-nearest samples in the training data.
	"""
	def __init__(self, k: int, distance: callable = euclidean_distance):
		"""
		Initialize the KNN classifier object.

		Parameters
		------------
		k: The number of nearest neighbors to use
		distance: The distance function to use

		Attributes
		------------
		dataset: The training data
		"""
		self.k = k
		# calculates the distance of one sample to others that are present in the dataset
		self.distance = distance
		self.dataset = None

	def fit(self, dataset: Dataset):
		"""
		Stores the dataset.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Dataset
		"""
		self.dataset = dataset
		return self

	def _get_closet_label(self, x: np.ndarray):
		"""
		Predicts the class with the highest frequency.

		Parameters
		------------
		x: Sample.

		Returns
		------------
		Indexes of the classes with the highest frequency.
		"""
		# Calculates the distance between the samples and the dataset
		distances = self.distance(x, self.dataset.x)

		# Sort the distances and get indexes
		# get the first k indexes of the sorted distances array
		knn = np.argsort(distances)[:self.k]
		knn_labels = self.dataset.y[knn]

		# Returns the unique classes and the number of occurrences from the matching classes
		labels, counts = np.unique(knn_labels, return_counts=True)

		# Gets the most frequent class
		# get the indexes of the classes with the highest frequency/count
		high_freq_lab = labels[np.argmax(counts)]
		return high_freq_lab

	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts the class with the highest frequency.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Class with the highest frequency.
		"""
		# axis=1 means that we want to apply the distance function to each sample of the dataset
		return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

	def score(self, dataset: Dataset) -> float:
		"""
		Returns the accuracy of the model.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Accuracy.
		"""
		predictions = self.predict(dataset)
		# Returns the number of correct predictions divided by the total number of predictions (accuracy)
		return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
