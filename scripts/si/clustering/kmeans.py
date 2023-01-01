import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class Kmeans:
	"""
	It performs k-means clustering on the dataset.
	It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
	It returns the centroids and the indexes of the closest centroid for each point.
	"""
	def __init__(self, k, max_iter: int = 100, distance: euclidean_distance = euclidean_distance):
		"""
		It initializes the K-means clustering algorithm.

		Parameters
		------------
		k: Number of clusters.
		max_iter: Maximum number of iterations.
		distance: Distance function.

		Attributes
		------------
		centroids: Centroids of the clusters.
		labels: Labels of the clusters.
		"""
		self.k = k
		self.max_iter = max_iter
		self.distance = distance
		self.centroids = None
		self.labels = None


	def _init_centroids(self, dataset: Dataset):
		"""
		Initializes the centroids.

		Parameters
		------------
		dataset: Dataset object.
		"""
		# randomly selects k samples from the dataset
		seed = np.random.permutation(dataset.x.shape[0])[:self.k]
		# uses them as centroids and at the beginning each centroid has only one sample
		self.centroids = dataset.x[seed, :]


	def _get_closest_centroid(self, x: np.ndarray) -> np.ndarray:
		"""
		Gets the index of the closest centroid for a given sample.

		Parameters
		------------
		x: Sample.

		Returns
		------------
		Index of the closest centroid.
		"""
		# calculates the distance between the sample and each centroid
		distance = self.distance(x, self.centroids)
		# gets the index of the centroid with the minimum distance for each sample
		closest_centroid_ind = np.argmin(distance, axis=0)

		return closest_centroid_ind


	def fit(self, dataset: Dataset) -> 'Kmeans':
		"""
		Calculates the score for each feature.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		SelectKBest object.
		"""
		# generate initial centroids
		self._init_centroids(dataset)

		# indicates if the algorithm has converged
		convergence = False
		# iteration counter
		i = 0
		# stores the labels of each sample
		labels = np.zeros(dataset.shape()[0])

		# while the algorithm has not converged and the maximum number of iterations has not been reached
		while not convergence and i < self.max_iter:
			# get closest centroid (apply function along each sample (axis=1))
			new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)
			# compute the new centroids
			centroids = []
			for j in range(self.k):
				# groups the samples according to the centroid they belong and calculates the mean over all samples that are in a centroid
				centroid = np.mean(dataset.x[new_labels == j], axis=0)
				centroids.append(centroid)

			self.centroids = np.array(centroids)
			# check if ther is conversion (convergence=True when there is no difference)
			convergence = np.any(new_labels != labels)
			# replace labels
			labels = new_labels
			i += 1

		self.labels = labels
		return self


	def _get_distance(self, x: np.ndarray) -> np.ndarray:
		"""
		Calculates the distance between two samples.

		Parameters
		------------
		x: Sample.

		Returns
		------------
		Distances between each sample and the closest centroid.
		"""
		return self.distance(x, self.centroids)


	def transform(self, dataset: Dataset) -> np.ndarray:
		"""
		Selects the k best features.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Transformed dataset.
		"""
		if self.labels is None:
			raise Exception("You must fit the select k best before transforming the dataset.")
		centroid_distance = np.apply_along_axis(self._get_distance, axis=1, arr=dataset.x)

		return centroid_distance


	def fit_transform(self, dataset: Dataset) -> np.ndarray:
		"""
		Fits the model and transforms the dataset.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		-----------
		Transformed dataset.
		"""
		self.fit(dataset)
		return self.transform(dataset)


	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts the label of a given sample.

		Parameters
		------------
		dataset: Dataset object.

		Returns
		------------
		Predicted labels.
		"""
		return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)

	def fit_predict(self, dataset: Dataset) -> np.ndarray:
		"""
		It fits and predicts the labels of the dataset

		Parameters
		------------
		dataset: Dataset object
		"""
		self.fit(dataset)
		return self.predict(dataset)

if __name__ == '__main__':
	from si.data.dataset import Dataset
	dataset = Dataset.from_random(100, 5)

	k_ = 3
	kmeans = Kmeans(k_)
	res = kmeans.fit_transform(dataset)
	predictions = kmeans.predict(dataset)
	print(res.shape)
	print(predictions.shape)
