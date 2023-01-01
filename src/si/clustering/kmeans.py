import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class Kmeans:
    def __init__(self, k, max_iter: int = 100, distance: euclidean_distance = euclidean_distance):
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
        # uses them as centroids
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
        self._init_centroids(dataset)

		# indicates if the algorithm has converged
        convergence = False
		# iteration counter
        i = 0
		# stores the labels of each sample
        labels = np.zeros(dataset.shape()[0])

		# while the algorithm has not converged and the maximum number of iterations has not been reached
        while not convergence and i < self.max_iter:

            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)

            centroids = []

            for j in range(self.k):
                centroid = np.mean(dataset.x[new_labels == j], axis=0)
                centroids.append(centroid)

            self.centroids = np.array(centroids)

            convergence = np.any(new_labels != labels)
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
            raise Exception("You must fit the select k best before transform the dataset.")

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
