import numpy as np

from si.data.dataset import Dataset

class PCA:

	def __init__(self, n_components: int = 2):
		"""
		Initializes the PCA.

		Parameters
        ------------
		n_components: Number of components to keep.
		"""
		self.n_components = n_components
		self.mean = None
		self.components = None
		self.explained_variance = None

	def get_centered_data(self, dataset: Dataset) -> np.ndarray:
		"""
		Centers the dataset.

		Parameters
        ------------
		dataset: Dataset object.

		Returns
        ------------
		A matrix with the centered data.
		"""
		# axis=0 means that we want to calculate the mean for each column
		self.mean = np.mean(dataset.x, axis=0)
		return dataset.x - self.mean

	def get_components(self, dataset: Dataset) -> np.ndarray:
		"""
		Calculates the components of the dataset.

		Parameters
        ------------
		dataset: Dataset object.

		Returns
        ------------
		A matrix with the components.
		"""

		# Get centered data
		centered_data = self.get_centered_data(dataset)

		# Get single value decomposition
		self.u_matrix, self.s_matrix, self.v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)

		# Get principal components. To do that we need to find the first n_compounds colums
		self.components = self.v_matrix_t[:, :self.n_components]

		return self.components

	def get_explained_variance(self, dataset: Dataset) -> np.ndarray:
		"""
		Calculates the explained variance.

		Parameters
        ------------
		dataset: Dataset object.

		Returns
        ------------
		A vector with the explained variance.
		"""
		# Get explained variance
		ev_formula = self.s_matrix ** 2 / (len(dataset.x) - 1)
		explained_variance = ev_formula[:self.n_components]

		return explained_variance

	def fit(self, dataset: Dataset):
		"""
		Calculates the mean, the components and the explained variance.

		Returns
        ------------
		Dataset.
		"""

		self.components = self.get_components(dataset)
		self.explained_variance = self.get_explained_variance(dataset)

		return self

	def transform(self, dataset: Dataset) -> Dataset:
		"""
		Transforms the dataset.

		Parameters
        ------------
		Dataset object.
		"""
		if self.components is None:
			raise Exception("You must fit the PCA before transform the dataset.")

		# Get centered data
		centered_data = self.get_centered_data(dataset)

		# Get transposed V matrix
		v_matrix = self.v_matrix_t.T

		# Get transformed data
		transformed_data = np.dot(centered_data, v_matrix)

		return Dataset(transformed_data, dataset.y, dataset.features_names, dataset.label_name)

	def fit_transform(self, dataset: Dataset) -> Dataset:
		"""
		Calculates the mean and the explained variance of the components and transforms the dataset.

		Returns
        ------------
		Dataset object.
		"""
		self.fit(dataset)
		return self.transform(dataset)

