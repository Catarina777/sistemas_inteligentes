import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import isnull
from typing import Tuple,Sequence


class Dataset:
	def __init__(self, x: ndarray, y: ndarray = None, features_names: Sequence[str] = None, label_name: str = None):
		"""
		Initializes the dataset.

		Parameters
		------------
		x: Values of the features.
		y: Samples.
		features: Names of the features.
		label: Name of the label.
		"""
		if x is None:
			raise ValueError("x cannot be None")

		if features_names is None:
			features_names = [str(i) for i in range(x.shape[1])]
		else:
			features_names = list(features_names)

		self.x = x
		self.y = y
		self.features_names = features_names
		self.label_name = label_name

	def shape(self) -> tuple[int, int]:
		"""
		Returns the shape of the dataset.

		Returns
		------------
		Tuple
		"""
		return self.x.shape

	def has_label(self) -> bool:
		"""
		Checks if the dataset has a label.

		Returns
		------------
		Boolean
		"""
		# If the y exists it means that is supervised
		if self.y is not None:
			return True
		return False

	def get_classes(self) -> np.ndarray:
		"""
		Returns the classes of the dataset.

		Returns
		------------
		ndarray
		"""
		if self.y is None:
			raise ValueError("Dataset does not have a label")
		return np.unique(self.y)

	def get_mean(self) -> np.ndarray:
		"""
		Returns the mean of the dataset for each feature.

		Returns
		------------
		ndarray
		"""
		return np.mean(self.x, axis=0)

	def get_variance(self) -> np.ndarray:
		"""
		Returns the variance of the dataset for each feature.

		Returns
		------------
		ndarray
		"""
		return np.var(self.x, axis=0)

	def get_median(self) -> np.ndarray:
		"""
		Returns the median of the dataset for each feature.

		Returns
		------------
		ndarray
		"""
		return np.median(self.x, axis=0)

	def get_min(self) -> np.ndarray:
		"""
		Returns the minimum value of the dataset for each feature.

		Returns
		------------
		ndarray
		"""
		return np.nanmin(self.x, axis=0)

	def get_max(self) -> np.ndarray:
		"""
		Returns the maximum value of the dataset for each feature.

		Returns
		------------
		ndarray
		"""
		return np.nanmax(self.x, axis=0)

	def dropna(self):
		"""
		Removes all of the samples that contain one null value minimum (NaN)
		"""
		if self.shape()[0] != len(self.y):
			raise ValueError("Examples must be equal to the length of y")
		# if it has y
		if self.has_label():
			self.y = self.y[~np.isnan(self.X).any(axis=1)]
		# makes the opposite and returns all the lines that don't have NaN
		self.X = self.X[~np.isnan(self.X).any(axis=1)]

	def fillna(self, val: int):
		"""
		Replaces all null values with a given value

		Paramaters
		----------
		val: Given value to replace the NaN values with
		"""
		# changes all the values that had NaN as a value, for one that we choose
		return np.nan_to_num(self.X, nan = val, copy = False)


	def summary(self):
		"""
		Prints a summary of the dataset with the mean, variance, median, min and max for each feature.

		Returns
		------------
		DataFrame
		"""

		return pd.DataFrame(
			{'mean': self.get_mean(),
			'median': self.get_median(),
			'min': self.get_min(),
			'max': self.get_max(),
			'var': self.get_variance()}
		)

	def from_dataframe(cls, df: pd.DataFrame, label: str = None):
		"""
		Creates a Dataset object from a pandas DataFrame

		Parameters
		------------
		df: The DataFrame
		label: The label name

		Returns
		------------
		Dataset
		"""
		if label:
			X = df.drop(label, axis=1).to_numpy()
			y = df[label].to_numpy()
		else:
			X = df.to_numpy()
			y = None

		features = df.columns.tolist()
		return cls(X, y, features=features, label=label)

	def to_dataframe(self) -> pd.DataFrame:
		"""
		Converts the dataset to a pandas DataFrame
		"""
		if self.y is None:
			return pd.DataFrame(self.X, columns=self.features)
		else:
			df = pd.DataFrame(self.X, columns=self.features)
			df[self.label] = self.y
			return df

	@classmethod
	def from_random(cls, n_samples: int, features_names: int, n_classes: int = 2, features = None, label_name: str = None):
		"""
		Creates a Dataset object from random data

		Parameters
		------------
		n_samples: The number of samples
		n_features: The number of features
		n_classes: The number of classes
		features: The feature names
		label: The label name
		"""
		X = np.random.rand(n_samples, features_names)
		y = np.random.randint(0, n_classes, n_samples)
		return cls(X, y, features_names=features_names, label_name=label_name)

if __name__ == '__main__':
	x1 = np.array([[1,2,3], [2,4,6], [3,np.nan,7], [8,6,np.nan]])
	y1 = np.array([2,4,5])
	dataset2 = Dataset(X=x1, y=y1)
	print(dataset2.shape())  # before
	dataset2.dropna()
	print(dataset2.shape())
