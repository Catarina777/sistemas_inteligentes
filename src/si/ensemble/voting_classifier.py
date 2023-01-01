import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
    def __init__(self, models: list):
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

    def predict(self, dataset) -> np.array:
        """
        Combines the previsions of each model with a voting system.

		Parameters
        ------------
		dataset: Dataset object to predict the labels of.

		Returns
        ------------
        The most represented class
        """

        def _get_most_represented_class(pred: np.array) -> int:
            labels, counts = np.unique(pred, return_counts=True)

            return labels[np.argmax(counts)]

        # list of the predictions
        predictions = []

        for model in self.models:
            predictions.append(model.predict(dataset))

        # computes the most represented class
        predictions = np.array(predictions)
        most_represented_class = np.apply_along_axis(_get_most_represented_class, axis=0, arr=predictions)

        return most_represented_class

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
