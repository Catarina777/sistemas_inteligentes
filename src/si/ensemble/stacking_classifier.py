import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    def __init__(self, models: list, final_model):
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

        # trains the final model
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
