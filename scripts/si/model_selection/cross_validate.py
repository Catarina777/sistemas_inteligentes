from typing import List, Callable, Dict, Union

import numpy as np
from traitlets import Float

from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from si.linear_module.logistic_regression import LogisticRegression

Num = Union[int, float]


def cross_validate(model, dataset: Dataset, scoring: Callable = None, cv: int = 3, test_size: float = 0.3) -> Dict[str, List[Num]]:
	"""
	Computes the cross-validated score for the given model and dataset.

	Parameters
	------------
	model: The model to evaluate.
	dataset: The dataset to evaluate.
	scoring: The scoring function to use.
	cv: The number of folds to use.
	test_size: The proportion of the dataset to include in the test split.

	Returns
	------------
	A dictionary with the cross-validated scores.
	"""
	scores = {
		'seed': [],
		'train': [],
		'test': [],
		'parameters': []
	}

	# computes the score for each fold of the score
	for i in range(cv):
		# set the random seed
		random_state = np.random.randint(0, 1000)
		# store the seed
		scores['seed'].append(random_state)
		# splits the train and test
		train, test = train_test_split(dataset = dataset, test_size = test_size, random_state = random_state)
		# fit the model on the train
		model.fit(train)

		# calculates the training score
		if scoring is None:
			# stores the train score
			scores['train'].append(model.score(train))
			# stores the test score
			scores['test'].append(model.score(test))

		else:
			y_train = train.y
			y_test = test.y
			# stores the train score
			scores['train'].append(scoring(train.y, model.predict(train)))
			# stores the test score
			scores['test'].append(scoring(test.y, model.predict(test)))

	return scores

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.neighbors.knn_classifier import KNNClassifier

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the KNN
    knn = KNNClassifier(k=3)

    # cross validate the model
    scores_ = cross_validate(knn, dataset_, cv=5)

    # print the scores
    print(scores_)
