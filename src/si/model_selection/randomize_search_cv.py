import itertools
from typing import Dict, Tuple, Callable, Union

import numpy as np
import random
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate

Num = Union[int, float]


def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Dict[str, Tuple], scoring: Callable = None, cv: int = 3, n_iter: int = 10, test_size: float = 0.3) -> Dict[str, Tuple[str, Num]]:
	"""
	Performs a randomized search cross validation on a model.

	Parameters
	------------
	model: The model to cross validate.
	dataset: The dataset to cross validate on.
	parameter_distribution: The parameter grid to use.
	scoring: The scoring function to use.
	cv: The cross validation folds.
	n_iter: Number of iterations.
	test_size: The test size.
	"""

	scores = []

	# for each combination
	for i in range(n_iter):

		# parameter configuration
		parameters = {}

		# set the parameters
		for parameter in parameter_distribution:
			#take a random value from the value distribution of each parameter
			value = np.random.choice(parameter_distribution[parameter])
			setattr(model, parameter, value)
			parameters[parameter] = value

		# cross validate the model
		score = cross_validate(model = model, dataset = dataset, scoring = scoring, cv = cv, test_size = test_size)

		# add the parameter configuration
		score['parameters'] = parameters

		# add the score
		scores.append(score)

	return scores
