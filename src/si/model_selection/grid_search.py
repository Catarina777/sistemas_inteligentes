import itertools

from typing import Callable, Tuple, List, Dict

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model,
                   dataset: Dataset,
                   parameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.3) -> Dict[str, List[float]]:
    """
    Performs a grid search cross validation for the given model and dataset.

	Parameters
    ------------
	model: The model to evaluate.
    dataset: The dataset to evaluate.
    parameter_grid: The parameter grid to use.
    scoring: The scoring function to use.
    cv: The number of folds to use.
    test_size: The proportion of the dataset to include in the test split.

    Returns
    ------------
	A dictionary with the parameter combination and the training and testing scores.
    """

    # checks if parameters exist in the model
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"The Model {model} does not have parameter {parameter}")

    # scores = {
    #    'parameters': [],
    #    'train': [],
    #    'test': []
    # }

    scores = []

    # computes the cartesian product for the given parameters
    for combination in itertools.product(*parameter_grid.values()):  # return a list with the combination of parameters

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)  # set the combination of parameter and its values to the model
            parameters[parameter] = value  # stores the parameter and its value

        # computes the model score
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # stores the parameter combination and the scores
        score['parameters'].append(parameters)

        # integrates the score
        scores.append(score)

    return scores
