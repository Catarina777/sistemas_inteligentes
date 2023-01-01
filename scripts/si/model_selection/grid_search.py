import itertools

from typing import Callable, Tuple, List, Dict

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset: Dataset, parameter_grid: Dict[str, Tuple], scoring: Callable = None, cv: int = 3, test_size: float = 0.3) -> Dict[str, List[float]]:
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
    scores = []

    # computes the cartesian product for the given parameters
	# return a list with the combination of parameters
    for combination in itertools.product(*parameter_grid.values()):
        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
			# set the combination of parameter and its values to the model
            setattr(model, parameter, value)
			# stores the parameter and its value
            parameters[parameter] = value

        # computes the model score
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)
        # stores the parameter combination and the scores
        score['parameters'].append(parameters)
        # integrates the score
        scores.append(score)

    return scores

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = grid_search_cv(knn, dataset_, parameter_grid=parameter_grid_, cv=3)
    # print the scores
    print(scores_)
