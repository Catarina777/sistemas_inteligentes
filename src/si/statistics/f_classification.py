from scipy.stats import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset):
    """
    Calculates the score for each feature for classification tasks.

	Parameters
    ------------
	dataset: Dataset object.

    Returns
    ------------
	F value for each feature.
    """
    dataset_classes = dataset.get_classes()
    dataset_groups = [dataset.x[dataset.y == c] for c in dataset_classes]  # group the dataset by class
    f_value, p_value = stats.f_oneway(*dataset_groups)

    return f_value, p_value
