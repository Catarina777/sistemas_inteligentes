from typing import Type, Union

import numpy
import numpy as np
import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    """
    Reads csv files.

	Parameters
	------------
    filename: Name or directory of the csv file.
    sep: The value that is used to separate the data.
    features: Boolean value that indicates if the dataset has a defined features.
    label: Boolean value that indicates if the dataset has defined labels.

	Returns
    ------------
    Dataset object.
    """
    dataframe = pd.read_csv(filename, sep=sep)

    if features:
        features_dataframe = dataframe.iloc[:, :-1].to_numpy()
        features_names = dataframe.columns[:-1].tolist()
    else:
        features_dataframe = None
        features_names = None

    if label:
        y = dataframe.iloc[:, -1].to_numpy()
        label_name = dataframe.columns[-1]
    else:
        y = None
        label_name = None

    return Dataset(features_dataframe, y, features_names, label_name)


def write_csv(dataset: Dataset, filename: str, sep: str = ',', features: bool = False, label: bool = None) -> None:
    """
    Writes a csv file.

	Parameters
    ------------
    dataset: The dataset that is going to be written.
    filename: Name or directory of the csv file that is going to be written.
    sep: The value that is used to separate the data.
    features: Boolean value that indicates if the dataset has defined features.
    label: Boolean value that indicates if the dataset has defined labels

	Returns
    ------------
    A csv file with the dataset.
    """

    if features:
        features = dataset.x
        features_names = dataset.features_names
    else:
        features = None
        features_names = None

    if label:
        label = dataset.y
        label_name = dataset.label_name
        # label_values = label.reshape(label.shape[0], 1)
    else:
        label = None
        label_name = None
        # label_values = np.array([])

    label_values = label.reshape(label.shape[0], 1)
    results = numpy.concatenate((features, label_values), axis=1)
    labels = features_names + [label_name]

    dataframe = pd.DataFrame(data=results, columns=labels)
    dataframe.to_csv(filename, sep=sep, index=False)

if __name__ == '__main__':
    df = read_csv(filename="datasets/iris.csv", sep=',')
    print(df.shape())
