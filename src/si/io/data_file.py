import numpy

from si.data.dataset import Dataset


def read_data_file(filename: str, label: bool = False, sep: str = ","):
    """
    Reads a data file.

	Parameters
    ------------
	filename: Name or directory of the data file.
    label: Boolean value that indicates if the dataset has defined labels.
    sep: The value that is used to separate the data.

	Returns
    ------------
    Dataset.
    """

    if label:

        # if it were to be a csv file read with numpy.genfromtxt:
        # dataframe = numpy.genfromtxt(filename, delimiter=sep, names=True, dtype=None, encoding='UTF-8')  # dtype =
        # None (doesn't define a dtype from the get go, allowing numpy to infer the dtype from the data), names
        # defines the names of the columns and encoding defines the encoding of the file's strings.
        # - retorns a structured array

        # label_name = dataframe.dtype.names[-1] # gets the name of the last column
        # features_names = list(dataframe.dtype.names)[:-1] # gets the names of the columns except the last one
        # y = dataframe[label_name]
        # x = dataframe[features_names]
        # x = np.array(x.tolist()) # convert structured numpy array to regular numpy array

        data = numpy.genfromtxt(filename, delimiter=sep)
        x = data[:, :-1]  # gets all the columns except the last one
        y = data[:, -1]  # gets the last column
    else:
        x = numpy.genfromtxt(filename, delimiter=sep)
        y = None

    return Dataset(x, y)


def write_data_file(dataset: Dataset, filename: str, label: bool = False, sep: str = ","):
    """
    Writes a data file.

	Parameters
    ------------
	dataset: The dataset that is going to be written.
    filename: Name or directory of the data file that is going to be written.
    sep: The value that is used to separate the data.
    label: Boolean value that indicates if the dataset has defined labels.

	Returns
    ------------
    A data file with the dataset.
    """

    if label:
        data = numpy.hstack((dataset.x, dataset.y.reshape(-1, 1)))  # hstack stacks the data horizontally
    else:
        data = dataset.x

    numpy.savetxt(filename, data, delimiter=sep)


if __name__ == '__main__':
    dataset = read_data_file('data.txt', label=True)
    print(dataset.print_dataframe())
    # print(dataset.shape())
    # print(dataset.has_label())
    # print(dataset.get_classes())
    # print(dataset.get_mean())
    # print(dataset.get_variance())
    # print(dataset.get_median())
    # print(dataset.get_min())
    # print(dataset.get_max())
    # print(dataset.summary())
