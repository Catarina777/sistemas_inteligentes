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
        data = numpy.genfromtxt(filename, delimiter=sep)
		# gets all the columns except the last one
        x = data[:, :-1]
		# gets the last column
        y = data[:, -1]
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
		# hstack stacks the data horizontally
        data = numpy.hstack((dataset.x, dataset.y.reshape(-1, 1)))
    else:
        data = dataset.x

    numpy.savetxt(filename, data, delimiter=sep)

