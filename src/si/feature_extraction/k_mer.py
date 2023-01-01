import itertools
from typing import Literal, get_args

import numpy as np

from si.data.dataset import Dataset

sequence_composition = Literal['dna', 'peptide']


class KMer:
    """
    Computes the k-mer composition of a given nucleotide or peptide sequence.
    """

    def __init__(self, k: int = 3, alphabet: str = 'dna'):
        # parameters
        if k < 1:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.alphabet = alphabet

        # attributes
        self.k_mers = None

    def fit(self, dataset: Dataset):
        """
        Estimates all possible k-mers from a dataset.

		Returns
        ------------
        self.
        """
        # check if sequence composition is a valid choice
        options = get_args(sequence_composition)
        assert self.alphabet in options, f"'{self.alphabet}' is not in {options}"

        if self.alphabet == 'dna':
            # if the sequence is composed of nucleotides
            self.alphabet = 'ACGT'  # DNA alphabet
            # Get all possible k-mers
            self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]  # iterates
            # the tuple and adds it to the string

        elif self.alphabet == 'peptide':
            # if the sequence is composed of amino acids
            self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
            self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]

        return self

    def _get_k_mer_count(self, sequence: str) -> np.ndarray:
        """
        Computes the normalized frequency of each k-mer in each sequence of the dataset

		Returns
        ------------
		Dataset.
        """

        # Builds a dictionary with all the possible k-mers with counts of zero
        k_mer_counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            k_mer_counts[k_mer] += 1

        # Normalize the counts
        return np.array([k_mer_counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Computes the k-mer composition of a given DNA sequence (ACGT).

		Returns
        ------------
		Dataset.
        """
        # Get k-mer counts
        k_mer_counts = np.array([self._get_k_mer_count(sequence) for sequence in dataset.x[:, 0]])

        # Create new dataset
        return Dataset(x=k_mer_counts, y=dataset.y, features_names=self.k_mers, label_name=dataset.label_name)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Runs the fit and the transform method.

		Returns
        ------------
		Dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)
