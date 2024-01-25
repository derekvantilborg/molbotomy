"""
Calculate distances between molecules

- MolecularDistanceMatrix: Class that computes a pairwise distance matrix from a list rdkit mols
- bulk_editdistance: Computes edit distances between SMILES strings, like RDkits' BulkTanimotoSimilarity function
- tanimoto_matrix: Computes pairwise tanimoto similarity from rdkit mols
- editdistance_matrix: Computes pairwise edit distances between SMILES strings (normalized by default)
- euclideandistance_matrix: Computes pairwise edit Euclidean distances between vectors


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import numpy as np
from Levenshtein import distance as editdistance
from tqdm.auto import tqdm
from typing import Callable, Union
from warnings import warn


class MolecularDistanceMatrix:
    distances = ['euclidean', 'tanimoto', 'edit']

    def __init__(self, mols):
        self.mols = mols
        self.dist = None

    def compute_dist(self, descriptor_func: Callable, distance: Union[str, Callable] = 'euclidean',
                     progressbar: bool = True, **kwargs) -> np.ndarray:

        x = descriptor_func(self.mols, **kwargs)

        if distance == 'euclidean':
            self.dist = euclideandistance_matrix(x)
        elif distance == 'tanimoto':
            self.dist = 1 - tanimoto_matrix(x, progressbar=progressbar)
        elif distance == 'edit':
            self.dist = editdistance_matrix(x)
        else:
            if type(distance) is str:
                warn(f"distance '{distance}' is not supported by default (choose from {self.distances}). Otherwise, "
                     f"supply your own callable that takes in the output from descriptor_func to compute a "
                     f"square distance matrix")
            self.dist = distance(x)

        return self.dist


def bulk_editdistance(smile: str, smiles: list[str], normalize: bool = True) -> np.ndarray:
    if normalize:
        return np.array([editdistance(smile, smi) / max(len(smile), len(smi)) for smi in smiles])
    else:
        return np.array([editdistance(smile, smi) for smi in smiles])


def tanimoto_matrix(fingerprints: list, progressbar: bool = False, fill_diagonal: bool = True, dtype=np.float16) \
        -> np.ndarray:
    """

    :param fingerprints: list of RDKit fingerprints
    :param progressbar: toggles progressbar (default = False)
    :param fill_diagonal: Fill the diagonal with 1's (default = True)

    :return: Tanimoto similarity matrix
    """
    from rdkit.DataStructs import BulkTanimotoSimilarity

    n = len(fingerprints)

    X = np.zeros([n, n], dtype=dtype)
    # Fill the upper triangle of the pairwise matrix
    for i in tqdm(range(n), disable=not progressbar, desc=f"Computing pairwise Tanimoto similarity of {n} molecules"):
        X[i, i+1:] = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
    # Mirror out the lower triangle
    X = X + X.T - np.diag(np.diag(X))

    if fill_diagonal:
        np.fill_diagonal(X, 1)

    return X


def editdistance_matrix(smiles: list[str], progressbar: bool = False, fill_diagonal: bool = True, dtype=np.float16,
                        **kwargs) -> np.ndarray:
    """

    :param smiles: List of SMILES strings
    :param progressbar: toggles progressbar (default = False)
    :param fill_diagonal: Fill the diagonal with 1's (default = True)

    :return: Edit distance matrix (mind you, this is a distance matrix, not a similarity matrix)
    """
    n = len(smiles)
    X = np.zeros([n, n], dtype=dtype)
    # Fill the upper triangle of the pairwise matrix
    for i in tqdm(range(n), disable=not progressbar, desc=f"Computing pairwise edit distance of {n} SMILES strings"):
        X[i, i+1:] = bulk_editdistance(smiles[i], smiles[i+1:], **kwargs)
    # Mirror out the lower triangle
    X = X + X.T - np.diag(np.diag(X))

    if fill_diagonal:
        np.fill_diagonal(X, 1)

    return X


def euclideandistance_matrix(x, fill_diagonal: bool = True):
    from scipy.spatial.distance import pdist, squareform

    X = pdist(x)
    X = squareform(X)
    if not fill_diagonal:
        np.fill_diagonal(X, 0)

    return X
