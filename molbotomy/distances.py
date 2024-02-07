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
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdchem import Mol
from molbotomy.descriptors import mols_to_ecfp, mols_to_maccs, mols_to_descriptors
from molbotomy.utils import mols_to_smiles
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm
from typing import Callable, Union
from warnings import warn
import sys


class MolecularDistanceMatrix:
    """ Compute a pairwise distance matrix using a type of molecular descriptor and a distance metric.

    :param descriptor: either 'ecfp', 'maccs', or 'physchem' or a callable that vectorizes rdkit molecule objects
        into a np.ndarray. (default = 'ecfp')
    :param distance: either 'euclidean', 'tanimoto', or 'edit' or a callable that computes the pairwise distance
     between np.ndarrays (default = 'tanimoto')
    """
    distances = ['euclidean', 'tanimoto', 'edit']
    descriptors = ['ecfp', 'maccs', 'physchem']
    is_sim = False

    def __init__(self, descriptor: Union[str, Callable] = 'ecfp', distance: Union[str, Callable] = 'tanimoto'):

        self.descriptor = descriptor
        self.distance = distance
        self.dist = None

        if descriptor == 'ecfp':
            self.descriptor = mols_to_ecfp
        elif descriptor == 'maccs':
            self.descriptor = mols_to_maccs
        elif descriptor == 'physchem':
            self.descriptor = mols_to_descriptors
        elif type(descriptor) is str:
            warn(f"distance '{descriptor}' is not supported by default (choose from {self.descriptors}). "
                 f"Otherwise, supply your own callable that takes in the output from descriptor_func to compute a "
                 f"square distance matrix")

        if distance == 'euclidean':
            self.distance = euclideandistance_matrix
            self.is_sim = False
        elif distance == 'tanimoto':
            self.distance = tanimoto_matrix
            self.is_sim = True
        elif distance == 'edit':
            self.distance = editdistance_matrix
            self.is_sim = False
        elif type(distance) is str:
            warn(f"distance '{distance}' is not supported by default (choose from {self.distances}). Otherwise, "
                 f"supply your own callable that takes in the output from descriptor_func to compute a "
                 f"square distance matrix")

    def compute_dist(self, mols: list[Mol], **kwargs) -> np.ndarray:
        """ Compute pairwise distances between molecules

        :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
        :param kwargs: kwargs passed to the descriptor function
        :return: pairwise distance matrix
        """
        if self.distance == 'edit':
            print(f'Computing {self.distance} distance between SMILES', flush=True, file=sys.stderr)
            smiles = mols_to_smiles(mols)
            self.dist = self.distance(smiles)
        else:
            print(f'Computing {self.distance} distance between {self.descriptor}', flush=True, file=sys.stderr)
            x = self.descriptor(mols, **kwargs)

            if self.is_sim:
                self.dist = 1 - self.distance(x)
            else:
                self.dist = self.distance(x)

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
    :param dtype: numpy dtype (default = np.float16)
    :param fill_diagonal: Fill the diagonal with 1's (default = True)

    :return: Tanimoto similarity matrix
    """
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
    :param dtype: numpy dtype (default = np.float16)
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


def euclideandistance_matrix(x, fill_diagonal: bool = True) -> np.ndarray:

    X = pdist(x)
    X = squareform(X)
    if not fill_diagonal:
        np.fill_diagonal(X, 0)

    return X
