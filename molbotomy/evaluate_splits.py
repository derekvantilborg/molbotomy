"""
Perform some checks on your data splits

- SanityCheck: checks for data leakage, stereoisomerism, duplicates, and smimilarities
- intersecting_smiles: checks for overlap between train and test
- intersecting_scaffolds: checks if any scaffolds overlap between train and test
- find_duplicates: looks for duplicates molecules
- train_test_sim: compute the Tanimoto similarity between the train and the test set

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

from molbotomy.utils import canonicalize_smiles, mols_to_scaffolds, smiles_to_mols, mols_to_smiles
from molbotomy.descriptors import mols_to_ecfp
from molbotomy.cleaning import flatten_stereochemistry
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm.auto import tqdm
from typing import Union
import numpy as np


class SanityCheck:

    similarities = {'min': None, 'min_mean': None, 'max': None, 'max_mean': None, 'mean': None, 'median': None}
    similarities_scaffold = {'min': None, 'min_mean': None, 'max': None, 'max_mean': None, 'mean': None, 'median': None}

    def __init__(self, train_smiles: list[str], test_smiles: list[str]) -> None:
        """ run SanityCheck.evaluate_split() to check for data leakage, stereoisomerism, duplicates, and smimilarities

        :param train_smiles: SMILES strings of the train set
        :param test_smiles: SMILES strings of the test set
        """
        self.train_smiles = np.array(train_smiles) if type(train_smiles) is list else train_smiles
        self.test_smiles = np.array(test_smiles) if type(test_smiles) is list else test_smiles

        self.intersection = []
        self.scaffold_intersection = []
        self.stereo_intersection = []
        self.train_duplicates = []
        self.test_duplicates = []
        self.train_stereoisomers = []
        self.test_stereoisomers = []

    def check_data_leakage(self):

        self.intersection = intersecting_smiles(self.train_smiles, self.test_smiles)
        self.scaffold_intersection = intersecting_scaffolds(self.train_smiles, self.test_smiles)
        self.check_stereoisomer_intersection()

        print(f"Data leakage:\n"
              f"\tFound {len(self.intersection )} intersecting SMILES between the train and test set.\n"
              f"\tFound {len(self.scaffold_intersection )} intersecting Bemis-Murcko scaffolds between the train and "
              f"test set.\n\tFound {len(self.stereo_intersection)} intersecting stereoisomers between the train and "
              f"test set.")

    def check_duplicates(self):

        self.train_duplicates = find_duplicates(self.train_smiles)
        self.test_duplicates = find_duplicates(self.test_smiles)

        print(f"Duplicates:\n"
              f"\tFound {len(self.train_duplicates)} duplicate SMILES in the train set.\n"
              f"\tFound {len(self.test_duplicates)} duplicate SMILES in the test set.")

    def similarity(self, scaffolds: bool = False):

        X = train_test_sim(self.train_smiles, self.test_smiles, scaffolds=scaffolds)

        if scaffolds:
            self.similarities_scaffold = {'min': np.min(X), 'min_mean': np.min(np.mean(X, 1)),
                                          'max': np.max(X), 'max_mean': np.max(np.mean(X, 1)),
                                          'mean': np.mean(X), 'median': np.median(X)}

            print(f"Scaffold Tanimoto similarity between the train and test set:\n"
                  f"\tMean:\t{self.similarities_scaffold['mean']}\n"
                  f"\tMedian:\t{self.similarities_scaffold['median']}\n"
                  f"\tMin:\t{self.similarities_scaffold['min']}\n"
                  f"\tMax:\t{self.similarities_scaffold['max']}")
        else:
            self.similarities = {'min': np.min(X), 'min_mean': np.min(np.mean(X, 1)),
                                 'max': np.max(X), 'max_mean': np.max(np.mean(X, 1)),
                                 'mean': np.mean(X), 'median': np.median(X)}

            print(f"Tanimoto similarity between the train and test set:\n"
                  f"\tMean:\t{self.similarities['mean']}\n"
                  f"\tMedian:\t{self.similarities['median']}\n"
                  f"\tMin:\t{self.similarities['min']}\n"
                  f"\tMax:\t{self.similarities['max']}")

    def check_stereoisomer_occurance(self):

        # Remove duplicates
        tr_set = set(canonicalize_smiles(self.train_smiles))
        tst_set = set(canonicalize_smiles(self.test_smiles))

        # Flatten molecules and canonicalize again
        tr_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tr_set]))
        tst_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tst_set]))

        self.train_stereoisomers = find_duplicates(tr_flat)
        self.test_stereoisomers = find_duplicates(tst_flat)

        print(f"Stereoisomers:\n"
              f"\tFound {len(self.train_stereoisomers)} Stereoisomer SMILES in the train set.\n"
              f"\tFound {len(self.test_stereoisomers)} Stereoisomer SMILES in the test set.")

    def check_stereoisomer_intersection(self):

        tr_orignal = set(canonicalize_smiles(self.train_smiles))
        tst_original = set(canonicalize_smiles(self.test_smiles))

        tr_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tr_orignal]))
        tst_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tst_original]))

        self.stereo_intersection = intersecting_smiles(tr_orignal, tst_flat) + intersecting_smiles(tst_original, tr_flat)

        return self.stereo_intersection

    def evaluate_split(self):
        self.check_data_leakage()
        self.check_duplicates()
        self.check_stereoisomer_occurance()
        self.check_stereoisomer_intersection()
        self.similarity(scaffolds=False)
        self.similarity(scaffolds=True)


def intersecting_smiles(smiles_a: Union[np.ndarray[str], list[str]], smiles_b: Union[np.ndarray[str], list[str]]) -> \
        list[str]:
    """ Finds the intersection between two sets of SMILES strings

    :param smiles_a: set of SMILES strings
    :param smiles_b: another set of SMILES strings
    :return: list of overlapping SMILES strings
    """
    smiles_a = np.array(canonicalize_smiles(smiles_a))
    smiles_b = np.array(canonicalize_smiles(smiles_b))

    intersection = list(np.intersect1d(smiles_a, smiles_b))

    return intersection


def intersecting_scaffolds(smiles_a: Union[np.ndarray[str], list[str]], smiles_b: Union[np.ndarray[str], list[str]]) \
        -> list[str]:
    """ Computes scaffolds in two sets of SMILES strings and returns intersecting scaffolds

    :param smiles_a: set of SMILES strings
    :param smiles_b: another set of SMILES strings
    :return: list of SMILES strings with scaffolds also present in the other set
    """

    scaffolds_a = mols_to_scaffolds(smiles_to_mols(smiles_a))
    scaffolds_b = mols_to_scaffolds(smiles_to_mols(smiles_b))

    scaffold_smiles_a = canonicalize_smiles(mols_to_smiles(scaffolds_a))
    scaffold_smiles_b = canonicalize_smiles(mols_to_smiles(scaffolds_b))

    intersection = list(np.intersect1d(scaffold_smiles_a, scaffold_smiles_b))

    return intersection


def find_duplicates(smiles):

    smiles = canonicalize_smiles(smiles)

    seen = set()
    dupes = [smi for smi in smiles if smi in seen or seen.add(smi)]

    return dupes


def train_test_sim(train_smiles: list[str], test_smiles: list[str], progressbar: bool = False, scaffolds: bool = False,
                   dtype=np.float16) -> np.ndarray:
    """ Computes the Tanimoto similarity between the train and the test set (on ECFPs, nbits=1024, radius=2)

    :param train_smiles: SMILES strings of the train set
    :param test_smiles: SMILES strings of the test set
    :param progressbar: toggles progressbar (default = False)
    :param scaffolds: toggles the use of scaffolds instead of the full molecule (default = False)
    :param dtype: numpy dtype (default = np.float16)
    :return: n_train x n_test similarity matrix
    """

    mols_train = smiles_to_mols(train_smiles, sanitize=True)
    mols_test = smiles_to_mols(test_smiles, sanitize=True)
    if scaffolds:
        mols_train = mols_to_scaffolds(mols_train)
        mols_test = mols_to_scaffolds(mols_test)

    fp_train = mols_to_ecfp(mols_train)
    fp_test = mols_to_ecfp(mols_test)

    n, m = len(fp_train), len(fp_test)
    X = np.zeros([n, m], dtype=dtype)

    for i in tqdm(range(n), disable=not progressbar):
        X[i] = BulkTanimotoSimilarity(fp_train[i], fp_test)

    return X
