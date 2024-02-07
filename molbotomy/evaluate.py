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


def check_splits(train_smiles: list[str], test_smiles: list[str]) -> None:

    # turn SMILES into a np.array
    train_smiles = np.array(train_smiles)
    test_smiles = np.array(test_smiles)

    print(f"Evaluating {len(train_smiles)} train SMILES and {len(test_smiles)} test SMILES:")

    # Check data leakage
    intersection = intersecting_smiles(train_smiles, test_smiles)
    scaffold_intersection = intersecting_scaffolds(train_smiles, test_smiles)
    stereo_intersection = check_stereoisomer_intersection(train_smiles, test_smiles)

    print(f"Data leakage:\n"
          f"\tFound {len(intersection)} intersecting SMILES between the train and test set.\n"
          f"\tFound {len(scaffold_intersection)} intersecting Bemis-Murcko scaffolds between the train and "
          f"test set.\n\tFound {len(stereo_intersection)} intersecting stereoisomers between the train and "
          f"test set.")

    # check duplicates
    train_duplicates = find_duplicates(train_smiles)
    test_duplicates = find_duplicates(test_smiles)
    print(f"Duplicates:\n"
          f"\tFound {len(train_duplicates)} duplicate SMILES in the train set.\n"
          f"\tFound {len(test_duplicates)} duplicate SMILES in the test set.")

    # check stereoisomer occurance
    train_stereoisomers = check_stereoisomer_occurance(train_smiles)
    test_stereoisomers =  check_stereoisomer_occurance(test_smiles)
    print(f"Stereoisomers:\n"
          f"\tFound {len(train_stereoisomers)} Stereoisomer SMILES in the train set.\n"
          f"\tFound {len(test_stereoisomers)} Stereoisomer SMILES in the test set.")


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


def check_stereoisomer_intersection(train_smiles: list[str], test_smiles: list[str]) -> list[str]:
    """ Checks if molecules in the train set have stereoisomers in the test set and vice verca

    :param train_smiles: list of SMILES strings
    :param test_smiles: list of SMILES strings
    :return: list of intersecting stereoisomers
    """
    tr_orignal = set(canonicalize_smiles(train_smiles))
    tst_original = set(canonicalize_smiles(test_smiles))

    tr_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tr_orignal]))
    tst_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in tst_original]))

    stereo_intersection = intersecting_smiles(tr_orignal, tst_flat) + intersecting_smiles(tst_original, tr_flat)

    return stereo_intersection


def check_stereoisomer_occurance(smiles: list[str]) -> list[str]:
    """ Look for SMILES that have a stereoisomer in the same set

    :param smiles: list of SMILES strings
    :return: duplicated stereoisomers
    """

    # Remove duplicates
    smiles = set(canonicalize_smiles(smiles))

    # Flatten molecules and canonicalize again
    smiles_flat = np.array(canonicalize_smiles([flatten_stereochemistry(smi) for smi in smiles]))

    # look for duplicated molecules
    stereoisomers = find_duplicates(smiles_flat)

    return stereoisomers


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
