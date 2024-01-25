"""
Code to split a set of molecules

Methods:
    - random
    - scaffold + a distance: Find all scaffold sets and then get all the scaffolds that are maximally distanced from the
      rest untill we have enough molecules in our holdout set
    - clustering

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""


import numpy as np
from rdkit import Chem
from molbotomy.utils import mols_to_scaffolds, smiles_to_mols, map_scaffolds
from molbotomy.Descriptors import mols_to_ecfp
from molbotomy.Distances import tanimoto_matrix
from warnings import warn
import sys


class Splitter:
    split_methods = ['random', 'scaffold', 'cluster']
    modes = ['random', 'balanced', 'OOD']
    cluster_methods = [None, ]

    def __init__(self, smiles, sanitize_smiles: bool = True):
        """
        :param smiles: List of SMILES strings
        :param sanitize_smiles: toggle SMILES sanitization

        """
        self.mols = smiles_to_mols(smiles, sanitize=sanitize_smiles)
        self.mode = None
        self.split_method = None
        self.train_idx = None
        self.test_idx = None

    def split(self, split_method: str = 'random', mode: str = 'balanced', ratio: float = 0.2, seed: int = 42,
              cluster_method: str = None,
              progressbar: bool = True, **kwargs) -> (np.ndarray, np.ndarray):
        """

        :param split_method:
        :param mode: 'balanced' or 'OOD' (default = 'balanced'). 'balanced' will try to produce a test
        split that is representative of the train split, whereas 'OOD' will produce a test split that is different from
        the train split.
        :param ratio: test split ratio (default = 0.2, splits off 20% of the data into a test set)
        :param seed: random seed (default = 42)
        :param cluster_method:
        :param progressbar: toggles progressbar (default = True)
        :param kwargs:
        :return: train indices, test indices
        """
        assert mode in self.modes, f"method '{mode}' is not supported. Pick from: {self.modes}"
        assert split_method in self.split_methods, f"method '{split_method}' is not supported. Pick from: " \
                                                   f"{self.split_methods}"
        self.mode = mode
        self.split_method = split_method

        if split_method == 'random':
            if self.mode != 'random':
                warn("random splits for mode='OOD' or mode='balanced' will be the same as for mode='random'")
            self.train_idx, self.test_idx = random_split(self.mols, ratio=ratio, seed=seed)

        elif split_method == 'scaffold':
            self.train_idx, self.test_idx = scaffold_split(self.mols, mode=self.mode, ratio=ratio, seed=seed,
                                                           progressbar=progressbar)

        elif split_method == 'cluster':
            assert cluster_method in self.cluster_methods, f"method '{cluster_method}' is not supported. " \
                                                           f"Pick from: {self.cluster_methods}"

            self.train_idx, self.test_idx = cluster_split()

        return self.train_idx, self.test_idx


def random_split(x, ratio: float = 0.2, seed: int = 42) -> (np.ndarray, np.ndarray):
    """ Random split data into a train and test split

    :param x: int or iterable to split
    :param ratio: test split ratio (default = 0.2, splits off 20% of the data into a test set)
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """

    rng = np.random.default_rng(seed=seed)
    rand_idx = np.arange(len(x))
    rng.shuffle(rand_idx)

    test_idx = rand_idx[:round(len(x)*ratio)]
    train_idx = rand_idx[round(len(x)*ratio):]

    return train_idx, test_idx


def scaffold_split(mols: list, mode: str = 'random', ratio: float = 0.2, progressbar: bool = True, seed: int = 42)\
        -> (np.ndarray, np.ndarray):
    """ Generates an out-of-distribution split based on Bismurcko scaffolds. Sets of unique scaffolds with the largest
    average dissimilarity to all other scaffold sets (based on Tanimoto similarity on ECFPs) are taken as the test set.

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param mode: 'random', 'balanced' or 'OOD'. 'random' distributes random scaffold sets over train and train,
    'balanced' takes the most similar scaffold sets for the test set and 'OOD' takes the least similar scaffold sets.
    :param ratio: test split ratio (default = 0.2, splits off a maximum of 20% of the data into a test set). Exact size
    of the split depends on scaffold set sizes.
    :param progressbar: toggles progressbar (default = True)
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """
    # Get scaffolds

    print('Looking for scaffolds', flush=True, file=sys.stderr)
    scaffolds, scaff_map = map_scaffolds(mols)

    # Find the scaffold membership (i.e. to which group each molecule belongs)
    scaffold_membership = np.zeros(len(scaffolds), dtype=int)
    for i, (k, v) in enumerate(scaff_map.items()):
        scaffold_membership[v] = i

    if mode == 'random':
        rng = np.random.default_rng(seed=seed)
        set_order = np.arange(len(scaff_map))
        rng.shuffle(set_order)

        train_mols, test_mols = [], []
        for i in set_order:
            new_scaffs = np.where(scaffold_membership == i)[0].tolist()
            if len(test_mols) + len(new_scaffs) <= round(len(mols) * ratio):
                test_mols.extend(new_scaffs)
            else:
                break
    else:
        # Tanimoto similarity between all scaffolds
        tani_scaff = tanimoto_matrix(mols_to_ecfp(scaffolds), progressbar=progressbar)

        print('Finding similarities between scaffold sets', flush=True, file=sys.stderr)
        # For each group of scaffolds, find the MEAN distance to all other clusters
        cluster_sims = np.zeros((len(scaff_map), len(scaff_map)), dtype=np.float16)
        for i, (k, v) in enumerate(scaff_map.items()):
            for j, (k_, v_) in enumerate(scaff_map.items()):
                cluster_sims[i, j] = np.mean(tani_scaff[v][:, v_])

        # Find the scaffold clusters with the lowest average similarity to all other clusters
        sim_order = np.argsort(np.mean(cluster_sims, 0))
        if mode == 'balanced':  # when mode == balanced, we want to use the highest average similarity instead
            sim_order = np.argsort(np.mean(cluster_sims, 0))[::-1]

        train_mols, test_mols = [], []
        for i in sim_order:
            new_scaffs = np.where(scaffold_membership == i)[0].tolist()
            if len(test_mols) + len(new_scaffs) <= round(len(mols) * ratio):
                test_mols.extend(new_scaffs)
            else:
                break

    # the train molecules are molecules not in the test set
    train_mols = [i for i in range(len(mols)) if i not in test_mols]

    return np.array(train_mols), np.array(test_mols)


def cluster_split():
    return None, None





