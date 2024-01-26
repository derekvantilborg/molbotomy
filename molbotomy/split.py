"""
Code to split a set of molecules

- Splitter: Splits a list of SMILES strings into a train and test set based on scaffolds, clustering, or at random
- random_split: splits a list at random
- scaffold_split: splits molecules based on their scaffold in three modes: random, balanced, and out-of-distribution
- cluster_split: splits molecules based on clustering in three modes: random, balanced, and out-of-distribution


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import numpy as np
from molbotomy.utils import smiles_to_mols, map_scaffolds
from molbotomy.distances import MolecularDistanceMatrix
from molbotomy.clustering import ClusterMolecularDistanceMatrix
import sys


class Splitter:
    """
    Splits a list of SMILES strings into a train and test set. Molecules can be split based on molecular scaffolds,
    clustering, or at random. Splitting is done in three modes, random, balanced, and out-of-distribution (OOD)

    :param split_method: 'random', 'scaffold', 'cluster' (default = 'random')
    :param mode: 'random', 'balanced' or 'OOD' (default = 'random')
    :param seed: random seed (default = 42)
    :param n_clusters: number of clusters used for clustering (default = 20)

    Scaffold splitting
        - random: sets of scaffolds are randomly distributed between train and test, meaning that all molecules that
            belong to a set of scaffolds are in either the train or test set together
        - balanced: sets of scaffolds are distributed between train and test so that the test set molecules are
            maximally representative of the train set. Here too, molecules that share the same scaffold are kept
            together in either the train or test set.
        - OOD: sets of scaffolds are distributed between train and test so that the test set molecules are maximally
            different of the train set (the opposite of balanced). Here too, molecules that share the same scaffold are
            kept together in either the train or test set.

    Cluster splitting:
        - random: clusters of molecules are randomly distributed between train and test, meaning that all molecules that
            belong to a cluster are in either the train or test set together
        - balanced: clusters of molecules are distributed between train and test so that the test set molecules are
            maximally representative of the train set. Here too, molecules that share the same cluster are kept
            together in either the train or test set.
        - OOD: clusters of molecules are distributed between train and test so that the test set molecules are maximally
            different of the train set (the opposite of balanced). Here too, molecules that share the same cluster are
            kept together in either the train or test set.

    Random splitting:
        - random: molecules are distributes at random between the train and test set.

    :param smiles: List of SMILES strings
    :param sanitize_smiles: toggle SMILES sanitization
    """
    split_methods = ['random', 'scaffold', 'cluster']
    modes = ['random', 'balanced', 'OOD']

    def __init__(self, split_method: str = 'random', mode: str = 'balanced', seed: int = 42, n_clusters: int = 20):
        self.seed = seed
        self.n_clusters = n_clusters
        self.train_idx = None
        self.test_idx = None
        if mode == 'ood':
            mode = 'OOD'

        assert mode in self.modes, f"method '{mode}' is not supported. Pick from: {self.modes}"
        assert split_method in self.split_methods, f"method '{split_method}' is not supported. Pick from: " \
                                                   f"{self.split_methods}"
        self.mode = mode
        self.split_method = split_method

    def split(self, smiles, ratio: float = 0.2, sanitize_smiles: bool = True, **kwargs) -> (np.ndarray, np.ndarray):
        """ Split a list of SMILES strings

        :param smiles: List of SMILES strings
        :param ratio: test split ratio (default = 0.2, splits off 20% of the data into a test set)
        :param sanitize_smiles: toggle SMILES sanitization
        :param kwargs: keyword args given to the splitting method
        :return: train indices, test indices
        """

        mols = smiles_to_mols(smiles, sanitize=sanitize_smiles)

        if self.split_method == 'random':
            self.train_idx, self.test_idx = random_split(mols, ratio=ratio, seed=self.seed, **kwargs)

        elif self.split_method == 'scaffold':
            self.train_idx, self.test_idx = scaffold_split(mols, mode=self.mode, ratio=ratio, seed=self.seed, **kwargs)

        elif self.split_method == 'cluster':
            self.train_idx, self.test_idx = cluster_split(mols, mode=self.mode, n_clusters=self.n_clusters, ratio=ratio,
                                                          seed=self.seed, **kwargs)

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
        tani_scaff = MolecularDistanceMatrix().compute_dist(scaffolds)

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


def cluster_split(mols: list, mode: str = 'random', n_clusters: int = 10, ratio: float = 0.2, seed: int = 42)\
        -> (np.ndarray, np.ndarray):
    """ ... TODO


    Generates an out-of-distribution split based on clustering. Clusters with the largest average dissimilarity to all
    other scaffold sets (based on Tanimoto similarity on ECFPs) are taken as the test set.

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param mode: 'random', 'balanced' or 'OOD'. 'random' distributes random scaffold sets over train and train,
    'balanced' takes the most similar scaffold sets for the test set and 'OOD' takes the least similar scaffold sets.
    :param ratio: test split ratio (default = 0.2, splits off a maximum of 20% of the data into a test set). Exact size
    of the split depends on scaffold set sizes.
    :param progressbar: toggles progressbar (default = True)
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """
    import kmedoids

    tani = MolecularDistanceMatrix().compute_dist(mols)
    clustering = ClusterMolecularDistanceMatrix(clustering_method='kmedoids', n_clusters=n_clusters, seed=seed)
    clustering.fit(tani)

    cluster_membership = clustering.labels_
    cluster_map = {i: np.where(cluster_membership == i)[0] for i in range(n_clusters)}

    if mode == 'random':
        rng = np.random.default_rng(seed=seed)
        set_order = np.arange(len(cluster_map))
        rng.shuffle(set_order)

        train_mols, test_mols = [], []
        for i in set_order:
            new_mols = np.where(cluster_membership == i)[0].tolist()
            if len(test_mols) + len(new_mols) <= round(len(mols) * ratio):
                test_mols.extend(new_mols)
            else:
                break
    else:
        print('Finding similarities between scaffold sets', flush=True, file=sys.stderr)
        # For each group of scaffolds, find the MEAN distance to all other clusters
        cluster_sims = np.zeros((n_clusters, n_clusters), dtype=np.float16)

        for i, (k, v) in enumerate(cluster_map.items()):
            for j, (k_, v_) in enumerate(cluster_map.items()):
                cluster_sims[i, j] = np.mean(tani[v][:, v_])

        # Find the scaffold clusters with the lowest average similarity to all other clusters
        sim_order = np.argsort(np.mean(cluster_sims, 0))
        if mode == 'balanced':  # when mode == balanced, we want to use the highest average similarity instead
            sim_order = np.argsort(np.mean(cluster_sims, 0))[::-1]

        train_mols, test_mols = [], []
        for i in sim_order:
            new_mols = np.where(cluster_membership == i)[0].tolist()
            if len(test_mols) + len(new_mols) <= round(len(mols) * ratio):
                test_mols.extend(new_mols)
            else:
                break

    # the train molecules are molecules not in the test set
    train_mols = [i for i in range(len(mols)) if i not in test_mols]

    return np.array(train_mols), np.array(test_mols)
