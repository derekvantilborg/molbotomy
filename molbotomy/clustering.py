"""
Cluster molecules with several pre-set methods

- ClusterMolecularDistanceMatrix: Cluster molecules from a distance matrix (n x n pairwise distance matrix)
- kmedoids_clustering: Perform k-medoids clustering on a pre-computed distance matrix.
- affinitypropegation_clustering: Perform affinity propegation clustering on a pre-computed distance matrix.
- spectral_clustering: Perform Spectral clustering on a pre-computed distance matrix.
- hierarchical_clustering:  Perform Hierarchical clustering on a pre-computed distance matrix.

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import numpy as np
from collections import Counter
import kmedoids
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering


class ClusterMolecularDistanceMatrix:
    """ Cluster molecules from a distance matrix (n x n pairwise distance matrix)

    :param clustering_method: "kmedoids", "affinitypropegation", "spectral", or "hierarchical", (default = kmedoids)
    :param n_clusters: number of clusters (default = 10)
    :param seed: random seed (default = 42)
    """
    allowed_methods = ["kmedoids", "affinitypropegation", "spectral", "hierarchical"]
    cluster_algo = None
    clustering = None
    labels = labels_ = None
    label_count = None

    def __init__(self, clustering_method: str = 'kmedoids', n_clusters: int = 10, seed: int = 42) -> None:

        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.seed = seed

        assert clustering_method in self.allowed_methods, f"clustering_method '{clustering_method}' is not supported. "\
                                                          f"Pick from: {self.allowed_methods}"

        # select clustering function
        if clustering_method == 'kmedoids':
            self.cluster_algo = kmedoids_clustering
        elif clustering_method == 'affinitypropegation':
            self.cluster_algo = affinitypropegation_clustering
        elif clustering_method == 'spectral':
            self.cluster_algo = spectral_clustering
        elif clustering_method == 'hierarchical':
            self.cluster_algo = hierarchical_clustering

    def fit(self, dist: np.ndarray, **kwargs) -> np.ndarray:
        """ Fit a distance matrix

        :param dist: n x n distance matrix, see Distances.MolecularDistanceMatrix
        :param kwargs: keyword arguments given to the clustering method
        :return: labels/clusters
        """

        self.clustering = self.cluster_algo(dist, n_clusters=self.n_clusters, seed=self.seed, **kwargs)
        self.labels = self.labels_ = self.clustering.labels_
        self.label_count = dict(Counter(self.labels))

        return self.clustering.labels_

    def __repr__(self):
        return f"ClusterMolecularDistanceMatrix(clustering_method={self.clustering_method}, n_clusters=" \
               f"{self.n_clusters}, seed={self.seed})\nlabels: {self.labels}\nlabel count: {self.label_count}"


def kmedoids_clustering(dist: np.ndarray, n_clusters: int = 10, seed: int = 42, **kwargs) -> kmedoids:
    """ Perform k-medoids clustering on a pre-computed matrix on molecular distances

    :param dist: n x n pairwise distance matrix for n molecules
    :param n_clusters: number of clusters (default = 10)
    :param seed: random seed (default = 42)
    :param kwargs: arguments passed through to kmedoids.fasterpam()
    :return: cluster membership of each molecule
    """

    clustering = kmedoids.fasterpam(dist.astype(float), n_clusters, random_state=seed, **kwargs)
    clustering.labels_ = clustering.labels

    return clustering


def affinitypropegation_clustering(dist: np.ndarray, seed: int = 42, n_clusters=None, **kwargs) -> AffinityPropagation:
    """ Perform affinity propegation clustering on a pre-computed matrix on molecular distances. This clustering method
        determines the number of cluster by itself

    :param dist: n x n pairwise distance matrix for n molecules
    :param seed: random seed (default = 42)
    :param kwargs: arguments passed through to AffinityPropagation()
    :param n_clusters: does nothing, here for convention
    :return: cluster membership of each molecule
    """
    A = 1 - dist

    clustering = AffinityPropagation(random_state=seed, affinity='precomputed', **kwargs)
    clustering.fit(A)

    return clustering


def spectral_clustering(dist: np.ndarray, seed: int = 42, n_clusters: int = 10, **kwargs) -> np.ndarray:
    """ Perform Spectral clustering on a pre-computed matrix on molecular distances

    :param dist: n x n pairwise distance matrix for n molecules
    :param seed: random seed (default = 42)
    :param n_clusters: number of clusters (default=10)
    :param kwargs: arguments passed through to SpectralClustering()
    :return: cluster membership of each molecule
    """
    from sklearn.cluster import SpectralClustering
    A = 1 - dist

    clustering = SpectralClustering(random_state=seed, affinity='precomputed', n_clusters=n_clusters, **kwargs)
    clustering.fit(A)

    return clustering


def hierarchical_clustering(dist: np.ndarray, n_clusters: int = 10, linkage: str = 'average', seed=None, **kwargs) \
        -> AgglomerativeClustering:
    """ Perform Hierarchical clustering on a pre-computed matrix on molecular distances

    :param dist: n x n pairwise distance matrix for n molecules
    :param seed: random seed (default = 42)
    :param n_clusters: number of clusters (default=10)
    :param linkage: linkage method. Can be 'average', 'complete'/'maximum', or 'single' (default = 'average')
    :param seed: does nothing, here for convention
    :param kwargs: arguments passed through to AgglomerativeClustering()
    :return:
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='precomputed', **kwargs)
    clustering.fit(dist)

    return clustering
