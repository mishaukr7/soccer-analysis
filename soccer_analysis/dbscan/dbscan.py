# -*- coding: utf-8 -*-

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise

import numpy as np
import math
from numba import njit

UNCLASSIFIED = -2
NOISE = -1


@njit
def _dist(p: np.ndarray, q: np.ndarray):
    return math.sqrt(np.power(p-q, 2).sum())


@njit
def _eps_neighborhood(p: np.ndarray, q: np.ndarray, eps: float):
    return _dist(p, q) < eps

@njit
def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:,point_id], m[:,i], eps):
            seeds.append(i)
    return seeds

@njit
def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = -1
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == -2 or \
                       classifications[result_point] == -1:
                        if classifications[result_point] == -2:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

@njit
def dbscan(m, classifications, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN

    scikit-learn probably has a better implementation

    Uses Euclidean Distance as the measure

    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster

    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id: int = 1
    n_points: int = m.shape[1]
    for point_id in range(0, n_points):
        if classifications[point_id] == -2:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1


def test_dbscan():
    m = np.matrix('1 1.2 0.8 3.7 3.9 3.6 10; 1.1 0.8 1 4 3.9 4.1 10')
    eps = 0.5
    min_points = 2
    print(m)
    classifications = np.full(m.shape[1], -2, dtype=int)
    result = dbscan(m, classifications, eps, min_points)
    expected = [1, 1, 1, 2, 2, 2, -1]
    assert np.array_equal(result,expected), f'{result} != {expected}'


def test_dbscan_instance():
    x = np.array(
                [[1, 1.2, 0.8, 3.7, 3.9, 3.6, 10],
                 [1.1, 0.8, 1, 4, 3.9, 4.1, 10],
                 [1.1, 0.8, 1, 4, 3.9, 4.1, 10]]
                 ).T
    eps = 0.5
    min_points = 2
    print(x)
    model = DBSCAN(eps, min_points)
    model.fit(x)
    result = model.labels_
    expected = np.array([1, 1, 1, 2, 2, 2, -1])
    assert np.array_equal(result,expected), f'{result} != {expected}'


class DBSCAN:
    eps: float
    min_samples: int
    labels_: np.ndarray

    def __init__(self, eps=0.5, min_samples=5) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, x: np.ndarray):
        labels = np.full(x.shape[1], -2, dtype=int)
        dbscan(x, labels, self.eps, self.min_samples)
        self.labels_ = labels
        return self


if __name__ == '__main__':
    test_dbscan()
    test_dbscan_instance()
