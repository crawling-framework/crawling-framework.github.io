import logging
from operator import itemgetter

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import _get_weights

import numpy as np


from base.cgraph import MyGraph
from crawlers.cbasic import Crawler

logger = logging.getLogger(__name__)


def _KNeighborsRegressor_predict(neigh_dist, neigh_ind, knn_model):
    """
    Copy of KNeighborsRegressor.predict() with precomputed kNN.

    :param neigh_dist:
    :param neigh_ind:
    :param knn_model:
    :return: y prediction
    """
    weights = _get_weights(neigh_dist, knn_model.weights)

    _y = knn_model._y
    if _y.ndim == 1:
        _y = _y.reshape((-1, 1))

    y_pred = np.empty((1, _y.shape[1]), dtype=np.float64)
    denom = np.sum(weights, axis=1)

    for j in range(_y.shape[1]):
        num = np.sum(_y[neigh_ind, j] * weights, axis=1)
        y_pred[:, j] = num / denom

    if _y.ndim == 1:
        y_pred = y_pred.ravel()

    return y_pred


class KNN_UCB_Crawler(Crawler):
    """
    Implementation of KNN-UCB crawling strategy based on multi-armed bandit approach.
    "A multi-armed bandit approach for exploring partially observed networks" (2019)
    https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    short = 'KNN-UCB'

    def __init__(self, graph: MyGraph, initial_seed: int=-1,
                 alpha: float=0.5, k: int=10, n0: int=20, n_features: int=4, **kwargs):
        """
        :param graph: original graph
        :param initial_seed: start node
        :param alpha: search ratio
        :param k: number of nearest neighbors to estimate expected reward
        :param n0: number of starting random nodes, default 20
        :param n_features: number of features to use, default 4
        """
        # TODO append features to params to differ them in filenames
        if initial_seed != -1 and n0 < 1:
            kwargs['initial_seed'] = initial_seed
        super().__init__(graph, alpha=alpha, k=k, n0=n0, n_features=n_features, **kwargs)

        self._node_feature = {}  # node_id -> (feature vector, observed_reward)
        self.n_features = n_features

        # pick a random seed from original graph
        if len(self._observed_set) == 0 and n0 < 1:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)
            self._node_feature[initial_seed] = [[0] * self.n_features, 0]

        self.k = k
        self.alpha = alpha
        self.n0 = n0
        self._random_pool = [] if n0 == -1 else self._orig_graph.random_nodes(n0)

        self._knn_model = KNeighborsRegressor(n_neighbors=self.k, weights='distance')
        self._fit_period = 1  # fit kNN model once in a period dynamically changing

    def _expected_reward(self, node: int) -> float:
        """
        :param node: node id
        :return: expected reward for observed node
        """
        # Expected reward predicted by kNN regressor
        feature = self._node_feature[node][0]
        neigh_dist, neigh_ind = self._knn_model.kneighbors([feature])
        f = _KNeighborsRegressor_predict(neigh_dist, neigh_ind, self._knn_model)

        # Average distance to kNN
        sigma = np.mean(neigh_dist)

        return f + self.alpha * sigma

    def _compute_feature(self, node: int):
        """
        Calculates feature vector for the node:
        obs_degree - observed degree of the node
        avg_neigh_degree - average degree of adjacent crawled nodes
        max_neigh_degree - maximum degree of adjacent crawled nodes
        crawled_neigh_frac - fraction of crawled neighbors
        # n_triangles - the number of triangles containing the current node

        :param node: the node for which the feature vector is calculated
        """
        obs_degree = self._observed_graph.deg(node)

        max_neigh_degree = 0
        avg_neigh_degree = 0
        crawled_neigh_frac = 0
        for n in self._observed_graph.neighbors(node):
            if n in self._crawled_set:
                crawled_neigh_frac += 1 / obs_degree
            deg = self._observed_graph.deg(n)
            if deg > max_neigh_degree:
                max_neigh_degree = deg
            avg_neigh_degree += deg / obs_degree

        res = [obs_degree, avg_neigh_degree, max_neigh_degree, crawled_neigh_frac][:self.n_features]
        return res

    def crawl(self, seed: int):
        res = super().crawl(seed)
        # Obtained reward = the number of newly open nodes
        self._node_feature[seed] = [None, len(res)]  # will be updated
        for n in res:
            self._node_feature[n] = [None, 0]  # will be updated

        # Update features for seed, its neighbors, and 2nd neighborhood
        to_be_updated = {seed}
        for n in self._observed_graph.neighbors(seed):
            to_be_updated.add(n)
            to_be_updated.update(self._observed_graph.neighbors(n))  # TODO seems to have no effect experimentally

        for n in to_be_updated:
            self._node_feature[n][0] = self._compute_feature(n)

        return res

    def next_seed(self):
        crawled = len(self._crawled_set)
        self._fit_period = 1 + crawled // 20

        # Yield random node first n0 times
        if self.n0 > crawled:
            return self._random_pool[crawled]

        if crawled == 0:
            return next(iter(self._observed_set))

        # Fit kNN model
        if crawled % self._fit_period == 0:
            # X, y = zip(*self._node_feature.values())  # all nodes
            X, y = zip(*[self._node_feature[n] for n in self._crawled_set])  # crawled nodes
            self._knn_model = KNeighborsRegressor(n_neighbors=min(len(y), self.k), weights='distance')
            self._knn_model.fit(X, y)

        # Choosing the best node from observed nodes for crawling
        best_node = -1
        best_expected_reward = -1
        for n in self._observed_set:
            r = self._expected_reward(n)
            if r > best_expected_reward:
                best_node = n
                best_expected_reward = r

        return best_node
