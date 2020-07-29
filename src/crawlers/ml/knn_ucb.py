import logging
from collections import deque

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import _get_weights

import numpy as np

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler
from crawlers.ml.with_features import CrawlerWithFeatures

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

    y_pred = np.empty((neigh_ind.shape[0], _y.shape[1]), dtype=np.float64)
    denom = np.sum(weights, axis=1)

    for j in range(_y.shape[1]):
        num = np.sum(_y[neigh_ind, j] * weights, axis=1)
        y_pred[:, j] = num / denom

    if _y.ndim == 1:
        y_pred = y_pred.ravel()

    return y_pred


class KNN_UCB_Crawler(CrawlerWithFeatures):
    """
    Implementation of KNN-UCB crawling strategy based on multi-armed bandit approach.
    "A multi-armed bandit approach for exploring partially observed networks" (2019)
    https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    short = 'KNN-UCB'

    def __init__(self, graph: MyGraph, initial_seed: int=-1,
                 alpha: float=0.5, k: int=30, n0: int=0, tau: int=-1, features: list=['OD'], **kwargs):
        """
        :param graph: original graph
        :param initial_seed: start node
        :param alpha: exploration coefficient, default 0.5
        :param k: number of nearest neighbors to estimate expected reward, default 30
        :param n0: number of starting random nodes, default 0
        :param tau: sliding window size, number of last crawled nodes used for learning and prediction, default use all (-1)
        :param features: list of features to use (see FEATURES), default ['OD']
        """
        if initial_seed != -1 and n0 < 1:
            kwargs['initial_seed'] = initial_seed

        super().__init__(graph=graph, alpha=alpha, k=k, n0=n0, tau=tau, features=features, **kwargs)

        self._node_reward = {}  # node_id -> observed_reward

        # pick a random seed from original graph
        if len(self._observed_set) == 0 and n0 < 1:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        super().init()  # compute features for observed nodes

        self.k = k
        self.alpha = alpha
        self.n0 = n0
        self._random_pool = [] if n0 == -1 else self._orig_graph.random_nodes(n0)

        self._knn_model = KNeighborsRegressor(n_neighbors=self.k, weights='distance', n_jobs=None)
        self._fit_period = 1  # fit kNN model once in a period dynamically changing
        # self._scaler = StandardScaler()

    def _expected_rewards(self, node_list: list):
        """
        :param node_list: list of node ids
        :return: expected rewards for the nodes
        """
        # Expected reward predicted by kNN regressor
        # feature = self._scaler.transform([self._node_feature[node][0] for node in node_list])
        feature = np.array([self._node_feature[n] for n in node_list])  # FIXME quite a lot time for conversion to numpy array
        neigh_dist, neigh_ind = self._knn_model.kneighbors(feature)
        f = _KNeighborsRegressor_predict(neigh_dist, neigh_ind, self._knn_model)

        # Average distance to kNN
        sigma = np.mean(neigh_dist, axis=1)

        return f.reshape(len(node_list)) + self.alpha * sigma

    def crawl(self, seed: int):
        res = super().crawl(seed)

        # Obtained reward = the number of newly open nodes
        self._node_reward[seed] = len(res)
        for n in res:
            self._node_reward[n] = 0

        return res

    def next_seed(self):
        crawled = len(self._crawled_set)

        # Yield random node first n0 times
        if self.n0 > crawled:
            return self._random_pool[crawled]

        if crawled == 0:
            return next(iter(self._observed_set))

        # Fit kNN model
        if crawled % self._fit_period == 0:
            X = [self._node_feature[n] for n in self._nodes_learning_queue]  # crawled nodes within a learning queue
            y = [self._node_reward[n] for n in self._nodes_learning_queue]
            # X = self._scaler.fit_transform(X)
            self._knn_model = KNeighborsRegressor(n_neighbors=min(len(y), self.k), weights='distance')
            self._knn_model.fit(X, y)

        self._fit_period = 1  # + crawled // 2

        # Choosing the best node from observed nodes for crawling
        candidates = list(self._observed_set)
        rs = self._expected_rewards(candidates)
        return candidates[np.argmax(rs)]
