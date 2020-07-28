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

    y_pred = np.empty((neigh_ind.shape[0], _y.shape[1]), dtype=np.float64)
    denom = np.sum(weights, axis=1)

    for j in range(_y.shape[1]):
        num = np.sum(_y[neigh_ind, j] * weights, axis=1)
        y_pred[:, j] = num / denom

    if _y.ndim == 1:
        y_pred = y_pred.ravel()

    return y_pred


FEATURES = [
    'OD',   # observed degree of the node
    'CC',   # clustering coefficient
    'CNF',  # crawled neighbors fraction (at the moment of crawling)
    'AND',  # average neighbour degree
    'MND',  # maximum neighbour degree
]


class KNN_UCB_Crawler(Crawler):
    """
    Implementation of KNN-UCB crawling strategy based on multi-armed bandit approach.
    "A multi-armed bandit approach for exploring partially observed networks" (2019)
    https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    short = 'KNN-UCB'

    def __init__(self, graph: MyGraph, initial_seed: int=-1,
                 alpha: float=0.5, k: int=30, n0: int=0, features: list=['OD'], **kwargs):
        """
        :param graph: original graph
        :param initial_seed: start node
        :param alpha: exploration coefficient, default 0.5
        :param k: number of nearest neighbors to estimate expected reward, default 30
        :param n0: number of starting random nodes, default 0
        :param n_features: number of features to use, default 1
        """
        # TODO append features to params to differ them in filenames?
        if initial_seed != -1 and n0 < 1:
            kwargs['initial_seed'] = initial_seed

        features = sorted(features)
        super().__init__(graph, alpha=alpha, k=k, n0=n0, features=features, **kwargs)

        self.features = features
        self._node_feature = {}  # node_id -> (feature dict, observed_reward)
        self._max_deg = 1  # max degree in observed graph, for feature normalization
        self.node_clust = {}  # node_id -> clustering coeff

        # pick a random seed from original graph
        if len(self._observed_set) == 0 and n0 < 1:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        for n in self.nodes_set:
            self.node_clust[n] = self._observed_graph.clustering(n)
            self._node_feature[n] = [self._compute_feature(n), None]

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
        feature = np.array([[self._node_feature[node][0][f] for f in self.features] for node in node_list])  # FIXME quite a lot time for conversion to numpy array
        neigh_dist, neigh_ind = self._knn_model.kneighbors(feature)
        f = _KNeighborsRegressor_predict(neigh_dist, neigh_ind, self._knn_model)

        # Average distance to kNN
        sigma = np.mean(neigh_dist, axis=1)

        return f.reshape(len(node_list)) + self.alpha * sigma

    def _compute_feature(self, node: int):
        """
        Calculates node feature dict for all features.

        :param node: the node for which the feature vector is calculated
        :return: dict {feature -> value}
        """
        res = {}
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

        res['OD'] = obs_degree / self._max_deg
        res['AND'] = avg_neigh_degree / self._max_deg
        res['MND'] = max_neigh_degree / self._max_deg
        # CNF is updated until node is not crawled
        res['CNF'] = crawled_neigh_frac if node not in self._crawled_set else self._node_feature[node][0]['CNF']

        res['CC'] = self.node_clust[node]  # self._observed_graph.clustering(node)
        return res

    def crawl(self, seed: int):
        res = super().crawl(seed)
        self._max_deg = max(self._max_deg, self._observed_graph.deg(seed))

        # Obtained reward = the number of newly open nodes
        self._node_feature[seed][1] = len(res)
        for n in res:
            self._node_feature[n] = [None, 0]  # feature will be updated

        # Update CC for seed and its neighbors
        upd = []
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                upd.append(n)
        self.node_clust[seed] = self._observed_graph.clustering(seed)
        for n in upd:
            d = self._observed_graph.deg(n)
            conn_neigs = 0
            for neigh in self._observed_graph.neighbors(n):
                if self._observed_graph.has_edge(seed, neigh):
                    conn_neigs += 1

            if n not in self.node_clust:
                self.node_clust[n] = 0
            self.node_clust[n] = (self.node_clust[n] * (d-1) * (d-1) / 2 + conn_neigs) / d / d * 2

        # Update features for seed, its neighbors, and 2nd neighborhood
        to_be_updated = {seed}
        for n in self._observed_graph.neighbors(seed):
            to_be_updated.add(n)
            # to_be_updated.update(self._observed_graph.neighbors(n))  # TODO seems to have no effect experimentally

        for n in to_be_updated:
            assert n in self._node_feature
            self._node_feature[n][0] = self._compute_feature(n)

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
            # X, y = zip(*self._node_feature.values())  # all nodes
            # print([self._node_feature[2][0][f] for f in self.features])
            X, y = zip(*[([self._node_feature[n][0][f] for f in self.features], self._node_feature[n][1]) for n in self._crawled_set])  # crawled nodes
            # X = self._scaler.fit_transform(X)
            self._knn_model = KNeighborsRegressor(n_neighbors=min(len(y), self.k), weights='distance')
            self._knn_model.fit(X, y)

        self._fit_period = 1  # + crawled // 2

        # Choosing the best node from observed nodes for crawling
        candidates = list(self._observed_set)
        rs = self._expected_rewards(candidates)
        best_node = candidates[np.argmax(rs)]

        return best_node
