import logging
from collections import deque
from math import log
import numpy as np

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler

logger = logging.getLogger(__name__)


FEATURES = [
    'OD',   # observed degree of the node
    'CC',   # clustering coefficient
    'CNF',  # crawled neighbors fraction (at the moment of crawling)
    'AND',  # average neighbour degree
    'MND',  # median neighbour degree
]


class CrawlerWithFeatures(Crawler):
    """
    Computes and stores a feature vector for each node during the crawling.
    """
    def __init__(self, graph: MyGraph, features: list=['OD'], tau=-1, **kwargs):
        """
        :param features: list of feature codes to compute, default ['OD']
        :param tau: sliding window size, number of last crawled nodes used for learning and prediction, default use all (-1)
        :param kwargs: additional args for Crawler constructor including graph, name, initial_seed, etc
        """
        print('init CrawlerWithFeatures')
        assert all([f in FEATURES for f in features])
        features = sorted(features)
        super().__init__(graph, features=features, tau=tau, **kwargs)
        print('CWF after super')

        self.features = features
        self.tau = tau

        self._nodes_learning_queue = deque(maxlen=self.tau if self.tau != -1 else None)  # limited size queue of nodes used for learning and prediction
        self._node_feature = {}  # node_id -> feature vector
        self._node_clust = {}  # node_id -> clustering coeff
        self._max_deg = 1  # max degree in observed graph, for feature normalization

        self._neigh_deg_features = 'AND' in self.features or 'MND' in self.features  # flag whether to compute AND, MND

        # Compute features for observed nodes before crawling.
        for n in self._observed_set:
            self._node_clust[n] = self._observed_graph.clustering(n)
            self.update_feature(n)

    def update_feature(self, node: int):
        """
        Calculate and update node feature vector for specified node.

        :param node: the node for which the feature vector is calculated
        :return:
        """
        res = {}
        obs_degree = self._observed_graph.deg(node)

        # FIXME we compute all features which is redundant, may be very consuming
        crawled_neigh_frac = 0
        for n in self._observed_graph.neighbors(node):
            if n in self._crawled_set:
                crawled_neigh_frac += 1 / obs_degree

        if self._neigh_deg_features:
            neigh_degrees = np.array([self._observed_graph.deg(n) for n in self._observed_graph.neighbors(node)])
            avg_neigh_degree = np.average(neigh_degrees) if len(neigh_degrees) > 0 else 0
            median_neigh_degree = np.median(neigh_degrees) if len(neigh_degrees) > 0 else 0
            res['AND'] = avg_neigh_degree / self._max_deg
            res['MND'] = median_neigh_degree / self._max_deg

        res['OD'] = log(1 + obs_degree)
        res['CNF'] = crawled_neigh_frac
        res['CC'] = self._node_clust[node]  # self._observed_graph.clustering(node)

        self._node_feature[node] = [res[f] for f in self.features]

    def crawl(self, seed: int):
        res = super().crawl(seed)
        self._max_deg = max(self._max_deg, self._observed_graph.deg(seed))

        # Nodes to update - observed neighbors
        to_be_updated = set()
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                to_be_updated.add(n)

        # Update CC for seed and its neighbors
        self._node_clust[seed] = self._observed_graph.clustering(seed)
        for n in to_be_updated:
            d = self._observed_graph.deg(n)
            conn_neigs = 0
            for neigh in self._observed_graph.neighbors(n):
                if self._observed_graph.has_edge(seed, neigh):
                    conn_neigs += 1

            if n not in self._node_clust:
                self._node_clust[n] = 0
            self._node_clust[n] = (self._node_clust[n] * (d-1) * (d-1) / 2 + conn_neigs) / d / d * 2

        # Update features for seed, its neighbors(, and 2nd neighborhood). We update only observed nodes
        # to_be_updated.add(seed)
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                # to_be_updated.add(n)
                to_be_updated.update(self._observed_graph.neighbors(n))  # TODO seems to have no effect experimentally

        # Update features for observed nodes only
        for n in to_be_updated:
            if n in self._observed_set:
                self.update_feature(n)

        # Update learning queue
        self._nodes_learning_queue.append(seed)

        return res
