import numpy as np
from sklearn.linear_model import LinearRegression

from base.cgraph import MyGraph
from crawlers.ml.with_features import CrawlerWithFeatures


class LinReg_Crawler(CrawlerWithFeatures):
    """
    Use Linear Regression to predict reward.
    """
    short = 'LinReg'

    def __init__(self, graph: MyGraph, initial_seed: int=-1, tau=-1, features: list=['OD'], **kwargs):
        """
        :param graph: original graph
        :param initial_seed: start node
        :param tau: sliding window size, number of last crawled nodes used for learning and prediction, default use all (-1)
        :param features: list of features to use (see FEATURES), default ['OD']
        """
        if initial_seed != -1:
            kwargs['initial_seed'] = initial_seed

        super().__init__(graph=graph, tau=tau, features=features, **kwargs)

        self._node_reward = {}  # node_id -> observed_reward

        # pick a random seed from original graph
        if len(self._observed_set) == 0:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        super().init()  # compute features for observed nodes

        self._predictor = LinearRegression(fit_intercept=True, copy_X=False)
        self._fit_period = 1  # fit model once in a period dynamically changing
        # self._scaler = StandardScaler()

    def _expected_rewards(self, node_list: list):
        """
        :param node_list: list of node ids
        :return: expected rewards for the nodes
        """
        # Expected reward predicted by Linear regressor
        feature = np.array([self._node_feature[n] for n in node_list])  # FIXME quite a lot time for conversion to numpy array
        r = self._predictor.predict(feature)
        return r

    def crawl(self, seed: int):
        res = super().crawl(seed)

        # Obtained reward = the number of newly open nodes
        self._node_reward[seed] = len(res)
        for n in res:
            self._node_reward[n] = 0

        return res

    def next_seed(self):
        crawled = len(self._crawled_set)

        # # Yield random node first n0 times
        # if self.n0 > crawled:
        #     return self._random_pool[crawled]

        if crawled == 0:
            return next(iter(self._observed_set))

        # Fit kNN model
        if crawled % self._fit_period == 0:
            X = [self._node_feature[n] for n in self._nodes_learning_queue]  # crawled nodes within a learning queue
            y = [self._node_reward[n] for n in self._nodes_learning_queue]
            # X = self._scaler.fit_transform(X)
            self._predictor.fit(X, y)
            print(self._predictor.coef_)

        self._fit_period = 1  # + crawled // 2

        # Choosing the best node from observed nodes for crawling
        candidates = list(self._observed_set)
        rs = self._expected_rewards(candidates)
        return candidates[np.argmax(rs)]