from math import log

import numpy as np
from sklearn.linear_model import LinearRegression

from base.cgraph import MyGraph
from crawlers.cbasic import CrawlerWithInitialSeed
from crawlers.ml.with_features import CrawlerWithFeatures


REGRESSORS = ['ElasticNet', 'Lasso', 'LinearRegression', 'KNeighborsRegressor', 'SVR']


class RegressionRewardCrawler(CrawlerWithFeatures, CrawlerWithInitialSeed):
    """
    Crawler uses a regression model trained on crawled nodes features to predict reward for observed nodes. The next
    seed at each step is chosen as the node with the highest reward.
    """
    short = 'RegReward'

    def __init__(self, graph: MyGraph, initial_seed: int=-1, tau=-1, features: list=['OD'], regr: str='LinearRegression', regr_args: dict={}, **kwargs):
        """
        :param graph: original graph
        :param initial_seed: start node
        :param tau: sliding window size, number of last crawled nodes used for learning and prediction, default use all (-1)
        :param features: list of features to use (see FEATURES), default ['OD']
        :param regr: which regressor to use for reward prediction, default LinearRegression
        :param regr_args: dict of arguments for the regressor
        """
        assert regr in REGRESSORS
        super(RegressionRewardCrawler, self).__init__(graph=graph, initial_seed=initial_seed, tau=tau, features=features, regr=regr, regr_args=regr_args, **kwargs)
        self._node_reward = {}  # node_id -> observed_reward

        # Add some extra args depending on the regressor
        if regr == 'LinearRegression':
            regr_args['copy_X'] = False
        elif regr == 'SVR':
            if 'gamma' not in regr_args:
                regr_args['gamma'] = 'auto'

        # FIXME for KNeighborsRegressor n_neighbors must be > n_samples, so we need either update it or not use for first n steps

        self._regressor = eval(regr)(**regr_args)

        self._fit_period = 1  # fit model once in a period dynamically changing
        # self._scaler = StandardScaler()

    def _expected_rewards(self, node_list: list):
        """
        :param node_list: list of node ids
        :return: expected rewards for the nodes
        """
        # Expected reward predicted by Linear regressor
        # TODO compute for nodes in self._nodes_learning_queue
        feature = np.array([self._node_feature[n] for n in node_list])  # FIXME quite a lot time for conversion to numpy array
        r = self._regressor.predict(feature)
        return r

    def crawl(self, seed: int):
        res = super().crawl(seed)

        self._node_reward[seed] = log(1 + len(res))  # Obtained reward = the number of newly open nodes
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
            self._regressor.fit(X, y)
            # print(self._regressor.coef_)

        self._fit_period = 1  # + crawled // 2

        # Choosing the best node from observed nodes for crawling
        candidates = list(self._observed_set)
        rs = self._expected_rewards(candidates)
        return candidates[np.argmax(rs)]