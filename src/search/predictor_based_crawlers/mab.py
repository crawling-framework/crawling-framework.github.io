import numpy as np
import random

from base.cgraph import MyGraph
from crawlers.cadvanced import NodeFeaturesUpdatableCrawlerHelper
from crawlers.declarable import Declarable, declaration_to_filename
from search.predictor_based_crawlers.predictor_based import PredictorBasedCrawler
from search.predictors.simple_predictors import Predictor


class MultiPredictor(Predictor):
    """ Wrapper for multiple predictors in a bunch.
    """
    def __init__(self, predictors: (list, tuple), name=None, **kwargs):
        """
        :param predictors: list of predictors or their declarations
        :param name: name for plotting
        """
        # Translate to declarations, sort (to avoid file naming dependency on the order), and build predictors
        _predictors = [p.declaration if isinstance(p, Predictor) else p for p in predictors]
        _predictors.sort(key=declaration_to_filename)
        _predictors = [Declarable.from_declaration(p) for p in _predictors]

        # _predictors = []
        # for p in predictors:
        #     _p = Declarable.from_declaration(p.declaration if isinstance(p, Predictor) else p)
        #     _predictors.append(_p)

        name = ",".join(type(p).__name__ for p in _predictors) if name is None else name
        super().__init__(predictors=_predictors, name=name, **kwargs)
        self.predictors = _predictors

    @property
    def used_attributes(self):
        """ Union of all attributes used in predictors. """
        used_attributes = []
        for p in self.predictors:
            used_attributes.extend(p.used_attributes)
        return list(set(used_attributes))

    def extract_features(self, node, crawler_helper: NodeFeaturesUpdatableCrawlerHelper):
        """ Returns list of feature vectors """
        return [p.extract_features(node, crawler_helper) for p in self.predictors]

    def train(self, Xs, ys):
        """ Train all subpredictors. """
        # for p, xs in zip(self.predictors, np.transpose(Xs)): - smth strange
        # Transpose Xs manually
        pred_xs = [[] for p in self.predictors]
        for X in Xs:
            for i, x in enumerate(X):
                pred_xs[i].append(x)
        Xs = pred_xs
        for p, xs in zip(self.predictors, Xs):
            p.train(xs, ys)

    def predict_score(self, X):
        """ Returns list of scores. """
        return [p.predict_score(x) for p, x in zip(self.predictors, X)]

    def __len__(self):
        return len(self.predictors)

    def __iter__(self):
        for p in self.predictors:
            yield p


class MultiPredictorCrawler(PredictorBasedCrawler):
    """ Uses several predictors and MAB strategy on them
    """
    short = 'MultiPredCrawler'

    def __init__(self, graph: MyGraph, predictor: MultiPredictor, oracle, name=None, **kwargs):
        """
        :param predictors: multi-predictor
        """
        assert 'predictor' not in kwargs, "Should use 'predictors' instead of 'predictor'"
        self.predictor = predictor
        # Additional dict of estimated scores for each predictor
        self._node_scores = {}  # {node -> [list of scores]}

        super(MultiPredictorCrawler, self).__init__(
            graph, predictor=predictor, oracle=oracle, name=name, **kwargs)

    def re_estimate_nodes(self, nodes):
        """ Same as in superclass, but manages both _node_score and _node_scores. """
        og = self.observed_graph

        # All 1-degree neighbors of a node have same score - we store them to speed up
        one_degree_neighbor_scores = dict()  # {node -> score of its 1 degree neighbors}
        for n in nodes:
            # check if can we use computed score
            if og.deg(n) == 1:
                neighbor = next(og.neighbors(n))
                if neighbor in one_degree_neighbor_scores:
                    self._node_score[n], self._node_scores[n] = one_degree_neighbor_scores[neighbor]
                else:
                    self.estimate_node(n)
                    one_degree_neighbor_scores[neighbor] = self._node_score[n], self._node_scores[n]
            else:
                self.estimate_node(n)

    def estimate_node(self, node) -> float:
        raise NotImplementedError("Must be implemented in subclass")

    def train_predictor(self, Xs, ys):
        self.predictor.train(Xs, ys)


class AverageModelsMultiPredictorCrawler(MultiPredictorCrawler):
    short = "AvgMultiPredCrawler"

    def __init__(self, graph: MyGraph, predictor, oracle, name=None, **kwargs):
        super(AverageModelsMultiPredictorCrawler, self).__init__(
            graph, predictor=predictor, oracle=oracle, name=name, **kwargs)

    def estimate_node(self, node) -> float:
        # Average over predictors
        scores = self.predictor.predict_score(self.predictor.extract_features(node, self))
        self._node_scores[node] = scores
        self._node_score[node] = score = np.mean(scores)
        return score


class MABCrawler(MultiPredictorCrawler):
    short = "MABCrawler"

    def __init__(self, graph: MyGraph, predictor: MultiPredictor, name=None, **kwargs):
        # Weights are initially uniform
        self._weights = [1 / len(predictor)] * len(predictor)

        super(MABCrawler, self).__init__(
            graph, predictor=predictor, name=name, **kwargs)

    def estimate_node(self, node) -> float:
        # Weighted average over predictors
        scores = self.predictor.predict_score(self.predictor.extract_features(node, self))
        self._node_scores[node] = scores
        self._node_score[node] = score = sum(w * s for w, s in zip(self._weights, scores))
        return score


class ExponentialDynamicWeightsMultiPredictorCrawler(MABCrawler):
    short = "ExpDynWeightMABCrawler"

    def __init__(self, graph: MyGraph, uniform_distribution=False, name=None, **kwargs):
        """
        :param uniform_distribution: if True use uniform additive for weights at each step.
        """
        super(ExponentialDynamicWeightsMultiPredictorCrawler, self).__init__(
            graph, uniform_distribution=uniform_distribution, name=name, **kwargs)
        self.uniform_distribution = uniform_distribution

    def crawl(self, seed) -> list:
        """ Update weights based on whether seed to be crawled is target
        """
        res = super().crawl(seed)
        is_target = self.oracle(seed, self.orig_graph)

        # Update weights according to predictors' scores
        scores = self._node_scores[seed]
        loc_sum = 0
        if self.uniform_distribution:
            additive = 1 / len(self.predictor)
        else:
            additive = 0
        for i in range(len(self.predictor)):
            if is_target == 1:
                loc_weight = self._weights[i] * 2 ** ((scores[i] - 0.5) * 3)
            elif is_target == 0:
                loc_weight = self._weights[i] * 2 ** ((0.5 - scores[i]) * 3)
            else:  # not labeled
                loc_weight = self._weights[i]
                # raise NotImplementedError("Not expected that node label is undefined")

            loc_sum += loc_weight + additive
            self._weights[i] = loc_weight + additive
        self._weights = [weight / loc_sum for weight in self._weights]

        return res


class FollowLeaderMABCrawler(MABCrawler):
    short = "FollowLeadMABCrawler"

    def __init__(self, graph: MyGraph, name=None, **kwargs):
        super(FollowLeaderMABCrawler, self).__init__(graph, name=name, **kwargs)
        self._sum_models_regret = [0] * len(self.predictor)

    def crawl(self, seed) -> list:
        """ Update weights based on whether seed to be crawled is target
        """
        res = super().crawl(seed)
        is_target = self.oracle(seed, self.orig_graph)

        # Update models regrets
        scores = self._node_scores[seed]
        for i, predictor in enumerate(self.predictor):
            if is_target == 1:
                self._sum_models_regret[i] -= (scores[i] - 0.5)
            elif is_target == 0:
                self._sum_models_regret[i] -= (0.5 - scores[i])
            else:
                pass
                # raise NotImplementedError("Not expected that node label is undefined")

        # Set the weight of predictor with minimal regret to 1
        min_regret = 1e9
        ix = -1
        for i in range(len(self.predictor)):
            if self._sum_models_regret[i] < min_regret:
                min_regret = self._sum_models_regret[i]
                ix = i
            self._weights[i] = 0
        self._weights[ix] = 1

        return res


class BetaDistributionMultiPredictorCrawler(MABCrawler):
    short = "BetaDistrMABCrawler"

    def __init__(self, graph: MyGraph, beta_distr_param_thr=2, name=None, **kwargs):
        """
        :param beta_distr_param_thr:
        """
        super(BetaDistributionMultiPredictorCrawler, self).__init__(
            graph, beta_distr_param_thr=beta_distr_param_thr, name=name, **kwargs)
        self.beta_distr_param_thr = beta_distr_param_thr

        self._weights = [0] * len(self.predictor)
        self._predictors_beta_distr_params = [[2, 2]] * len(self.predictor)

        self._curr_predictor_ix = random.randint(0, len(self.predictor) - 1)
        self._weights[self._curr_predictor_ix] = 1

    def crawl(self, seed) -> list:
        """ Update weights based on whether seed to be crawled is target
        """
        res = super().crawl(seed)
        is_target = self.oracle(seed, self.orig_graph)

        # Adjust beta distribution parameters
        param = self._predictors_beta_distr_params[self._curr_predictor_ix]
        thr = self.beta_distr_param_thr
        if is_target == 1:
            if param[0] + param[1] < thr:
                param[0] += 1
            else:
                param[0] = (param[0] + 1) * thr / (thr + 1)
        elif is_target == 0:
            if param[0] + param[1] < thr:
                param[1] += 1
            else:
                param[1] = (param[1] + 1) * thr / (thr + 1)
        else:
            pass
            # raise NotImplementedError("Not expected that node label is undefined")

        # Set the leading predictor according to beta distribution random variable
        r = -1
        for ix in range(len(self.predictor)):
            loc_random = np.random.beta(*self._predictors_beta_distr_params[ix])
            if loc_random > r:
                r = loc_random
                self._curr_predictor_ix = ix

        for ix in range(len(self.predictor)):
            self._weights[ix] = 0
        self._weights[self._curr_predictor_ix] = 1

        return res
