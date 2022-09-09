from base.cgraph import MyGraph
from crawlers.cadvanced import NodeFeaturesUpdatableCrawlerHelper
from crawlers.cbasic import InitialSeedCrawlerHelper
from search.crawler_statistics_helper import StatisticsCrawlerHelper
from search.oracles import Oracle
from search.predictors.simple_predictors import Predictor


class PredictorBasedCrawler(
    NodeFeaturesUpdatableCrawlerHelper, StatisticsCrawlerHelper, InitialSeedCrawlerHelper):
    """
    Parent class for crawlers based on a predictor that can estimate the score (probability) of an
    observed node to be target.
    """
    short = 'PredCrawler'

    def __init__(self, graph: MyGraph, predictor: Predictor, oracle: Oracle,
                 training_strategy=None, re_estimate='after_train',
                 initial_seed=None, name=None, **kwargs):
        """
        :param graph:
        :param predictor: node predictor model (or list of predictors).
        :param oracle: target node detector, returns 1/0/None.
        :param training_strategy: strategy to train predictor, one of 'boost', 'online'.
        :param re_estimate_all: how many nodes to re-estimate after each seed crawling: if True,
         estimate all observed nodes which is precise, if False, seed's neighbors only which is
         faster but not precise. Default True.
        :param initial_seed: one of [None, <integer>, 'target']
        :param kwargs: args for subclass
        """
        assert re_estimate in {'always', 'after_train', 'neighbors'}
        name = f"{self.short}[{predictor},{training_strategy}]" if name is None else name
        self.oracle = oracle  # needed here, used in InitialSeedCrawlerHelper constructor
        super().__init__(
            graph, predictor=predictor, oracle=oracle, training_strategy=training_strategy,
            re_estimate=re_estimate, initial_seed=initial_seed,
            attributes=predictor.used_attributes, name=name, **kwargs)

        self.predictor = predictor
        self.training_strategy = training_strategy
        self.re_estimate = re_estimate

        self._last_seed = None
        self._just_trained = False
        self._node_score = {}  # predictor estimations for observed nodes
        for n in self.observed_set:
            self.estimate_node(n)

    def __str__(self):
        return self.name

    def estimate_node(self, node) -> float:
        """ Estimate class=1 score (probability) by the predictor.
        """
        # This is most time-consuming call
        score = self.predictor.predict_score(self.predictor.extract_features(node, self))
        self._node_score[node] = score
        return score

    def crawl(self, seed) -> list:
        res = NodeFeaturesUpdatableCrawlerHelper.crawl(self, seed)
        self._last_seed = seed
        self._just_trained = False
        return res

    def re_estimate_nodes(self, nodes):
        """ Recompute predictor estimations for the given nodes. """
        og = self.observed_graph

        # All 1-degree neighbors of a node have same score - we store them to speed up
        one_degree_neighbor_scores = dict()  # {node -> score of its 1 degree neighbors}
        for n in nodes:
            # check if can we use computed score
            if og.deg(n) == 1:
                neighbor = next(og.neighbors(n))
                if neighbor in one_degree_neighbor_scores:
                    self._node_score[n] = one_degree_neighbor_scores[neighbor]
                else:
                    one_degree_neighbor_scores[neighbor] = self.estimate_node(n)
            else:
                self.estimate_node(n)

    def train_predictor(self, Xs, ys):
        """ Train predictor using the training data; update estimations for all observed nodes.
        """
        self.predictor.train(Xs, ys)
        self._just_trained = True

    def next_seed(self):
        """ Choose the next seed as one with maximal score.
        """
        # Train predictor if needed
        if self.training_strategy is not None:
            self.training_strategy(self)

        # Update nodes estimations
        if self.re_estimate == 'always' or (
                self.re_estimate == 'after_train' and self._just_trained):
            # Compute score for all observed nodes.
            # NOTE: this variant is the most time cost
            nodes = self.observed_set
        else:
            # Recompute only for neighbors of last crawled node
            nodes = [n for n in self.observed_graph.neighbors(self._last_seed)
                     if n in self._observed_set] if self._last_seed is not None else []
        self.re_estimate_nodes(nodes)

        # Choose the node with maximal estimation - greedy variant
        seed = None
        score_max = -1
        for node in self.observed_set:
            score = self._node_score[node]
            if score > score_max:
                seed = node
                score_max = score

        return seed
