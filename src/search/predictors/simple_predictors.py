import abc
import logging
from abc import abstractmethod
import numpy as np

from crawlers.cadvanced import NodeFeaturesUpdatableCrawlerHelper
from crawlers.declarable import Declarable
from search.feature_extractors import NeighborsFeatureExtractor


def import_by_name(name: str, packs: list = None) -> type:
    """ Import name from packages, return class

    :param name: class name, full or relative
    :param packs: list of packages to search in
    :return: <class>
    """
    from pydoc import locate
    if packs is None:
        return locate(name)
    else:
        for pack in packs:
            klass = locate(f"{pack}.{name}")
            if klass is not None:
                return klass

    raise ImportError(f"Unknown sklearn model '{name}', couldn't import.")


class Predictor(Declarable):
    """ Parent class for node property prediction based on graph neighborhood.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self.name = "Predictor" if name is None else name

    def __str__(self):
        return self.name

    @property
    def used_attributes(self):
        return []

    @abstractmethod
    def reset(self):
        """ Reset the model parameters. Makes the model untrained. """
        pass

    @abstractmethod
    def extract_features(self, node, crawler_helper: NodeFeaturesUpdatableCrawlerHelper):
        """ Extract feature vector for a node using NodeFeaturesUpdatableCrawlerHelper.
        """
        pass

    @abstractmethod
    def train(self, Xs, ys):
        """ Train the model on data samples (Xs, ys). """
        pass

    @abstractmethod
    def predict_score(self, X):
        """ Compute score for feature vector X. It is preferred to be in interval [0; 1] and could
        be considered as the probability of target class.
        """
        pass


class SklearnPredictor(Predictor):
    """ Predictor based on a sci-kit-learn model.
    """

    def __init__(self, model: str, feature_extractor: NeighborsFeatureExtractor,
                 max_train_samples=None, name=None, **model_kwargs):
        """
        :param model: name of sklearn model, e.g. 'GradientBoostingClassifier'
        :param max_train_samples: upper limit on training data (actual for XGB, RF)
        :param model_kwargs: keyword args for the model
        :param feature_extractor: callable (neighborhood, index, graph, oracle) -> (Xs, ys)
        """
        kw = {}
        if max_train_samples is not None:
            kw['max_train_samples'] = max_train_samples

        super(SklearnPredictor, self).__init__(
            model=model, feature_extractor=feature_extractor, **kw,
            **model_kwargs)
        # imports from sklearn
        klass = import_by_name(model, None if model.startswith("sklearn") else [
            "sklearn.ensemble", "sklearn.linear_model", "sklearn.neighbors", "sklearn.cluster",
            "sklearn.svm"])
        self.model = klass(**model_kwargs)
        self.extract_features = feature_extractor
        self.max_train_samples = max_train_samples
        self.name = "%s[%s]" % (type(self.model).__name__, feature_extractor.name) \
            if name is None else name
        self._trained = False

    @property
    def used_attributes(self):
        return self.extract_features.attributes

    def reset(self):
        self._trained = False

    def train(self, Xs, ys):
        Xs = Xs[:self.max_train_samples]
        ys = ys[:self.max_train_samples]
        logging.info(f"Training {self.name} {np.bincount(np.array(ys))}")
        try:
            self.model.fit(Xs, ys)
        except ValueError as e:
            # Debugging ValueError
            if 'y contains 1 class after sample_weight' in e.args[0]:
                print("ValueError", e.args[0])
                print("model", str(self.model))
                print("ys", ys)
            else:
                raise e

        self._trained = True

    def predict_score(self, X):
        if not self._trained:  # not trained
            # logging.warning(f"{self.name} is used without training")
            return np.random.uniform(0, 1)
        try:
            p = self.model.predict_proba([X])[0][1]
            return p
        except IndexError as e:
            # Debugging IndexError
            print("IndexError", e)
            try:
                p = self.model.predict_proba([X])[0][1]
            except: pass
            print("model", str(self.model))
            print("X", X)
            return np.random.uniform(0, 1)
        except ValueError as e:
            # Debugging ValueError
            if 'Expected n_neighbors <= n_samples' in e.args[0]:
                print("KNN train samples is not enough yet")
                # KNN train samples is not enough yet
                self._trained = False
                return np.random.uniform(0, 1)
            raise e


class MaximumTargetNeighborsPredictor(Predictor):
    """ Predictor which encourages nodes with maximum number of target neighbors.
    """
    def __init__(self, name=None, **kwargs):
        """
        """
        super(MaximumTargetNeighborsPredictor, self).__init__(**kwargs)
        self.name = "MTN" if name is None else name
        self.depth = 1

        self._node_target = {}

    def extract_features(self, node, crawler_helper: NodeFeaturesUpdatableCrawlerHelper):
        X = 0  # target neighbors
        for n in crawler_helper.observed_graph.neighbors(node):
            if n not in self._node_target:
                self._node_target[n] = 1 if crawler_helper.oracle(n, crawler_helper._orig_graph) == 1 else 0
            X += self._node_target[n]
        return X

    def predict_score(self, X):
        """ Assume X = number target neighbors """
        p = 1 - 1/(X+1)  # probability of being target, in [0;1)
        return p
