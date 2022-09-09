import logging
import numpy as np
from tqdm import tqdm

from base.cgraph import MyGraph
from crawlers.cadvanced import NodeFeaturesUpdatableCrawlerHelper
from crawlers.cbasic import InitialSeedCrawlerHelper, NoNextSeedError
from crawlers.declarable import Declarable
from search.feature_extractors import AttrHelper
from search.oracles import Oracle
from search.predictor_based_crawlers.predictor_based import PredictorBasedCrawler


class PredictorTrainStrategy(Declarable):
    """ Prepares training data for predictor during crawling.
    Variants: online (history of neighborhoods of crawled nodes),
    boost (resample crawled set to increase features variablility).
    """
    short = "TrainStrategy"

    def __init__(self,
                 train_from_size=10,
                 retrain_step_exponent=1.15,
                 train_max_samples=300, name=None, **kwargs):
        """
        :param train_from_size: number of nodes crawled when 1st training iteration will start.
         Default 10.
        :param retrain_step_exponent: if last training was at step s, the next training will occur
         at step s*exponent. Default 1.15.
        :param train_max_samples: maximal len(Xs) for training predictor. Default 300.
        :param kwargs:
        """
        super(PredictorTrainStrategy, self).__init__(
            train_from_size=train_from_size, retrain_step_exponent=retrain_step_exponent,
            train_max_samples=train_max_samples, **kwargs)

        self.name = f"{type(self).__name__}-{train_from_size}*{retrain_step_exponent}^" \
                    f"{train_max_samples}" if name is None else name
        self.train_from_size = train_from_size
        self.retrain_step_exponent = retrain_step_exponent
        self.train_max_samples = train_max_samples

        self._last_trained = 0

    def __str__(self):
        return self.name

    def __call__(self, predictor_based_crawler: PredictorBasedCrawler):
        """ """
        raise NotImplementedError("Must be defined in subclass")


class OnlineTrainStrategy(PredictorTrainStrategy):
    """
    """
    short = "Online"

    def __init__(self, name=None, **kwargs):
        super(OnlineTrainStrategy, self).__init__(name=name, **kwargs)

        self._seen_Xs = []  # sequence of seen Xs
        self._seen_ys = []

    def __call__(self, crawler: PredictorBasedCrawler):
        # Append info for the last crawled
        seed = crawler._last_seed
        if seed is None:
            return
        X = crawler.predictor.extract_features(seed, crawler)
        y = 1 if crawler.oracle(seed, crawler.orig_graph) == 1 else 0
        self._seen_Xs.append(X)
        self._seen_ys.append(y)

        # Check if it's time to retrain
        count = len(self._seen_Xs)
        if count >= self.train_from_size and\
                (self._last_trained == 0 or count >= self._last_trained * self.retrain_step_exponent):
            ys = self._seen_ys[-self.train_max_samples:]
            while len(np.unique(ys)) < 2:
                size = int(len(ys) * 1.5)
                if size > count:
                    logging.error(
                        f"{self.name} cannot train: all encountered data up to "
                        f"{count} has the same class, while at least 2 classes are needed for "
                        f"training. Omit training")
                    # Try to increase omit_first or train_last parameter."
                    return
                ys = self._seen_ys[-size:]

            xs = self._seen_Xs[-len(ys):]
            crawler.train_predictor(xs, ys)
            # logging.info('fit %s at %s step' % (self.name, count))
            self._last_trained = count


class SubsetReCrawler(InitialSeedCrawlerHelper):
    """ Picks a next node randomly from a specified subset of nodes.
    """
    short = "_Subset"

    def __init__(self, graph: MyGraph, subset: set, targets: set=None):
        """
        :param graph: full graph
        :param subset: subset of nodes to crawl, must be connected
        :param targets: target nodes, must be a subset of the `subset`
        """
        initial_seed = int(np.random.choice(list(subset)))
        super().__init__(graph, initial_seed=initial_seed)
        self.subset = subset
        self.targets = targets

        self.candidates = [initial_seed]

    def next_seed(self):
        if len(self.candidates) == 0:
            raise NoNextSeedError()
        return self.candidates.pop()

    def crawl(self, seed: int):
        res = super().crawl(seed)

        # Add newly seen nodes from subset to candidates
        for n in res:
            if n in self.subset:
                self.candidates.append(n)

        np.random.shuffle(self.candidates)
        return res


class SubsetPicker:
    """ Picks a next node randomly from a specified subset of nodes.
    """
    short = "_Subset"

    def __init__(self, graph: MyGraph, oracle: Oracle, subset: set, targets: set=None, untargets: set=None):
        """
        :param graph: (observed) graph to crawl
        :param oracle: oracle
        :param subset: subset of nodes to crawl, must be connected
        :param targets: target nodes, must be a subset of the `subset`
        :param untargets: untargets nodes
        """
        self.graph = graph
        self.oracle = oracle
        self.subset = subset
        self.targets = targets
        self.untargets = untargets

    def subsequence(self, fraction=0.2, targets_ratio=0.5):
        """
        Generate random crawling sequence and return its subsequence satisfying to specified
        constraints.

        :param fraction: part of the crawl sequence to be used for training, preferred at the end
        :param targets_ratio: wished ratio of target/untarget nodes in result subsequence
        :return: sequence, indices
        """
        sequence = []
        # Generate crawling sequence
        src = SubsetReCrawler(self.graph, self.subset)
        for _ in self.subset:
            n = src.next_seed()
            src.crawl(n)
            sequence.append(n)

        # count = max(min_count, round(fraction * len(sequence)))
        count = fraction * len(sequence)
        # t_num = round(targets_ratio * count)
        t_num = targets_ratio * count
        # Randomized rounding - gives a good average at early steps when called many times
        t_num = int(t_num) if np.random.random() > t_num-int(t_num) else int(t_num) + 1
        u_num = count - t_num
        u_num = int(u_num) if np.random.random() > u_num-int(u_num) else int(u_num) + 1
        t = 0
        u = 0
        indices = []
        # NOTE: down to 1 not 0 - to avoid target node being the first
        for ix in range(len(sequence) - 1, 0, -1):
            if ix < 0: break
            node = sequence[ix]
            if node in self.targets and t < t_num:
                t += 1
                indices.append(ix)
            elif node in self.untargets and u < u_num:
                u += 1
                indices.append(ix)
            if t >= t_num and u >= u_num:
                break
        return sequence, reversed(indices)


class ObservedGraphCrawlerWithNeighborhood(NodeFeaturesUpdatableCrawlerHelper, InitialSeedCrawlerHelper):
    def __init__(self, observed_graph, oracle_function, **kwargs):
        super(ObservedGraphCrawlerWithNeighborhood, self).__init__(
            observed_graph, oracle=oracle_function, depth=2, rounding=True, **kwargs)


class BoostingTrainStrategy(PredictorTrainStrategy):
    """
    """
    short = "Boost"

    def __init__(self, max_boost_iterations=20, last_boost_steps_fraction=0.2, name=None, **kwargs):
        """
        :param max_boost_iterations: maximal number of boosting iterations
        :param last_boost_steps_fraction: fraction of steps taken from the end of resampling run
        """
        kw = {}
        if last_boost_steps_fraction != 0.2:
            kw['last_boost_steps_fraction'] = last_boost_steps_fraction
        super(BoostingTrainStrategy, self).__init__(
            max_boost_iterations=max_boost_iterations, **kw,
            # last_boost_steps_fraction=last_boost_steps_fraction,
            name=name, **kwargs)

        self.max_boost_iterations = max_boost_iterations
        self.last_boost_steps_fraction = last_boost_steps_fraction

    def __call__(self, crawler: PredictorBasedCrawler):
        count = len(crawler.crawled_set)
        if count >= self.train_from_size and\
                (self._last_trained == 0 or count >= self._last_trained * self.retrain_step_exponent):
            self._boost(crawler)
            self._last_trained = count

    def _boost_iteration(self, crawler, picker, targets, Xs, ys, pbar):
        sequence, indices = picker.subsequence(
            fraction=self.last_boost_steps_fraction, targets_ratio=0.5)
        indices = set(indices)

        # Emulates oracle with original graph, without passing the graph in arguments
        class anOracle(Oracle):
            def __call__(self, id, _):
                return 1 if crawler.oracle(node, crawler.orig_graph) == 1 else 0

        AttrHelper.share_graph_attributes(crawler.orig_graph, crawler._observed_graph)
        booster_crawler = ObservedGraphCrawlerWithNeighborhood(
            crawler._observed_graph, anOracle(), initial_seed=sequence[0],
            attributes=crawler.predictor.used_attributes)

        for ix, node in enumerate(sequence):
            if ix in indices:
                X = crawler.predictor.extract_features(node, booster_crawler)
                y = 1 if node in targets else 0
                Xs.append(X)
                ys.append(y)
                pbar.update(1)
                if len(Xs) >= self.train_max_samples:  # enough training examples
                    break
            booster_crawler.crawl(node)

    def _boost(self, crawler):
        """ Boost current sample and train predictor.
        """
        # Get target nodes
        targets = set()
        untargets = set()
        crawled_set = crawler.crawled_set
        for n in crawled_set:
            y = crawler.oracle(n, crawler._orig_graph)
            if y == 0:
                untargets.add(n)
            elif y == 1:
                targets.add(n)
            # if undefined class - do not train

        if len(targets) == 0 or len(untargets) == 0:
            logging.warning(f"{crawler.name} will not boost since sample has only 1 class, "
                            f"while 2 classes are needed for training. Also omit training.")
            return

        # Collect a dataset
        Xs = []
        ys = []
        picker = SubsetPicker(crawler.observed_graph, crawler.oracle,
                              crawled_set, targets, untargets)

        total = min(self.train_max_samples, int(
            self.max_boost_iterations * len(crawled_set) * self.last_boost_steps_fraction))
        pbar = tqdm(position=0, leave=True,
                    total=min(self.train_max_samples, int(
                        self.max_boost_iterations*len(crawled_set)*self.last_boost_steps_fraction)),
                    desc=f"Boosting sample at step {len(crawled_set)}")
        # boost_iterations = min(
        #     self.max_boost_iterations, max(
        #         1, self.train_max_samples // int(len(crawled_set) * self.last_boost_steps_fraction)))
        for _ in range(self.max_boost_iterations):
            self._boost_iteration(crawler, picker, targets, Xs, ys, pbar)
            # pbar.update(len(Xs)-pbar.n)
            if len(Xs) >= self.train_max_samples:  # enough training examples
                break
        pbar.close()
        # print("Boosted", len(Xs), "of", total)

        # Train predictor
        if all(y == 1 for y in ys):
            # Model is good
            logging.warning(f"All {len(ys)} samples have class 1, omit training")
        elif all(y == 0 for y in ys):
            # Model could be stuck!
            logging.warning(f"All {len(ys)} samples have class 0, omit training")
        else:
            crawler.train_predictor(Xs, ys)
