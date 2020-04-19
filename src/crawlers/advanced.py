import logging
from math import ceil
from operator import itemgetter

import numpy as np

from crawlers.basic import Crawler, CrawlerError
from graph_io import MyGraph


class CrawlerWithAnswer(Crawler):
    """
    Crawler which makes a limited number of iterations and generates an answer as its result.
    """
    def __init__(self, graph: MyGraph, limit: int):
        super().__init__(graph)
        self.limit = limit
        self.answer = set()
        self.seeds_generator = self._seeds_generator()

    def next_seed(self) -> int:
        try:
            return next(self.seeds_generator)
        except StopIteration:
            raise CrawlerError("Reached maximum number of iterations %d" % self.limit)

    def crawl_budget(self, budget: int, *args):
        try:
            super().crawl_budget(budget, *args)
        except CrawlerError:
            # Reached maximum number of iterations or any other Crawler exception
            self._compute_answer()

    def _seeds_generator(self):
        """ Creates generator of seeds according to the algorithm.
        """
        raise NotImplementedError()

    def _compute_answer(self):
        """ Compute result set of nodes.
        """
        raise NotImplementedError()


class AvrachenkovCrawler(CrawlerWithAnswer):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph, limit=n)
        assert n1 <= n <= self.orig_graph.snap.GetNodes()
        assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

        self.top_observed_seeds = []  # nodes for 2nd step

    def _get_mod_nodes(self, nodes_set, count) -> list:
        """
        Get list of nodes with maximal observed degree from the specified set.

        :param nodes_set: subset of the observed graph nodes
        :param count: top-list size
        :return: list of nodes by decreasing of their observed degree
        """
        candidates_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in nodes_set]
        top_candidates_deg = sorted(candidates_deg, key=itemgetter(1), reverse=True)[:count]
        return [n for n, _ in top_candidates_deg]

    def _seeds_generator(self):
        # 1) random seeds
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        random_seeds = [int(n) for n in np.random.choice(graph_nodes, self.n1, replace=True)]
        for i in range(self.n1):
            yield random_seeds[i]

        # 2) detect MOD batch
        self.top_observed_seeds = self._get_mod_nodes(self.observed_set, self.n - self.n1)
        for node in self.top_observed_seeds:
            yield node

    def _compute_answer(self):
        self.answer = self._get_mod_nodes(self.top_observed_seeds, self.k)


class TwoStageCrawler(CrawlerWithAnswer):
    """
    """
    def __init__(self, graph: MyGraph, s=500, n=1000, p=0.1):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph, limit=n)
        self.s = s
        self.n = n
        self.pN = int(p*self.orig_graph.snap.GetNodes())
        assert s <= n <= self.pN

        self.random_seeds = []  # S
        self.e1s = set()  # E1*
        self.top_observed_seeds = []  # E1* as list
        self.e2s = set()  # E2*
        self.e1 = set()  # E1
        self.e2 = set()  # E2

    def _get_mod_nodes(self, nodes_set, count) -> list:
        """
        fixme duplicate code
        Get list of nodes with maximal observed degree from the specified set.

        :param nodes_set: subset of the observed graph nodes
        :param count: top-list size
        :return: list of nodes by decreasing of their observed degree
        """
        candidates_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in nodes_set]
        top_candidates_deg = sorted(candidates_deg, key=itemgetter(1), reverse=True)[:count]
        return [n for n, _ in top_candidates_deg]

    def _seeds_generator(self):
        # 1) random seeds
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        random_seeds = [int(n) for n in np.random.choice(graph_nodes, self.s, replace=True)]
        for i in range(self.s):
            yield random_seeds[i]

        # memorize E1
        self.e1 = set(self.observed_set)
        logging.debug("|E1|=", len(self.e1))

        # Check that e1 size is more than (n-s)
        if self.n - self.s > len(self.e1):
            raise CrawlerError("E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." %
                               (len(self.e1), self.n - self.s))

        # 2) detect MOD batch
        self.top_observed_seeds = self._get_mod_nodes(self.observed_set, self.n - self.s)
        self.e1s = set(self.top_observed_seeds)
        logging.debug("|E1*|=", len(self.e1s))

        for node in self.top_observed_seeds:
            yield node

    def _compute_answer(self):
        # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN

        # memorize E2
        self.e2 = set(self.observed_set)
        logging.debug("|E2|=", len(self.e2))

        # Get v=(pN-n+s) max degree observed nodes
        self.e2s = set(self._get_mod_nodes(self.e2, self.pN - self.n + self.s))
        logging.debug("|E2*|=", len(self.e2s))

        # Final answer - E* = E1* + E2*
        self.answer = self.e1s.union(self.e2s)
        logging.debug("|E*|=", len(self.answer))


class TwoStageCrawlerBatches(TwoStageCrawler):
    """
    """
    def __init__(self, graph: MyGraph, s=500, n=1000, p=0.1, b=10):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        :param b: batch size
        """
        assert 1 <= b <= n-s
        super().__init__(graph=graph, s=s, n=n, p=p)
        self.b = b

        self.batches_count = ceil((self.n-self.s) / self.b)
        self.e1_batches = [set()] * self.batches_count
        self.e1s_batches = [set()] * self.batches_count

    def _seeds_generator(self):
        """ Creates generator of seeds according to algorithm
        """
        # 1) random seeds
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        random_seeds = [int(n) for n in np.random.choice(graph_nodes, self.s, replace=False)]
        for i in range(self.s):
            yield random_seeds[i]

        # 2) batches of MOD. Batches of equal size except, probably, the last one
        batch_sizes = [self.b] * (self.batches_count-1) + \
                      [self.n-self.s - self.b*(self.batches_count-1)]
        for batch in range(self.batches_count):
            # Prepare batch
            self.e1_batches[batch] = set(self.observed_set)
            logging.debug("|E1[%d]|=%d" % (batch, len(self.e1_batches[batch])))

            obs_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in self.observed_set]
            top_obs_deg = sorted(obs_deg, key=itemgetter(1), reverse=True)[:batch_sizes[batch]]

            self.e1s_batches[batch] = set([n for n, _ in top_obs_deg])
            logging.debug("|E1*[%d]|=%d" % (batch, len(self.e1s_batches[batch])))

            for seed in self.e1s_batches[batch]:
                yield seed

    def _compute_answer(self):
        # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(all E1*[i] + E2*) of size pN

        # memorize E2
        self.e2 = set(self.observed_set)
        logging.debug("|E2|=", len(self.e2))

        # Get v=(pN-n+s) max degree observed nodes
        self.e2s = set(self._get_mod_nodes(self.e2, self.pN - self.n + self.s))
        logging.debug("|E2*|=", len(self.e2s))

        # Final answer - E* = all E1*[i] + E2*
        self.answer = self.e2s
        for e1_batch in self.e1_batches:
            self.answer = self.answer.union(e1_batch)
        logging.debug("|E*|=", len(self.answer))


def test_generator():
    def next_seed():
        for i in range(10):
            yield i

        for i in range(100, 120):
            yield i

    p = next_seed()
    for _ in range(20):
        s = next(p)
        print(s)


if __name__ == '__main__':
    test_generator()