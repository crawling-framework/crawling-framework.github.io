import logging
from operator import itemgetter

import numpy as np

from crawlers.basic import Crawler, CrawlerError
from graph_io import MyGraph


class AvrachenkovCrawler(Crawler):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph)
        assert n1 <= n <= self.orig_graph.snap.GetNodes()
        assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

        self.counter = 0  # may differ from crawled set size
        self.random_seeds = []  # will be initialized at 1st step
        self.top_observed_seeds = []  # nodes for 2nd step
        self.hubs_detected = {}

    def next_seed(self):
        ctr = self.counter
        if ctr == 0:  # 1st phase: pick random seed
            graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
            self.random_seeds = [int(n) for n in np.random.choice(graph_nodes, self.n1, replace=True)]

        if ctr < self.n1:  # 1st phase: crawl random seed
            self.counter += 1
            return self.random_seeds[ctr]

        if ctr == self.n1:  # 2nd phase: detect MOD batch
            obs_deg = [(o_id, self.observed_graph.snap.GetNI(o_id).GetDeg()) for o_id in self.observed_set]
            top_obs_deg = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n - self.n1]
            self.top_observed_seeds = [n for n, _ in top_obs_deg]

        if ctr < self.n:  # 2nd phase: crawl MOD batch
            self.counter += 1
            return self.top_observed_seeds[ctr-self.n1]

        raise CrawlerError("Reached maximum number of iterations %d" % self.n)

    def _get_candidates(self):
        candidates_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in
                          self.top_observed_seeds]
        top_candidates_deg = sorted(candidates_deg, key=itemgetter(1), reverse=True)[:self.k]
        self.hubs_detected = set([n for n, _ in top_candidates_deg])
        return self.hubs_detected

    def crawl_budget(self, budget: int, *args):
        try:
            super().crawl_budget(budget, *args)
        except: pass

        # Finalize
        if self.counter >= self.n:
            self._get_candidates()


class TwoStageCrawler(Crawler):
    """
    """
    def __init__(self, graph: MyGraph, s=500, n=1000, p=0.1):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph)
        self.s = s
        self.n = n
        self.pN = int(p*self.orig_graph.snap.GetNodes())
        assert s <= n <= self.pN

        self.counter = 0  # may differ from crawled set size
        self.random_seeds = []  # S
        self.e1s = set()  # E1*
        self.top_observed_seeds = []  # E1* as list
        self.hubs_detected = set()  # E2*
        self.e1 = set()  # E1
        self.e2 = set()  # E2
        self.es = set()  # E*

    def next_seed(self) -> int:
        ctr = self.counter

        # 1) Crawl s random seeds (same as Avrachenkov), obtain E1 - first neighbourhood.
        if ctr == 0:
            # pick random seeds
            graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
            self.random_seeds = [int(n) for n in np.random.choice(graph_nodes, self.s, replace=False)]

        if ctr < self.s:  # 1st phase: crawl random seed
            self.counter += 1
            return self.random_seeds[ctr]

        # 2) Detect (n-s) nodes by MOD from E1 -> E1*. Crawl E1* and obtain E2 - second neighbourhood.
        if ctr == self.s:
            # memorize E1
            self.e1 = set(self.observed_set)
            logging.debug("|E1|=", len(self.e1))

            # Check that e1 size is more than (n-s)
            if self.n - self.s > len(self.e1):
                raise CrawlerError("E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." %
                                   (len(self.e1), self.n - self.s))

            # Get (n-s) max degree observed nodes
            obs_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in self.e1]
            top_obs_deg = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n - self.s]
            self.top_observed_seeds = [n for n, _ in top_obs_deg]
            self.e1s = set(self.top_observed_seeds)
            logging.debug("|E1*|=", len(self.e1s))

        if ctr < self.n:  # 2nd phase: crawl MOD batch
            self.counter += 1
            return self.top_observed_seeds[ctr-self.s]

        raise CrawlerError("Reached maximum number of iterations %d" % self.n)

    def _get_candidates(self):
        # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN

        # memorize E2
        self.e2 = set(self.observed_set)
        logging.debug("|E2|=", len(self.e2))

        # Get v=(pN-n+s) max degree observed nodes
        candidates_deg = [(n, self.observed_graph.snap.GetNI(n).GetDeg()) for n in self.e2]
        top_candidates_deg = sorted(candidates_deg, key=itemgetter(1), reverse=True)[:self.pN - self.n + self.s]
        self.hubs_detected = set([n for n, _ in top_candidates_deg])
        logging.debug("|E2*|=", len(self.hubs_detected))

        # Final answer - E* = E1* + E2*
        self.es = self.e1s.union(self.hubs_detected)
        logging.debug("|E*|=", len(self.es))
        return self.es

    def crawl_budget(self, budget: int, *args):
        try:
            super().crawl_budget(budget, *args)
        except Exception as e:
            logging.exception(e)

        # Finalize
        if self.counter >= self.n:
            self._get_candidates()