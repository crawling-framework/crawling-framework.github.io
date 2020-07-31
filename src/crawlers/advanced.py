import logging

from statistics import get_top_centrality_nodes, Stat

from base.cgraph import MyGraph, seed_random, get_UniDevInt
from crawlers.cbasic import Crawler, CrawlerException, \
    MaximumObservedDegreeCrawler, NoNextSeedError, RandomWalkCrawler, DepthFirstSearchCrawler, RandomCrawler
from crawlers.cadvanced import CrawlerWithAnswer


class AvrachenkovCrawler(CrawlerWithAnswer):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    short = '2-Stage'

    def __init__(self, graph: MyGraph, n: int=1000, n1: int=500, k: int=100, **kwargs):
        super().__init__(graph, limit=n, n=n, n1=n1, k=k, **kwargs)
        assert n1 <= n
        # assert n1 <= n <= self._orig_graph.nodes()
        #assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

        self._top_observed_seeds = set()

    def seeds_generator(self):
        # 1) random seeds
        random_seeds = self._orig_graph.random_nodes(self.n1)
        for i in range(self.n1):
            yield random_seeds[i]

        # 2) detect MOD batch
        self._get_mod_nodes(self._observed_set, self._top_observed_seeds, self.n - self.n1)
        for node in self._top_observed_seeds:
            yield node

    def _compute_answer(self):
        self._answer.clear()
        self._get_mod_nodes(self._top_observed_seeds, self._answer, self.k)
        return 0


class EmulatorWithAnswerCrawler(CrawlerWithAnswer):
    short = 'EmulatorWA'

    def __init__(self, graph: MyGraph, crawler_def, n: int, target_size: int, **kwargs):
        """
        :param crawler_def: crawler to emulate
        :param n: limit of crawls (budget)
        :param target_size: max size of answer
        """
        super().__init__(graph, n=n, target_size=target_size, crawler_def=crawler_def, **kwargs)
        self.n = n
        self.target_size = target_size

        _, ckwargs = crawler_def
        ckwargs['observed_graph'] = self._observed_graph
        ckwargs['crawled_set'] = self._crawled_set
        ckwargs['observed_set'] = self._observed_set

        self.crawler = Crawler.from_definition(self._orig_graph, crawler_def)

    def crawl(self, seed: int):
        self._actual_answer = False
        return self.crawler.crawl(seed)

    def seeds_generator(self):
        for i in range(self.n):
            yield self.crawler.next_seed()

    def _compute_answer(self):
        self._answer.clear()
        self._get_mod_nodes(self._crawled_set, self._answer, self.target_size)
        return 0


class ThreeStageCrawler(CrawlerWithAnswer):
    """
    """
    short = '3-Stage'

    def __init__(self, graph: MyGraph, s: int=500, n: int=1000, p: float=0.1, **kwargs):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph, limit=n, s=s, n=n, p=p, **kwargs)
        self.s = s
        self.n = n
        self.p = p
        self.pN = int(self.p * self._orig_graph.nodes())
        # assert s <= n <= self.pN

        self.h = set()  # Hubs from start seeds
        self.e1s = set()  # E1*
        self.e2s = set()  # E2*
        self.e1 = set()  # E1
        self.e2 = set()  # E2

    def seeds_generator(self):
        # 1) random seeds
        random_seeds = self._orig_graph.random_nodes(self.s)
        for i in range(self.s):
            yield random_seeds[i]

        # Get hubs from start seeds
        self._get_mod_nodes(self._crawled_set, self.h, int(self.p * self.s))
        assert len(self.h) <= (self.p * self.s)

        # memorize E1
        self.e1 = self._observed_set.copy()
        logging.debug("|E1|=%s" % len(self.e1))

        # Check that e1 size is more than (n-s)
        if self.n - self.s > len(self.e1):
            msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (len(self.e1), self.n - self.s)
            logging.error(msg)
            raise CrawlerException(msg)

        # 2) detect MOD
        self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
        logging.debug("|E1*|=%s" % len(self.e1s))

        # NOTE: e1s is not sorted by degree
        for node in self.e1s:
            yield node

    def _compute_answer(self):
        self._answer.clear()
        if (len(self.e1s) + len(self.h)) < self.pN:
            self.e2s.clear()
            # Get v=(pN-n+s) max degree observed nodes
            self._get_mod_nodes(self._observed_set, self.e2s, self.pN - len(self.e1s) - len(self.h))
            logging.debug("|E2*|=%s" % len(self.e2s))

            # Final answer - E* = E1* + E2*
            self._answer.update(self.h, self.e1s, self.e2s)
        else:
            # Top-pN from all crawled
            self._get_mod_nodes(self._crawled_set, self._answer, self.pN)

        logging.debug("|E*|=%s" % len(self._answer))
        # assert len(self._answer) <= self.pN
        return 0


class ThreeStageCrawlerSeedsAreHubs(ThreeStageCrawler):
    """
    Artificial version of ThreeStageCrawler, where instead of initial random seeds we take hubs
    """
    short = '3-StageHubs'

    def __init__(self, graph: MyGraph, s: int=500, n: int=1000, p: float=0.1, **kwargs):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph, s=s, n=n, p=p, **kwargs)
        self.h = set()  # S

    def seeds_generator(self):
        # 1) hubs as seeds
        hubs = get_top_centrality_nodes(self._orig_graph, Stat.DEGREE_DISTR, count=self.s)
        for i in range(self.s):
            self.h.add(hubs[i])
            yield hubs[i]

        # memorize E1
        self.e1 = self._observed_set.copy()  # FIXME copying and updating ref
        logging.debug("|E1|=%s" % len(self.e1))

        # Check that e1 size is more than (n-s)
        if self.n - self.s > len(self.e1):
            msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (len(self.e1), self.n - self.s)
            logging.error(msg)
            raise CrawlerException(msg)

        # 2) detect MOD
        self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
        logging.debug("|E1*|=%s" % len(self.e1s))

        # NOTE: e1s is not sorted by degree
        for node in self.e1s:
            yield node

    def _compute_answer(self):  # E* = S + E1* + E2*
        self.e2 = self._observed_set.copy()
        logging.debug("|E2|=%s" % len(self.e2))

        # Get v=(pN-n+|self.h|) max degree observed nodes
        self.e2s.clear()
        self._get_mod_nodes(self.e2, self.e2s, self.pN - self.n + len(self.h))
        logging.debug("|E2*|=%s" % len(self.e2s))

        # Final answer - E* = S + E1* + E2*, |E*|=pN
        self._answer.clear()
        self._answer.update(self.h, self.e1s, self.e2s)
        logging.debug("|E*|=%s" % len(self._answer))
        return 0


class ThreeStageMODCrawler(CrawlerWithAnswer):
    """
    """
    short = '3-StageMOD'

    def __init__(self, graph: MyGraph, s: int=500, n: int=1000, p: float=0.1, b: int=10, **kwargs):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        :param b: batch size
        """
        # assert 1 <= b <= n-s
        super().__init__(graph, limit=n, s=s, n=n, p=p, b=b, **kwargs)
        self.s = s
        self.n = n
        self.p = p
        self.pN = int(self.p * self._orig_graph.nodes())
        # assert s <= n <= self.pN
        self.b = b

        self.h = set()  # Hubs from start seeds
        self.e1s = set()  # E1*
        self.e2s = set()  # E2*
        self.e1 = set()  # E1
        self.e2 = set()  # E2

        self.mod_on = False

    def crawl(self, seed: int):
        """ Apply MOD when time comes
        """
        self._actual_answer = False
        if not self.mod_on:
            return super().crawl(seed)

        res = self.mod.crawl(seed)
        self.e1s.add(seed)
        return res  # FIXME copying

    def seeds_generator(self):
        # 1) random seeds
        random_seeds = self._orig_graph.random_nodes(self.s)
        for i in range(self.s):
            yield random_seeds[i]

        # Get hubs from start seeds
        self._get_mod_nodes(self._crawled_set, self.h, int(self.p * self.s))
        assert len(self.h) <= (self.p * self.s)

        # 2) run MOD
        # TODO should we cimport it to avoid pythonizing?
        self.mod = MaximumObservedDegreeCrawler(
            self._orig_graph, batch=self.b, observed_graph=self._observed_graph,
            crawled_set=self._crawled_set, observed_set=self._observed_set)
        self.mod_on = True

        for i in range(self.n - self.s):
            yield self.mod.next_seed()

    def _compute_answer(self):
        self._answer.clear()
        if (len(self.e1s) + len(self.h)) < self.pN:
            self.e2s.clear()
            # Get v=(pN-n+s) max degree observed nodes
            self._get_mod_nodes(self._observed_set, self.e2s, self.pN - len(self.e1s) - len(self.h))
            logging.debug("|E2*|=%s" % len(self.e2s))

            # Final answer - E* = E1* + E2*
            self._answer.update(self.h, self.e1s, self.e2s)
        else:
            # Top-pN from crawled
            self._get_mod_nodes(self._crawled_set, self._answer, self.pN)

        logging.debug("|E*|=%s" % len(self._answer))
        # assert len(self._answer) <= self.pN
        return 0


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
