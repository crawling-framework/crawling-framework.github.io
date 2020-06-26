import logging

from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec, address as addr

from cbasic import NoNextSeedError, CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler
from cbasic cimport CCrawler, CCrawlerUpdatable
from base.cgraph cimport CGraph, str_to_chars, t_random
from base.node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'

from statistics import get_top_centrality_nodes, Stat


cdef class CrawlerWithAnswer(CCrawler):
    """
    Crawler which makes a limited number of iterations and generates an answer as its result.
    """
    cdef int _limit
    cdef readonly set _answer
    cdef public bint _actual_answer  # indicates whether to recompute answer. public is needed for python access

    def __init__(self, CGraph graph, int limit=-1, **kwargs):
        super().__init__(graph, **kwargs)
        self._limit = limit
        self._answer = set()
        self._actual_answer = True
        self._seeds_generator = self.seeds_generator()  # TODO make it cdef?

    @property
    def answer(self) -> set:
        if not self._actual_answer:
            self._compute_answer()
            self._actual_answer = True
        return self._answer

    cpdef int next_seed(self) except -1:
        try:
            return next(self._seeds_generator)
        except StopIteration:
            raise NoNextSeedError("Reached maximum number of iterations %d" % self._limit)

    cpdef vector[int] crawl(self, int seed) except *:
        """ 
        NOTE: don't forget to set _actual_answer=False when overwrite crawl(seed) in subclasses
        """
        self._actual_answer = False
        return CCrawler.crawl(self, seed)

    cpdef int crawl_budget(self, int budget) except -1:
        try:
            CCrawler.crawl_budget(self, budget)
        except CrawlerException:
            # Reached maximum number of iterations or any other Crawler exception
            if not self._actual_answer:
                self._compute_answer()
                self._actual_answer = True

    def seeds_generator(self):
        """ Creates generator of seeds according to the algorithm.
        """
        raise NotImplementedError()

    cpdef int _compute_answer(self) except -1:
        """ Compute result set of nodes.
        """
        raise NotImplementedError()

    cpdef void _get_mod_nodes(self, from_set, to_set, int count=-1):
        """
        Get list of nodes with maximal observed degree from the specified set.

        :param from_set: set to pick from
        :param to_set: set to add to
        :param count: top-list size
        """
        if count == -1:
            count = len(from_set)
        count = min(count, len(from_set))

        cdef cset[pair[int, int]] candidates
        cdef int n, i = 0
        for n in from_set:
            candidates.insert(pair[int, int](self._observed_graph.deg(n), n))

        cdef cset[pair[int, int]].const_reverse_iterator it = candidates.const_rbegin()
        cdef pair[int, int] p
        for i in range(count):
            to_set.add(deref(it).second)
            inc(it)


# cdef class AvrachenkovCrawler(CrawlerWithAnswer):
#     """
#     Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
#     https://arxiv.org/pdf/1410.0571.pdf
#     """
#     short = '2-Stage'
#     cdef int n, n1, k
#     cdef set _top_observed_seeds  # nodes for 2nd step
#
#     def __init__(self, CGraph graph, int n=1000, int n1=500, int k=100, **kwargs):
#         super().__init__(graph, limit=n, n=n, n1=n1, k=k, **kwargs)
#         assert n1 <= n <= self._orig_graph.nodes()
#         #assert k <= n-n1
#         self.n1 = n1
#         self.n = n
#         self.k = k
#
#         self._top_observed_seeds = set()
#
#     def seeds_generator(self):
#         # 1) random seeds
#         cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.n1)
#         cdef int i, node
#         for i in range(self.n1):
#             yield random_seeds[i]
#
#         # 2) detect MOD batch
#         self._get_mod_nodes(self._observed_set, self._top_observed_seeds, self.n - self.n1)
#         for node in self._top_observed_seeds:
#             yield node
#
#     cpdef int _compute_answer(self) except -1:
#         self._answer.clear()
#         self._get_mod_nodes(self._top_observed_seeds, self._answer, self.k)
#         return 0
#
#
# cdef class ThreeStageCrawler(CrawlerWithAnswer):
#     """
#     """
#     short = '3-Stage'
#     cdef int s, n, pN
#     cdef float p
#     cdef set e1
#     cdef set e1s
#     cdef set e2
#     cdef set e2s
#
#     def __init__(self, CGraph graph, int s=500, int n=1000, p: float=0.1, **kwargs):
#         """
#         :param graph: original graph
#         :param s: number of initial random seed
#         :param n: number of nodes to be crawled, must be >= seeds
#         :param p: fraction of graph nodes to be returned
#         """
#         super().__init__(graph, limit=n, s=s, n=n, p=p, **kwargs)
#         self.s = s
#         self.n = n
#         self.pN = int(p * self._orig_graph.nodes())
#         # assert s <= n <= self.pN
#
#         self.e1s = set()  # E1*
#         self.e2s = set()  # E2*
#         self.e1 = set()  # E1
#         self.e2 = set()  # E2
#
#     def seeds_generator(self):
#         # 1) random seeds
#         cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.s)
#         cdef int i, node
#         for i in range(self.s):
#             yield random_seeds[i]
#
#         # memorize E1
#         self.e1 = self._observed_set.copy()
#         logging.debug("|E1|=%s" % len(self.e1))
#
#         # Check that e1 size is more than (n-s)
#         if self.n - self.s > len(self.e1):
#             msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (len(self.e1), self.n - self.s)
#             logging.error(msg)
#             raise CrawlerException(msg)
#
#         # 2) detect MOD
#         self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
#         logging.debug("|E1*|=%s" % len(self.e1s))
#
#         # NOTE: e1s is not sorted by degree
#         for node in self.e1s:
#             yield node
#
#     cpdef int _compute_answer(self) except -1:
#         # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN
#
#         # memorize E2
#         self.e2 = self._observed_set.copy()
#         logging.debug("|E2|=%s" % len(self.e2))
#
#         # Get v=(pN-n+s) max degree observed nodes
#         self.e2s.clear()
#         self._get_mod_nodes(self.e2, self.e2s, self.pN - self.n + self.s)
#         logging.debug("|E2*|=%s" % len(self.e2s))
#
#         # Final answer - E* = E1* + E2*
#         self._answer = self.e1s.union(self.e2s)
#         logging.debug("|E*|=%s" % len(self._answer))
#         return 0
#
#
# cdef class ThreeStageCrawlerSeedsAreHubs(ThreeStageCrawler):
#     """
#     Artificial version of ThreeStageCrawler, where instead of initial random seeds we take hubs
#     """
#     short = '3-StageHubs'
#     cdef set h   # S
#
#     def __init__(self, CGraph graph, int s=500, int n=1000, p: float=0.1, **kwargs):
#         """
#         :param graph: original graph
#         :param s: number of initial random seed
#         :param n: number of nodes to be crawled, must be >= seeds
#         :param p: fraction of graph nodes to be returned
#         """
#         super().__init__(graph, limit=n, s=s, n=n, p=p, **kwargs)
#         self.h = set()
#
#     def seeds_generator(self):
#         # 1) hubs as seeds
#         hubs = get_top_centrality_nodes(self._orig_graph, Stat.DEGREE_DISTR, count=self.s)
#         for i in range(self.s):
#             self.h.add(hubs[i])
#             yield hubs[i]
#
#         # memorize E1
#         self.e1 = self._observed_set.copy()  # FIXME copying and updating ref
#         logging.debug("|E1|=%s" % len(self.e1))
#
#         # Check that e1 size is more than (n-s)
#         if self.n - self.s > len(self.e1):
#             msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (len(self.e1), self.n - self.s)
#             logging.error(msg)
#             raise CrawlerException(msg)
#
#         # 2) detect MOD
#         self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
#         logging.debug("|E1*|=%s" % len(self.e1s))
#
#         # NOTE: e1s is not sorted by degree
#         for node in self.e1s:
#             yield node
#
#     cpdef int _compute_answer(self) except -1:  # E* = S + E1* + E2*
#         self.e2 = self._observed_set.copy()
#         logging.debug("|E2|=%s" % len(self.e2))
#
#         # Get v=(pN-n+|self.h|) max degree observed nodes
#         self.e2s.clear()
#         self._get_mod_nodes(self.e2, self.e2s, self.pN - self.n + len(self.h))
#         logging.debug("|E2*|=%s" % len(self.e2s))
#
#         # Final answer - E* = S + E1* + E2*, |E*|=pN
#         self._answer = self.h.union([self.e1s, self.e2s])
#         logging.debug("|E*|=%s" % len(self._answer))
#         return 0
#
#
# cdef class ThreeStageMODCrawler(CrawlerWithAnswer):
#     """
#     """
#     short = '3-StageMOD'
#     cdef int s, n, pN, b
#     cdef float p
#     cdef set e1
#     cdef set e1s
#     cdef set e2
#     cdef set e2s
#     cdef CCrawler mod
#     cdef bint mod_on
#
#     def __init__(self, CGraph graph, int s=500, int n=1000, p: float=0.1, int b=10, **kwargs):
#         """
#         :param graph: original graph
#         :param s: number of initial random seed
#         :param n: number of nodes to be crawled, must be >= seeds
#         :param p: fraction of graph nodes to be returned
#         :param b: batch size
#         """
#         assert 1 <= b <= n-s
#         super().__init__(graph, limit=n, s=s, n=n, p=p, b=b, **kwargs)
#         self.s = s
#         self.n = n
#         self.pN = int(p * self._orig_graph.nodes())
#         # assert s <= n <= self.pN
#         self.b = b
#
#         self.e1s = set()  # E1*
#         self.e2s = set()  # E2*
#         self.e1 = set()  # E1
#         self.e2 = set()  # E2
#
#         self.mod_on = False
#
#     cpdef vector[int] crawl(self, int seed) except *:
#         """ Apply MOD when time comes
#         """
#         # FIXME res is set now
#         if not self.mod_on:
#             return CrawlerWithAnswer.crawl(self, seed)
#
#         self._actual_answer = False
#         cdef vector[int] res = self.mod.crawl(seed)
#         self.e1s.add(seed)
#         return res  # FIXME copying
#
#     def seeds_generator(self):
#         # 1) random seeds
#         cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.s)
#         cdef int i, node
#         for i in range(self.s):
#             yield random_seeds[i]
#
#         # 2) run MOD
#         # TODO should we cimport it to avoid pythonizing?
#         self.mod = MaximumObservedDegreeCrawler(
#             self._orig_graph, batch=self.b, observed_graph=self._observed_graph,
#             crawled_set=self._crawled_set, observed_set=self._observed_set)
#         self.mod_on = True
#
#         for i in range(self.n - self.s):
#             yield self.mod.next_seed()
#
#     cpdef int _compute_answer(self) except -1:
#         # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN
#
#         if len(self.e1s) < self.pN:
#             self.e2s.clear()
#             # Get v=(pN-n+s) max degree observed nodes
#             self._get_mod_nodes(self._observed_set, self.e2s, self.pN - len(self.e1s))
#             logging.debug("|E2*|=%s" % len(self.e2s))
#
#             # Final answer - E* = E1* + E2*
#             self._answer = self.e1s.union(self.e2s)
#         else:
#             # Top-pN from e1s
#             self._get_mod_nodes(self._crawled_set, self._answer, self.pN)
#
#         logging.debug("|E*|=%s" % len(self._answer))
#         # assert len(self._answer) <= self.pN
#         return 0


# cdef class ThreeStageCustomCrawler(CrawlerWithAnswer):
#     raise NotImplementedError()


# time_top = 0
# time_clust = 0
# from time import time
# t = time()

cdef class DE_Crawler(CCrawlerUpdatable):
    """
    DE Crawler from http://kareekij.github.io/papers/de_crawler_asonam18.pdf
    DE-Crawler: A Densification-Expansion Algorithm for Online Data Collection
    """
    # NOTE: cython version is 25-30% faster than same python version (why?)
    short= 'DE'
    cdef int initial_budget, prev_seed, initial_seed
    cdef float s_d, s_e
    cdef float a1, a2, b1, b2
    cdef cmap[int, float] node_clust
    cdef ND_Set nd_set

    def __init__(self, CGraph graph, int initial_seed=-1, int initial_budget=-1, **kwargs):
        if initial_budget == -1:
            initial_budget = int(0.015 * graph.nodes())  # Default in paper: 15% of total budget (which is 10%)
        else:
            kwargs['initial_budget'] = initial_budget
        self.initial_budget = initial_budget

        if initial_seed != -1:
            kwargs['initial_seed'] = initial_seed

        super().__init__(graph, **kwargs)

        if len(self._observed_set) == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)
        self.initial_seed = initial_seed
        self.prev_seed = -1

        self.s_d = 0
        self.s_e = 0
        self.a1 = 1  # varying
        self.a2 = 1  # 64
        self.b1 = 0.5
        self.b2 = 0.5

        self.nd_set = ND_Set()
        for n in self._observed_set:
            self.nd_set.add(n, self._observed_graph.deg(n))
            self.node_clust[n] = self._observed_graph.clustering(n)

    cpdef void update(self, vector[int] nodes):  # FIXME maybe ref faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n, neigh, conn_neigs
        cdef float cc, cc_new
        cdef cmap[int, float].iterator it
        for n in nodes:
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

            # Update clustering
            # t = time()
            conn_neigs = 0
            for neigh in self._observed_graph.neighbors(n):
                if self._observed_graph.has_edge(self.prev_seed, neigh):
                    conn_neigs += 1

            # NOTE: if n not in self.node_clust, self.node_clust[n] returns 0.0
            self.node_clust[n] = (self.node_clust[n] * (d-1) * (d-1) / 2 + conn_neigs) / d / d * 2
            # self.node_clust[n] = self._observed_graph.clustering(n)
            # self.time_clust += time()-t

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update observed ND_Set
        """
        cdef int d_seen, d_ex, d_new
        d_seen = self._observed_graph.deg(seed)

        cdef vector[int] res = CCrawler.crawl(self, seed)
        self.prev_seed = seed

        d_ex = self._observed_graph.deg(seed) - d_seen
        d_new = len(res)
        # logging.debug("seed %s: d_seen %s, d_ex %s, d_new %s" % (seed, d_seen, d_ex, d_new))

        # Update stats
        if len(self._crawled_set) >= self.initial_budget:
            self.s_d = self.a1 * (1. * d_new / d_ex if d_ex > 0 else 1) + self.b1 * self.s_d
            self.s_e = self.a2 * (1. * d_seen / d_ex if d_ex > 0 else 1) + self.b2 * self.s_e

        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                upd.push_back(n)
        self.node_clust[seed] = self._observed_graph.clustering(seed)
        self.update(upd)

        return res

    cpdef int next_seed(self) except -1:
        cdef int n
        if len(self._crawled_set) < self.initial_budget:  # RW step
            # return RandomWalkCrawler.next_seed(self)
            if self.prev_seed == -1:  # first step
                self.prev_seed = self.initial_seed
                self.nd_set.remove(self.initial_seed, self._observed_graph.deg(self.initial_seed))
                return self.initial_seed

            if len(self._observed_set) == 0:
                raise NoNextSeedError()

            # for walking we need to step on already crawled nodes too
            if self._observed_graph.deg(self.prev_seed) == 0:
                raise NoNextSeedError("No neighbours to go next.")

            # Go to a neighbor until encounter not crawled node
            while True:
                n = self._observed_graph.random_neighbor(self.prev_seed)
                self.prev_seed = n
                if n in self._observed_set:
                    break

            self.nd_set.remove(n, self._observed_graph.deg(n))
            return n

        if len(self._crawled_set) == self.initial_budget:  # Define a1
            # self.a1 = self._observed_graph['MAX_DEGREE'] / self._observed_graph['AVG_DEGREE']
            self.a1 = 1. * self._observed_graph.max_deg() * self._observed_graph.nodes() / (1. if self._observed_graph.directed else 2.) / self._observed_graph.edges()
            logging.debug("a1=%s" % (self.a1))

        if len(self._observed_set) == 0:
            assert self.nd_set.empty()
            raise NoNextSeedError()

        cdef vector[int] part
        cdef int count, deg, i, max_score_ix
        cdef float score, max_score

        # tn = time()
        # logging.debug("s_d=%s, s_e=%s" % (self.s_d, self.s_e))
        if self.s_d < self.s_e:  # Expansion. Random from the bottom 80% by observed degree
            logging.debug("Expansion")
            # t = time()
            count = max(1, int(0.8 * self.nd_set.size()))
            part = self.nd_set.bottom(count)
            n = part[t_random.GetUniDevInt(count)]
            # self.time_top += time()-t
            # deg = self._observed_graph.deg(n)
            # self.nd_set.remove(n, deg)
            # return n

        else:  # Densification.
            logging.debug("Densification")
            count = max(1, int(0.2 * self.nd_set.size()))
            part = self.nd_set.top(count)

            # Find argmax d*(1-CC)
            max_score = max_score_ix = 0
            for ix in range(count):
                n = part[ix]
                score = self._observed_graph.deg(n) * (1 - self.node_clust[n])
                # score = self._observed_graph.deg(n) * (1 - self._observed_graph.clustering(n))
                if score > max_score:
                    max_score = score
                    max_score_ix = ix
            n = part[max_score_ix]

        deg = self._observed_graph.deg(n)
        self.nd_set.remove(n, deg)
        # self.time_next += time()-tn
        return n


cpdef test_cadvanced():
    print("cadv")

    # from graph_io import GraphCollections
    # g = GraphCollections.get('petster-hamster')
    # ac = AvrachenkovCrawler(g)

    cdef cmap[int, float] m
    print(m[1])
