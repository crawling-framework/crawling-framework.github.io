import logging

from base.cbasic import NoNextSeedError, CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler
from cbasic cimport CCrawler, CCrawlerUpdatable
from cgraph cimport CGraph, str_to_chars, t_random
from node_deg_set cimport ND_Set  # FIXME try 'as ND_Set' if error 'ND_Set is not a type identifier'

from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec, address as addr

from statistics import get_top_centrality_nodes, Stat


cdef class CrawlerWithAnswer(CCrawler):
    """
    Crawler which makes a limited number of iterations and generates an answer as its result.
    """
    cdef int _limit
    cdef cset[int] _answer  # TODO make ref?
    cdef dict __dict__  # for pythonic fields, makes it slower

    def __init__(self, CGraph graph, int limit, name=None):
        super().__init__(graph, name=name)
        self._limit = limit
        # self._answer = None
        self._seeds_generator = self.seeds_generator()  # TODO make it cdef?

        name = name if name is not None else type(self).__name__
        self._name = str_to_chars(name)

    @property
    def answer(self):
        return self._answer

    cpdef int next_seed(self) except -1:
        try:
            return next(self._seeds_generator)
        except StopIteration:
            raise NoNextSeedError("Reached maximum number of iterations %d" % self._limit)

    cpdef int crawl_budget(self, int budget) except -1:
        try:
            CCrawler.crawl_budget(self, budget)
            # FIXME this is for intermediate result only
            self._compute_answer()
        except CrawlerException:
            # Reached maximum number of iterations or any other Crawler exception
            if self._answer.size() == 0:
                self._compute_answer()

    def seeds_generator(self):
        """ Creates generator of seeds according to the algorithm.
        """
        raise NotImplementedError()

    cpdef int _compute_answer(self) except -1:
        """ Compute result set of nodes.
        """
        raise NotImplementedError()

    cdef int _get_mod_nodes(self, cset[int]* from_set, cset[int]* to_set, int count=-1):
        """
        Get list of nodes with maximal observed degree from the specified set.

        :param from_set: subset of the observed graph nodes
        :param count: top-list size
        :return: list of nodes by decreasing of their observed degree
        """
        if count == -1:
            count = from_set.size()
        count = min(count, from_set.size())

        cdef cset[pair[int, int]] candidates
        cdef int n, i = 0
        for n in deref(from_set):
            candidates.insert(pair[int, int](self._observed_graph.deg(n), n))

        cdef cset[pair[int, int]].const_reverse_iterator it = candidates.const_rbegin()
        cdef pair[int, int] p
        for i in range(count):
            to_set.insert(deref(it).second)
            inc(it)
        return 0


cdef class AvrachenkovCrawler(CrawlerWithAnswer):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    cdef int n, n1, k
    cdef cset[int]* _top_observed_seeds  # nodes for 2nd step

    def __init__(self, CGraph graph, int n=1000, int n1=500, int k=100, name=None):
        super().__init__(graph, limit=n, name=name if name else 'Avrach_n=%s_n1=%s_k=%s' % (n, n1, k))
        assert n1 <= n <= self._orig_graph.nodes()
        #assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

        self._top_observed_seeds = new cset[int]()

    def seeds_generator(self):
        # 1) random seeds
        cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.n1)
        cdef int i, node
        for i in range(self.n1):
            yield random_seeds[i]

        # 2) detect MOD batch
        self._get_mod_nodes(self._observed_set, self._top_observed_seeds, self.n - self.n1)
        for node in deref(self._top_observed_seeds):
            yield node

    cpdef int _compute_answer(self) except -1:
        self._answer.clear()
        self._get_mod_nodes(self._top_observed_seeds, &self._answer, self.k)
        return 0


cdef class ThreeStageCrawler(CrawlerWithAnswer):
    """
    """
    cdef int s, n, pN
    cdef float p
    cdef cset[int]* e1
    cdef cset[int]* e1s
    cdef cset[int]* e2
    cdef cset[int]* e2s

    def __init__(self, CGraph graph, int s=500, int n=1000, float p=0.1, name=None):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph, limit=n, name=name if name else '3-Stage_s=%s_n=%s_p=%.3f' % (s, n, p))
        self.s = s
        self.n = n
        self.pN = int(p * self._orig_graph.nodes())
        # assert s <= n <= self.pN

        self.e1s = new cset[int]()  # E1*
        self.e2s = new cset[int]()  # E2*
        self.e1 = new cset[int]()  # E1
        self.e2 = new cset[int]()  # E2

    def seeds_generator(self):
        # 1) random seeds
        cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.s)
        cdef int i, node
        for i in range(self.s):
            yield random_seeds[i]

        # memorize E1
        self.e1 = new cset[int](deref(self._observed_set))  # FIXME copying and updating ref
        logging.debug("|E1|=%s" % self.e1.size())

        # Check that e1 size is more than (n-s)
        if self.n - self.s > self.e1.size():
            msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (self.e1.size(), self.n - self.s)
            logging.error(msg)
            raise CrawlerException(msg)

        # 2) detect MOD
        self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
        logging.debug("|E1*|=%s" % self.e1s.size())

        # NOTE: e1s is not sorted by degree
        for node in deref(self.e1s):
            yield node

    cpdef int _compute_answer(self) except -1:
        # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN

        # memorize E2
        self.e2 = new cset[int](deref(self._observed_set))  # FIXME copying and updating ref
        logging.debug("|E2|=%s" % self.e2.size())

        # Get v=(pN-n+s) max degree observed nodes
        self.e2s.clear()
        self._get_mod_nodes(self.e2, self.e2s, self.pN - self.n + self.s)
        logging.debug("|E2*|=%s" % self.e2s.size())

        # Final answer - E* = E1* + E2*
        self._answer.clear()
        self._answer.insert(self.e1s.begin(), self.e1s.end())
        self._answer.insert(self.e2s.begin(), self.e2s.end())
        logging.debug("|E*|=%s" % self._answer.size())
        return 0


cdef class ThreeStageCrawlerSeedsAreHubs(ThreeStageCrawler):
    """
    Artificial version of ThreeStageCrawler, where instead of initial random seeds we take hubs
    """
    cdef cset[int]* h   # S

    def __init__(self, CGraph graph, int s=500, int n=1000, float p=0.1, name=None):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        """
        super().__init__(graph, s=s, n=n, p=p, name=name if name else'3-StageHubs_s=%s_n=%s_p=%.3f' % (s, n, p))
        self.h = new cset[int]()

    def seeds_generator(self):
        # 1) hubs as seeds
        hubs = get_top_centrality_nodes(self._orig_graph, Stat.DEGREE_DISTR, count=self.s)
        for i in range(self.s):
            self.h.insert(hubs[i])
            yield hubs[i]

        # memorize E1
        self.e1 = new cset[int](deref(self._observed_set))  # FIXME copying and updating ref
        logging.debug("|E1|=%s" % self.e1.size())

        # Check that e1 size is more than (n-s)
        if self.n - self.s > self.e1.size():
            msg = "E1 too small: |E1|=%s < (n-s)=%s. Increase s or decrease n." % (self.e1.size(), self.n - self.s)
            logging.error(msg)
            raise CrawlerException(msg)

        # 2) detect MOD
        self._get_mod_nodes(self._observed_set, self.e1s, self.n - self.s)
        logging.debug("|E1*|=%s" % self.e1s.size())

        # NOTE: e1s is not sorted by degree
        for node in deref(self.e1s):
            yield node

    cpdef int _compute_answer(self) except -1:  # E* = S + E1* + E2*
        self.e2 = new cset[int](deref(self._observed_set))  # FIXME copying and updating ref
        logging.debug("|E2|=%s" % self.e2.size())

        # Get v=(pN-n+|self.h|) max degree observed nodes
        self.e2s.clear()
        self._get_mod_nodes(self.e2, self.e2s, self.pN - self.n + self.h.size())
        logging.debug("|E2*|=%s" % self.e2s.size())

        # Final answer - E* = S + E1* + E2*, |E*|=pN
        self._answer.clear()
        self._answer.insert(self.h.begin(), self.h.end())
        self._answer.insert(self.e1s.begin(), self.e1s.end())
        self._answer.insert(self.e2s.begin(), self.e2s.end())
        logging.debug("|E*|=%s" % self._answer.size())
        return 0


cdef class ThreeStageMODCrawler(CrawlerWithAnswer):
    """
    """
    cdef int s, n, pN, b
    cdef float p
    cdef cset[int]* e1
    cdef cset[int]* e1s
    cdef cset[int]* e2
    cdef cset[int]* e2s
    cdef CCrawler mod
    cdef bint mod_on

    def __init__(self, CGraph graph, int s=500, int n=1000, float p=0.1, int b=10, name=None):
        """
        :param graph: original graph
        :param s: number of initial random seed
        :param n: number of nodes to be crawled, must be >= seeds
        :param p: fraction of graph nodes to be returned
        :param b: batch size
        """
        assert 1 <= b <= n-s
        super().__init__(graph, limit=n, name=name if name else '3-StageMOD_s=%s_n=%s_p=%.3f_b=%s' % (s, n, p, b))
        self.s = s
        self.n = n
        self.pN = int(p * self._orig_graph.nodes())
        # assert s <= n <= self.pN
        self.b = b

        self.e1s = new cset[int]()  # E1*
        self.e2s = new cset[int]()  # E2*
        self.e1 = new cset[int]()  # E1
        self.e2 = new cset[int]()  # E2

        self.mod_on = False

    cpdef vector[int] crawl(self, int seed) except *:
        """ Apply MOD when time comes
        """
        # FIXME res is set now
        if not self.mod:
            return CCrawler.crawl(self, seed)

        cdef vector[int] res = self.mod.crawl(seed)
        self.e1s.insert(seed)
        return res  # FIXME copying

    def seeds_generator(self):
        # 1) random seeds
        cdef vector[int] random_seeds = self._orig_graph.random_nodes(self.s)
        cdef int i, node
        for i in range(self.s):
            yield random_seeds[i]

        # 2) run MOD
        # TODO should we cimport it to avoid pythonizing?
        self.mod = MaximumObservedDegreeCrawler(self._orig_graph, batch=self.b)  # FIXME initial seed will be randomly chosen - useless
        self.mod.set_observed_graph(self._observed_graph)
        self.mod.set_observed_set(self._observed_set)
        self.mod.set_crawled_set(self._crawled_set)
        self.mod_on = True

        for i in range(self.n-self.s):
            yield self.mod.next_seed()

    cpdef int _compute_answer(self) except -1:
        # 3) Find v=(pN-n+s) nodes by MOD from E2 -> E2*. Return E*=(E1* + E2*) of size pN

        # # E2
        # self.e2 = new cset[int](deref(self._observed_set))  # FIXME copying and updating ref
        # logging.debug("|E2|=%s" % self.e2.size())

        # Get v=(pN-n+s) max degree observed nodes
        self.e2s.clear()
        self._get_mod_nodes(self._observed_set, self.e2s, self.pN - self.n + self.s)
        logging.debug("|E2*|=%s" % self.e2s.size())

        # Final answer - E* = E1* + E2*
        self._answer.clear()
        self._answer.insert(self.e1s.begin(), self.e1s.end())
        self._answer.insert(self.e2s.begin(), self.e2s.end())
        logging.debug("|E*|=%s" % self._answer.size())
        # assert self._answer.size() <= self.pN
        return 0


# cdef class ThreeStageCustomCrawler(CrawlerWithAnswer):
#     raise NotImplementedError()


# time_top = 0
# time_clust = 0
from time import time
t = time()

cdef class DE_Crawler(CCrawlerUpdatable):
    """
    DE Crawler from http://kareekij.github.io/papers/de_crawler_asonam18.pdf
    DE-Crawler: A Densification-Expansion Algorithm for Online Data Collection
    """
    cdef int initial_budget, prev_seed, initial_seed
    cdef float s_d, s_e
    cdef float a1, a2, b1, b2
    cdef cmap[int, float] node_clust

    cdef public float time_top
    cdef public float time_clust
    cdef public float time_next

    # cdef class ND_Set_2(ND_Set):
    #     cdef int pop_from_bottom(self, int count):
    #         pass

    cdef ND_Set nd_set

    def __init__(self, CGraph graph, int initial_budget=10, int initial_seed=-1, name=None, **kwargs):
        super().__init__(graph, name=name if name else "DE", **kwargs)

        if self._observed_set.size() == 0:
            if initial_seed == -1:  # FIXME duplicate code in all basic crawlers?
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        self.s_d = 0
        self.s_e = 0
        self.a1 = 1  # varying
        self.a2 = 1  # 64
        self.b1 = 0.5
        self.b2 = 0.5

        self.initial_seed = initial_seed
        self.prev_seed = -1
        self.initial_budget = initial_budget

        self.nd_set = ND_Set()
        for n in deref(self._observed_set):
            self.nd_set.add(n, self._observed_graph.deg(n))
            self.node_clust[n] = self._observed_graph.clustering(n)

    cpdef void update(self, vector[int] nodes):  # FIXME maybe ref faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n, neigh, conn_neigs
        cdef float cc, cc_new
        for n in nodes:
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

            # Update clustering
            t = time()
            conn_neigs = 0
            for neigh in self._observed_graph.neighbors(n):
                if self._observed_graph.has_edge(self.prev_seed, neigh):
                    conn_neigs += 1
            self.node_clust[n] = (self.node_clust[n] * (d-1) * (d-1) / 2 + conn_neigs) / d / d * 2
            # self.node_clust[n] = self._observed_graph.clustering(n)
            self.time_clust += time()-t

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
        if self._crawled_set.size() >= self.initial_budget:
            self.s_d = self.a1 * (1. * d_new / d_ex if d_ex > 0 else 1) + self.b1 * self.s_d
            self.s_e = self.a2 * (1. * d_seen / d_ex if d_ex > 0 else 1) + self.b2 * self.s_e

        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            # if n in self._observed_set:
            if self._observed_set.find(n) != self._observed_set.end():
                upd.push_back(n)
        self.node_clust[seed] = self._observed_graph.clustering(seed)
        self.update(upd)

        return res

    cpdef int next_seed(self) except -1:
        cdef int n
        if self._crawled_set.size() < self.initial_budget:  # RW step
            # return RandomWalkCrawler.next_seed(self)
            if self.prev_seed == -1:  # first step
                self.prev_seed = self.initial_seed
                self.nd_set.remove(self.initial_seed, self._observed_graph.deg(self.initial_seed))
                return self.initial_seed

            if self._observed_set.size() == 0:
                raise NoNextSeedError()

            # for walking we need to step on already crawled nodes too
            if self._observed_graph.deg(self.prev_seed) == 0:
                raise NoNextSeedError("No neighbours to go next.")

            # Go to a neighbor until encounter not crawled node
            while True:
                n = self._observed_graph.random_neighbor(self.prev_seed)
                self.prev_seed = n
                if self._observed_set.find(n) != self._observed_set.end():
                    break

            self.nd_set.remove(n, self._observed_graph.deg(n))
            return n

        if self._crawled_set.size() == self.initial_budget:  # Define a1
            # self.a1 = self._observed_graph['MAX_DEGREE'] / self._observed_graph['AVG_DEGREE']
            self.a1 = 1. * self._observed_graph.max_deg() * self._observed_graph.nodes() / (1. if self._observed_graph.directed else 2.) / self._observed_graph.edges()
            logging.debug("a1=%s" % (self.a1))

        if self._observed_set.size() == 0:
            assert self.nd_set.empty()
            raise NoNextSeedError()

        cdef vector[int] part
        cdef int count, deg, i, max_score_ix
        cdef float score, max_score

        tn = time()
        # logging.debug("s_d=%s, s_e=%s" % (self.s_d, self.s_e))
        if self.s_d < self.s_e:  # Expansion. Random from the bottom 80% by observed degree
            logging.debug("Expansion")
            t = time()
            count = max(1, int(0.8 * self.nd_set.size()))
            part = self.nd_set.bottom(count)
            n = part[t_random.GetUniDevInt(count)]
            self.time_top += time()-t
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
        self.time_next += time()-tn
        return n


cpdef test_cadvanced():
    print("cadv")

    from graph_io import GraphCollections
    g = GraphCollections.get('petster-hamster')
    ac = AvrachenkovCrawler(g)
