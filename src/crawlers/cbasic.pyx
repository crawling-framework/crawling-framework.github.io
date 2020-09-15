import logging
import random
import re

from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.deque cimport deque
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, predecrement as dec, address as addr

from base.cgraph cimport MyGraph, t_random
cimport cbasic  # pxd import DON'T DELETE

logger = logging.getLogger(__name__)


def all_subclasses(cls):
    """ Get all subclasses of a class
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def definition_to_filename(definition) -> str:
    """ Convert crawler string definition into filename. Uniqueness is maintained
    """
    _class, kwargs = definition
    excluded = set(['name', 'observed_graph', 'crawled_set', 'observed_set'])  # will not be in filename
    args = ";".join("%s=%s" % (key, definition_to_filename(kwargs[key]) if key == 'crawler_def' else kwargs[key])
                    for key in sorted(kwargs.keys()) if key not in excluded)
    res = "%s(%s)" % (_class.short, args)
    return res


# short class name -> class
short_to_class = {}

def eval_(string: str):
    """ Same as eval but returns string as is if fails to eval"""
    try:
        return eval(string)
    except NameError:
        return string


def filename_to_definition(filename: str):
    """ Convert filename into crawler string definition. Uniqueness is maintained """
    if len(short_to_class) == 0:
        import crawlers.multiseed  # NOTE: needed to include Crawlers defined in multiseed module, add here any other modules with crawlers
        import crawlers.advanced  # NOTE: needed to include Crawlers defined in multiseed module, add here any other modules with crawlers
        from running.metrics_and_runner import Metric  # NOTE: same for metrics
        # Build short names dict
        for sb in set().union(all_subclasses(Crawler), all_subclasses(Metric)):
            if hasattr(sb, 'short'):
                name = sb.short
                assert name not in short_to_class
                short_to_class[name] = sb
            else:
                short_to_class[sb.__name__] = sb

    # Recursive unpack
    _class_str, params = re.findall("([^\(\)]*)\((.*)\)", filename)[0]
    _class = short_to_class[_class_str]
    kwargs = {}
    if len(params) > 0:
        for eq in params.split(';'):
            key, value = eq.split('=', 1)
            kwargs[key] = filename_to_definition(value) if key == 'crawler_def' else eval_(value)
    return _class, kwargs


cdef class Crawler:
    """
    The root class for all our crawlers. Keeps fields:

    * `observed_graph` - observed sample of all crawled and visible nodes and edges,
    * `crawled_set` - set of all crawled nodes,
    * `observed_set` - set of osberved but not crawled yet nodes.

    Supports methods:
    -----------------

    * int `next_seed` (self) - choose next seed among the observed ones, each crawler defines its own strategy.
    * vector[int] `crawl` (self, int seed) - crawl the specified node and return a list (vector) of newly observed nodes.
    * int `crawl_budget` (self, int budget) - crawl `budget` number of nodes according to the strategy.

    Crawler is initialized with the original graph (which must NOT be modified) and optionally with observed_graph,
    crawled_set, and observed_set - to start with.

    Crawler has its `definition` which determines the instance. Definition can be uniquely transformed into string
    filename and back in order to store the results of measurements.

    """
    # short = 'Crawler'  # Should be specified in all subclasses to distinguish them in filenames
    def __init__(self, MyGraph graph, name: str=None,
                 observed_graph: MyGraph=None, crawled_set: set=None, observed_set: set=None,
                 **kwargs):
        """
        :param graph: original graph, must remain unchanged
        :param name: specify to use in pictures, by default name == filename generated from definition
        :param observed_graph: optionally use a given observed graph, NOTE: the object will be modified, make a copy if needed
        :param crawled_set: optionally use a given crawled set, NOTE: the object will be modified, make a copy if needed
        :param observed_set: optionally use a given observed set, NOTE: the object will be modified, make a copy if needed
        :param kwargs: all additional parameters (needed in subclasses) - they will be encoded into string definition
        """
        # Original graph
        self._orig_graph = graph  # FIXME copying here?

        # Observed graph
        self._observed_graph = observed_graph if observed_graph is not None else \
            MyGraph(directed=self._orig_graph.directed, weighted=self._orig_graph.weighted)

        # Crawled set and observed set
        self._crawled_set = crawled_set if crawled_set is not None else set()
        self._observed_set = observed_set if observed_set is not None else set()
        # assert self._observed_graph.nodes() == len(self._crawled_set) + len(self._observed_set)  # only for single not Multi

        self._definition = type(self), kwargs

        self.name = name if name else definition_to_filename(self._definition)

    @staticmethod
    def from_definition(MyGraph graph, definition) -> Crawler:
        """ Build a Crawler instance from its definition
        """
        _class, kwargs = definition
        return _class(graph, **kwargs)

    @property
    def definition(self):
        """ Get definition. Definition is the pair (class, constructor kwargs) """
        return self._definition

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        # return set(n for n in self._observed_graph.iter_nodes())
        return self._crawled_set.union(self._observed_set)

    @property
    def crawled_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return self._crawled_set

    @property
    def observed_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return self._observed_set

    @property
    def orig_graph(self):
        return self._orig_graph

    cpdef bint observe(self, int node):
        """ Add the node to observed set and observed graph. """
        cdef bint already = self._observed_graph.has_node(node)
        if not already:
            self._observed_graph.add_node(node)
            self._observed_set.add(node)
        return already

    # FIXME reference to vector faster?
    cpdef vector[int] crawl(self, int seed) except *:
        """ Crawl specified node. The observed graph is updated, also are crawled and observed set.

        :param seed: node id to crawl
        :return: vector (list) of newly seen nodes
        """
        cdef vector[int] res
        if seed in self._crawled_set:  # FIXME simplify!
            logger.error("Already crawled: %s" % seed)
            return res  # if already crawled - do nothing

        self._crawled_set.add(seed)

        if self._observed_graph.has_node(seed):  # remove from observed set
            assert seed in self._observed_set
            self._observed_set.remove(seed)
        else:  # add to observed graph
            self._observed_graph.add_node(seed)

        # Iterate over neighbours
        for n in self._orig_graph.neighbors(seed):
            if not self._observed_graph.has_node(n):  # add to observed graph and observed set
                self._observed_graph.add_node(n)
                self._observed_set.add(n)
                res.push_back(n)
            self._observed_graph.add_edge(seed, n)
        return res

    cpdef int next_seed(self) except -1:
        """
        Core of the crawler - the algorithm to choose the next seed to be crawled.
        Seed must be a node of the original graph.

        :return: node id as int
        """
        raise NotImplementedError()

    cpdef int crawl_budget(self, int budget) except -1:
        """
        Perform `budget` number of crawls according to the algorithm.

        :param budget: so many nodes will be crawled. If can't crawl any more, raise NoNextSeedError
        :param args: customizable additional args for subclasses
        :return:
        """
        for i in range(budget):
            self.crawl(self.next_seed())
        return 0  # for exception compatibility


class CrawlerException(Exception):
    pass


class NoNextSeedError(CrawlerException):
    """ Can't get next seed exception.
    Called when the observed set becomes empty.
    Could mean the following:

    * all nodes are crawled
    * cannot reach some nodes because they are disconnected
    * algorithm errors in forming the observed set
    """
    def __init__(self, error_msg=None):
        super().__init__(self)
        self.error_msg = error_msg if error_msg else "Can't get next seed: no more observed nodes."

    def __str__(self):
        return self.error_msg


cdef class CrawlerWithInitialSeed(Crawler):
    """
    Starting seed is specified in constructor or randomly chosen from original graph.
    """
    def __init__(self, MyGraph graph, initial_seed: int=-1, **kwargs):
        """
        :param initial_seed: if observed set is empty, the crawler will start from the given initial
         node. (If not empty, starting seed is defined from `next_seed()` call). If -1 is given
         (which is default), a random node of original graph will be used as initial one.
        """
        if initial_seed != -1:  # if not random, we reflect it in definition and filename
            kwargs['initial_seed'] = initial_seed

        super().__init__(graph, **kwargs)

        # If no observed nodes, pick a random seed from original graph or a specified initial seed
        if self._observed_graph.nodes() == 0:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)


cdef class RandomCrawler(CrawlerWithInitialSeed):
    """
    Crawls a random node from the observed ones.
    """
    short = 'RC'
    cdef vector[int] next_seeds

    def __init__(self, MyGraph graph, **kwargs):
        super().__init__(graph, **kwargs)

        cdef int n
        for n in self._observed_set:
            self.next_seeds.push_back(n)
        # TODO shuffle next_seeds

    cpdef int next_seed(self) except -1:
        if len(self._observed_set) == 0:
            raise NoNextSeedError()
        cdef int n = self.next_seeds.back()
        self.next_seeds.pop_back()
        return n

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = Crawler.crawl(self, seed)
        cdef int n, size, r
        size = self.next_seeds.size()
        # insert to next_seeds in random order
        for n in res:
            self.next_seeds.insert(self.next_seeds.begin() + t_random.GetUniDevInt(size+1), n)
            size += 1
        return res


cdef class RandomWalkCrawler(CrawlerWithInitialSeed):
    """
    Crawls a random neighbor of the previously crawled node.
    If cannot crawl, goes to a random neighbor and proceed there.
    """
    short = 'RW'
    cdef int prev_seed

    def __init__(self, MyGraph graph, initial_seed: int=-1, **kwargs):
        super().__init__(graph, initial_seed=initial_seed, **kwargs)
        self.prev_seed = -1

    cpdef int next_seed(self) except -1:
        if self.prev_seed == -1:  # first step
            self.prev_seed = next(iter(self._observed_set))
            return self.prev_seed

        if len(self._observed_set) == 0:
            raise NoNextSeedError()

        # for walking we need to step on already crawled nodes too
        if self._observed_graph.deg(self.prev_seed) == 0:
            raise NoNextSeedError("No neighbours to go next.")

        # Go to a neighbor until encounter not crawled node
        cdef int seed
        while True:
            seed = self._observed_graph.random_neighbor(self.prev_seed)
            self.prev_seed = seed
            if seed in self._observed_set:
                return seed


cdef class BreadthFirstSearchCrawler(CrawlerWithInitialSeed):
    """
    Crawls in a breadth first manner.
    """
    short = 'BFS'
    cdef queue[int] bfs_queue

    def __init__(self, MyGraph graph, initial_seed: int=-1, **kwargs):
        super().__init__(graph, initial_seed=initial_seed, **kwargs)

        cdef int n
        for n in self._observed_set:
            self.bfs_queue.push(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.bfs_queue.size() > 0:
            seed = self.bfs_queue.front()
            self.bfs_queue.pop()  # DFS - from back, BFS - from front
            if seed not in self._crawled_set:
                return seed

        assert len(self._observed_set) == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = Crawler.crawl(self, seed)
        for n in res:
            self.bfs_queue.push(n)
        return res


cdef class DepthFirstSearchCrawler(CrawlerWithInitialSeed):
    """
    Crawls in a depth first manner.
    """
    short = 'DFS'
    cdef deque[int] dfs_queue

    def __init__(self, MyGraph graph, initial_seed: int=-1, **kwargs):
        super().__init__(graph, initial_seed=initial_seed, **kwargs)

        cdef int n
        for n in self._observed_set:
            self.dfs_queue.push_back(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.dfs_queue.size() > 0:
            seed = self.dfs_queue.back()
            self.dfs_queue.pop_back()  # DFS - from back, BFS - from front
            if seed not in self._crawled_set:  # FIXME simplify?
                return seed

        assert len(self._observed_set) == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = Crawler.crawl(self, seed)
        for n in res:
            self.dfs_queue.push_back(n)
        return res


cdef class SnowBallCrawler(CrawlerWithInitialSeed):
    """
    Every step of BFS taking neighbors with probability p.
    http://www.soundarajan.org/papers/CrawlingAnalysis.pdf
    https://arxiv.org/pdf/1004.1729.pdf
    """
    short = 'SBS'
    cdef float p
    cdef deque[int] sbs_queue
    cdef deque[int] sbs_backlog

    def __init__(self, MyGraph graph, initial_seed: int=-1, p: float=0.5, **kwargs):
        """
        :param p: probability of taking neighbor into queue
        """
        if 'name' not in kwargs:
            kwargs['name'] = "%s%2.f" % (self.short, 100*p)
        super().__init__(graph, initial_seed=initial_seed, p=p, **kwargs)

        self.p = p
        cdef int n
        for n in self._observed_set:
            self.sbs_queue.push_back(n)

    cpdef int next_seed(self) except -1:
        cdef int seed
        while self.sbs_queue.size() > 0:
            seed = self.sbs_queue.front()
            self.sbs_queue.pop_front()  # like BFS - from front
            if seed not in self._crawled_set:  # FIXME simplify?
                return seed

        while self.sbs_backlog.size() > 0:
            seed = self.sbs_backlog.back()
            self.sbs_backlog.pop_back()  # like DFS - from back
            if seed not in self._crawled_set:  # FIXME simplify?
                return seed

        assert len(self._observed_set) == 0
        raise NoNextSeedError()

    cpdef vector[int] crawl(self, int seed):
        cdef vector[int] res = Crawler.crawl(self, seed)
        for n in res:
            if random.random() < self.p:
                self.sbs_queue.push_back(n)
            else:
                self.sbs_backlog.push_back(n)
        return res


cdef class MaximumObservedDegreeCrawler(CrawlerWithInitialSeed):
    """
    Crawls a node with maximal observed degree.
    """
    short = 'MOD'
    cdef int batch
    cdef ND_Set nd_set

    def __init__(self, MyGraph graph, initial_seed: int=-1, int batch=1, **kwargs):
        """
        :param batch: batch size
        """
        if 'name' not in kwargs:
            kwargs['name'] = "%s%s" %  (self.short, '' if batch == 1 else batch)
        super().__init__(graph, initial_seed=initial_seed, batch=batch, **kwargs)

        self.batch = batch
        self.mod_set = set()  # we need to create, pop and lookup nodes
        cdef int n, d
        self.nd_set = ND_Set()
        for n in self._observed_set:
            self.nd_set.add(n, self._observed_graph.deg(n))

    cpdef void update(self, vector[int] nodes):  # TODO passing ref may be faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n
        for n in nodes:
            # assert n in self._observed_set and n not in self.mod_queue  # Just for debugging
            if n in self.mod_set:  # already in batch
                continue
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update observed ND_Set
        """
        cdef vector[int] res = Crawler.crawl(self, seed)
        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                upd.push_back(n)
        self.update(upd)
        return res

    cpdef int next_seed(self) except -1:
        """ Next node with highest degree
        """
        cdef int n
        cdef vector[int] vec
        if len(self.mod_set) == 0:  # making array of top-k degrees
            if self.nd_set.empty():
                assert len(self._observed_set) == 0
                raise NoNextSeedError()
            vec = self.nd_set.pop_top(self.batch)
            for n in vec:
                self.mod_set.add(n)
        return self.mod_set.pop()


cdef class PreferentialObservedDegreeCrawler(CrawlerWithInitialSeed):
    """
    selects for crawling one of the observed nodes with probability proportional to the observed degree.
    """
    short = 'POD'
    cdef int batch
    cdef ND_Set nd_set

    def __init__(self, MyGraph graph, initial_seed: int=-1, int batch=1, **kwargs):
        """
        :param batch: batch size
        """
        if 'name' not in kwargs:
            kwargs['name'] = "%s%s" %  (self.short, '' if batch == 1 else batch)
        super().__init__(graph, initial_seed=initial_seed, batch=batch, **kwargs)

        self.batch = batch
        cdef int n, d
        self.pod_set = set()  # we need to create, pop and lookup nodes
        self.nd_set = ND_Set()
        for n in self._observed_set:
            self.nd_set.add(n, self._observed_graph.deg(n))

    cpdef void update(self, vector[int] nodes):  # TODO passing ref may be faster?
        """ Update priority structures with specified nodes (suggested their degrees have changed).
        """
        cdef int d, n
        for n in nodes:
            # assert n in self._observed_set and n not in self.mod_queue  # Just for debugging
            if n in self.pod_set:  # already in batch
                continue
            d = self._observed_graph.deg(n)
            # logger.debug("%s.ND_Set.updating(%s, %s)" % (self.name, n, d))
            self.nd_set.update_1(n, d - 1)

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update observed ND_Set
        """
        cdef vector[int] res = Crawler.crawl(self, seed)
        cdef vector[int] upd
        cdef int n
        for n in self._observed_graph.neighbors(seed):
            if n in self._observed_set:
                upd.push_back(n)
        self.update(upd)
        return res

    cpdef int next_seed(self) except -1:
        cdef int n
        if len(self.pod_set) == 0:  # when batch ends, we create another one
            if len(self._observed_set) == 0:
                raise NoNextSeedError()
            for _ in range(min(self.batch, len(self.nd_set))):
                n = self.nd_set.pop_proportional_degree()
                # self.pod_set.insert(n)
                self.pod_set.add(n)
        return self.pod_set.pop()


cdef class MaximumExcessDegreeCrawler(CrawlerWithInitialSeed):
    """ Benchmark Crawler - greedy selection of next node with maximal excess (real - observed) degree
    """
    short = 'MED'
    cdef ND_Set nd_set

    def __init__(self, MyGraph graph, initial_seed: int=-1, **kwargs):
        super().__init__(graph, initial_seed=initial_seed, **kwargs)

        cdef int n, d
        self.nd_set = ND_Set()
        for n in self._observed_set:
            self.nd_set.add(n, self._orig_graph.deg(n))

    cpdef vector[int] crawl(self, int seed):
        """ Crawl specified node and update newly observed in ND_Set
        """
        cdef vector[int] res = Crawler.crawl(self, seed)
        cdef int n
        for n in res:
            self.nd_set.add(n, self._orig_graph.deg(n))  # some elems will be re-written
        return res

    cpdef int next_seed(self) except -1:
        """ Next node with highest real (unknown) degree
        """
        cdef int n
        if self.nd_set.empty():
            assert len(self._observed_set) == 0
            raise NoNextSeedError()
        return self.nd_set.pop_top(1)[0]
