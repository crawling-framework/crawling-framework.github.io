from graph_io import MyGraph

cdef class Crawler(object):

    cdef set observed_set

    def __init__(self, graph: MyGraph, name=None, **kwargs):
        # original graph
        self.orig_graph = graph

        # observed graph
        if 'observed_graph' in kwargs:
            self.observed_graph = kwargs['observed_graph']
        else:
            self.observed_graph = MyGraph.new_snap(directed=graph.directed, weighted=graph.weighted)

        # crawled ids set
        if 'crawled_set' in kwargs:
            self.crawled_set = kwargs['crawled_set']
        else:
            self.crawled_set = set()

        # observed ids set excluding crawled ones
        if 'observed_set' in kwargs:
            self.observed_set = kwargs['observed_set']
        else:
            self.observed_set = set()

        self.seed_sequence_ = []  # D: sequence of tries to add nodes to draw history and debug
        self.name = name if name is not None else type(self).__name__

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int) -> bool:
        """
        Crawl specified node. The observed graph is updated, also crawled and observed set.
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            logging.debug("Already crawled: %s" % seed)
            return False  # if already crawled - do nothing

        self.seed_sequence_.append(seed)
        self.crawled_set.add(seed)
        g = self.observed_graph.snap
        if g.IsNode(seed):  # remove from observed set
            self.observed_set.remove(seed)
        else:  # add to observed graph
            g.AddNode(seed)

        # iterate over neighbours
        for n in self.orig_graph.neighbors(seed):
            if not g.IsNode(n):  # add to observed graph and observed set
                g.AddNode(n)
                self.observed_set.add(n)
            g.AddEdge(seed, n)
        return True

    def next_seed(self) -> int:
        """
        Core of the crawler - the algorithm to choose the next seed to be crawled.
        Seed must be a node of the original graph.

        :return: node id as int
        """
        raise CrawlerException("Not implemented")

    def crawl_budget(self, budget: int, *args):
        """
        Perform `budget` number of crawls according to the algorithm.
        Note that `next_seed()` may be called more times - some returned seeds may not be crawled.

        :param budget: so many nodes will be crawled. If can't crawl any more, raise CrawlerError
        :param args: customizable additional args for subclasses
        :return:
        """
        for _ in range(budget):
            while not self.crawl(self.next_seed()):
                continue

