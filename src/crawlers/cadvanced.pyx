import numpy as np
from libcpp.vector cimport vector

from cbasic cimport Crawler
from base.cgraph cimport MyGraph

from search.feature_extractors import AttrHelper


cdef class NodeFeaturesUpdatableCrawlerHelper(Crawler):
    """
    Crawler helper interface.
    Calculates CC and CNF and keeps them up to date at each crawl() call.
    """
    def __init__(self, MyGraph graph, oracle, cc=True, cnf=True, tnf=True,
                 attributes:list=None, **kwargs):
        """

        :param graph:
        :param oracle: target node detector: node -> 1/0/None
        :param cc: if True, compute clustering coefficient
        :param cnf: if True, compute crawled neighbors fraction
        :param tnf: if True, compute target neighbors fraction (among crawled)
        :param attributes: list of attributes, for each value of each attribute compute fraction of
         neighbors with such value (among crawled)
        :param kwargs:
        """
        super(NodeFeaturesUpdatableCrawlerHelper, self).__init__(graph, oracle=oracle, **kwargs)
        self.oracle = oracle
        self.attributes = attributes if attributes is not None else []
        attrs_values = []
        # for a in self.attributes:
        #     for v in AttrHelper.attribute_vals(graph, a):
        #         attrs_values.append((a, v))
        self._do_cc = cc
        self._do_cnf = cnf
        self._do_tnf = tnf
        self._do_attributes = attributes is not None
        self.node_clust = {}
        self.node_cnf = {}
        self.node_tnf = {}
        self.node_crawled_deg = {}
        self.attr_node_vec = {a: {} for a in self.attributes}  # {attr -> {node -> fraction vector}}

        for n in self._observed_graph.iter_nodes():
            self.node_clust[n] = self._observed_graph.clustering(n)
            self.node_cnf[n] = 0
            # crawled_deg
            crawled_deg = sum([1 for _ in self._observed_graph.neighbors(n) if _ in self._crawled_set])
            self.node_crawled_deg[n] = crawled_deg
            # TNF
            n_tnf = 0
            for nn in self._observed_graph.neighbors(n):
                if nn in self._crawled_set and oracle(nn, graph) == 1:
                    n_tnf += 1
            self.node_tnf[n] = 0 if n_tnf == 0 else n_tnf / crawled_deg
            # Attributes
            for a in self.attributes:
                self.attr_node_vec[a][n] = np.zeros(len(AttrHelper.attribute_vals(graph, a)) + 1)
                for nn in self._observed_graph.neighbors(n):
                    if nn in self._crawled_set:
                        self.attr_node_vec[a][n] += AttrHelper.node_one_hot(graph, nn, a, add_none=True)
                if crawled_deg > 0:
                    self.attr_node_vec[a][n] /= crawled_deg

    cpdef vector[int] crawl(self, int seed) except *:
        cdef MyGraph g = self._observed_graph
        cdef MyGraph orig = self._orig_graph

        # if self._do_cc or self._do_cnf or self._do_tnf:
        # Remember seed neighbors before crawling - their degrees remain unchanged
        old_neighbors = set(g.neighbors(seed))

        # Crawl the node
        cdef vector[int] res = super(NodeFeaturesUpdatableCrawlerHelper, self).crawl(seed)
        cdef int n, d, new_triads = 0, seed_cnf = 0

        if self._do_cc:  # Update CC for seed and its neighbors
            cc = self.node_clust
            cc[seed] = g.clustering(seed)

            new_neighbors = set(res)  # newly discovered nodes -- CC remains unchanged
            other_neighbors = set()  # other (observed) neighbors
            for n in g.neighbors(seed):
                if n in new_neighbors:
                    cc[n] = 0
                    continue
                if n not in old_neighbors:  # other neighbors
                    other_neighbors.add(n)
                    new_triads = 0
                    for o in old_neighbors:
                        if g.has_edge(o, n):
                            new_triads += 1
                    d = g.deg(n)
                    if d < 2:
                        cc[n] = 0
                    else:
                        cc[n] = (1.0 * cc[n] * (d - 1) * (d - 2) + 2 * new_triads) / d / (d - 1)
                    # if (cc[n] - self._observed_graph.clustering(n)) ** 2 > 0.000001:
                    #     print('CC error !!!', n, cc[n], self._observed_graph.clustering(n), (cc[n] - self._observed_graph.clustering(n)) ** 2)

            for n in old_neighbors:
                new_triads = 0
                for o in other_neighbors:
                    if g.has_edge(o, n):
                        new_triads += 1
                d = g.deg(n)
                if d < 2:
                    cc[n] = 0
                else:  # d is unchanged
                    cc[n] += 2.0 * new_triads / d / (d - 1)
                # if (cc[n] - self._observed_graph.clustering(n)) ** 2 > 0.000001:
                #     print('CC error !!!', n, cc[n], self._observed_graph.clustering(n), (cc[n] - self._observed_graph.clustering(n)) ** 2)

        if self._do_cnf:  # Update CNF for seed and its neighbors
            cnf = self.node_cnf
            for n in g.neighbors(seed):
                if n in self._crawled_set:
                    seed_cnf += 1
                d = g.deg(n)
                if d == 1:  # newly observed
                    cnf[n] = 1.0
                else:
                    d_old = d if n in old_neighbors else d - 1
                    cnf[n] = (1.0 * cnf[n] * d_old + 1) / d
            cnf[seed] = 1.0 * seed_cnf / g.deg(seed)

        if self._do_tnf or self._do_attributes:  # update node_crawled_deg
            cd = self.node_crawled_deg
            for n in res:
                cd[n] = 0
            for n in g.neighbors(seed):  # including res
                cd[n] += 1

        if self._do_tnf:  # Update TNF for seed's old and new neighbors
            t = 1 if self.oracle(seed, self._orig_graph) == 1 else 0
            tnf = self.node_tnf
            # For all new (observed) neighbors - equal to (seed is target)
            for n in res:
                tnf[n] = 0
            # For all neighbors of seed - update.
            for n in g.neighbors(seed):  # including res
                x = tnf[n]
                tnf[n] += (t-x) / cd[n]

        if self._do_attributes:  # Update attributes fractions for seed's old and new neighbors
            for a in self.attributes:
                t = AttrHelper.node_one_hot(orig, seed, a, add_none=True)
                anv = self.attr_node_vec[a]
                # For all new (observed) neighbors - equal to 1-hot vector for seed
                for n in res:
                    anv[n] = np.zeros(len(t))
                for n in g.neighbors(seed):  # including res
                    x = anv[n]
                    anv[n] += (t-x) / cd[n]

        return res
