from operator import itemgetter

import numpy as np
import snap

from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph):
        self.graph = graph
        # observed graph
        self.g_observed = snap.TNGraph.New() if graph.directed else snap.TUNGraph.New()
        # crawled ids
        self.crawled = set()

    def crawl(self, ids):
        """
        Crawl specified nodes. The observed graph is updated.
        :param ids: nodes' ids to crawl
        :return:
        """
        if not isinstance(ids, list):
            ids = [ids]

        # Update observed graph
        for seed in ids:
            if seed in self.crawled:
                continue

            o = self.g_observed
            self.crawled.add(seed)
            if not o.IsNode(seed):
                o.AddNode(seed)

            neighs = snap.TIntV()
            # FIXME is there a simpler function to get neighbors?
            snap.GetNodesAtHop(self.graph.snap, seed, 1, neighs, self.graph.directed)
            for n in neighs:
                if not o.IsNode(n):
                    o.AddNode(n)
                o.AddEdge(seed, n)

    def get_observed_nodes(self) -> set:
        """ Get only observed nodes' ids excluding crawled ones
        """
        return set([n.GetId() for n in self.g_observed.Nodes()]) - self.crawled

    def get_all_nodes(self) -> set:
        """ Get crawled and observed nodes' ids
        """
        return set([n.GetId() for n in self.g_observed.Nodes()])


def get_top_hubs(graph, count):
    """
    Get top-count hubs of the graph
    :param graph:
    :param count:
    :return:
    """
    # TODO how to choose nodes at the border degree?
    node_deg = [(n.GetId(), n.GetDeg()) for n in graph.Nodes()]
    sorted_node_deg = sorted(node_deg, key=itemgetter(1), reverse=True)
    # print(sorted_node_deg)
    return [n for (n, d) in sorted_node_deg[:count]]


class AvrachenkovCrawler(Crawler):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """
    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph)
        assert n1 <= n <= self.graph.snap.GetNodes()
        assert k <= n-n1
        self.n1 = n1
        self.n = n
        self.k = k

    def first_step(self):
        graph_nodes = [n.GetId() for n in self.graph.snap.Nodes()]
        N = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[np.random.randint(N)]
            if seed in self.crawled:
                continue
            self.crawl(seed)
            i += 1
            if i == self.n1:
                break

    def second_step(self):
        observed_only = self.get_observed_nodes()

        # Get n2 max degree observed nodes
        obs_deg = []
        for o_id in observed_only:
            deg = self.g_observed.GetNI(o_id).GetDeg()
            obs_deg.append((o_id, deg))

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n-self.n1]

        # Crawl chosen nodes
        self.crawl([n for n, _ in max_degs])

        # assert len(self.crawled) == self.n
        # Take top-k of degrees
        hubs_detected = get_top_hubs(self.g_observed, self.k)
        return hubs_detected


