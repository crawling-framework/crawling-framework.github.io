from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, CrawlerWithInitialSeed

import networkit as nk
from networkit._NetworKit import PLP, PLM


class MaximumObservedCommunityDegreeCrawler(CrawlerWithInitialSeed):
    short = 'MOCD'

    def __init__(self, graph: MyGraph, initial_seed: int=-1, **kwargs):
        super().__init__(graph, initial_seed=initial_seed, **kwargs)

        self.nk_graph = nk.Graph()  # observed graph - networkit copy
        self.node_map = {}  # nodes mapping nk_id -> snap_id
        self.rev_node_map = {}  # reverse nodes mapping snap_id -> nk_id

    def observe(self, node: int):
        if not super(MaximumObservedCommunityDegreeCrawler, self).observe(node):
            nk_id = self.nk_graph.addNode()
            self.node_map[nk_id] = node
            self.rev_node_map[node] = nk_id

    def crawl(self, seed: int):
        new_seen = super(MaximumObservedCommunityDegreeCrawler, self).crawl(seed)
        # Add new nodes to nk graph and mapping
        for node in new_seen:
            nk_id = self.nk_graph.addNode()
            self.node_map[nk_id] = node
            self.rev_node_map[node] = nk_id
        # Add edges
        for node in self._observed_graph.neighbors(seed):
            self.nk_graph.addEdge(self.rev_node_map[seed], self.rev_node_map[node])

        return new_seen

    def next_seed(self):
        # Detect communities
        plp = PLM(self.nk_graph)
        plp.run()
        partition = plp.getPartition()
        # Community mapping
        id_comm = {}  # nk_id -> comm_id
        comms_list = []
        for i in range(partition.numberOfSubsets()):
            nk_comm = partition.getMembers(i)
            comm = []
            for nk_i in nk_comm:
                id_comm[nk_i] = i
            # if len(comm) > 0:
            #     comms_list.append(comm)

        # Get MOCD nodes
        id_cd = {}  # nk_id -> comm degree
        max_comms = -1
        candidate = -1
        # for nk_id in self.nk_graph.iterNodes():
        for node in self._observed_set:
            nk_id = self.rev_node_map[node]
            comms = set()
            for neigh_id in self.nk_graph.neighbors(nk_id):
                if not neigh_id in id_comm:
                    id_comm[neigh_id] = len(id_comm)
                comms.add(id_comm[neigh_id])

            id_cd[node] = len(comms)

            if len(comms) > max_comms:
                max_comms = len(comms)
                candidate = nk_id
        # print(len(self._crawled_set), "avg nodes per comm", [set([self.node_map[x] for x in partition.getMembers(i)]) for i in range(partition.numberOfSubsets())])

        assert candidate >= 0
        return self.node_map[candidate]

