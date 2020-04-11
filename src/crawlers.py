import glob
import heapq
import json
import logging
import os
from abc import ABC
from operator import itemgetter

import numpy as np
import snap
from numpy import random

from centralities import get_top_centrality_nodes
from experiments import drawing_graph
from graph_io import MyGraph


class Crawler(object):

    def __init__(self, graph: MyGraph):
        # original graph
        self.orig_graph = graph
        # observed graph
        self.observed_graph = MyGraph.new_snap(directed=graph.directed, weighted=graph.weighted)
        # observed snap graph
        # crawled ids set
        self.crawled_set = set()
        # observed ids set excluding crawled ones
        self.observed_set = set()

    @property
    def nodes_set(self) -> set:
        """ Get nodes' ids of observed graph (crawled and observed). """
        return set([n.GetId() for n in self.observed_graph.snap.Nodes()])

    def crawl(self, seed: int):
        """
        Crawl specified nodes. The observed graph is updated, also crawled and observed set.
        :param seed: node id to crawl
        :return: whether the node was crawled
        """
        seed = int(seed)  # convert possible int64 to int, since snap functions would get error
        if seed in self.crawled_set:
            return False  # if already crawled - do nothing

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

    def next_seed(self):
        raise NotImplementedError()

    def crawl_budget(self, budget: int, *args):
        for _ in range(budget):
            seed = self.next_seed()
            while not self.crawl(seed):
                continue
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))


class MultiSeedCrawler(Crawler, ABC):
    """
    great class to Avrachenkov and other crawlers starting with n1 seeds
    """

    def __init__(self, graph: MyGraph):
        super().__init__(graph)
        # assert n1 <= self.budget_left <= self.orig_graph.snap.GetNodes()
        # assert k <= self.budget_left - n1
        # self.n1 = n1  # n1 seeds crawled on first steps, then comes crawler
        self.seed_sequence_ = []  # sequence of tries to add nodes
        self.initial_seeds = []  # list of initial seeds to iter or jump into
        self.budget_left = 1  # how many iterations left. stops when 0
        # print("n1={}, budget={}, nodes={}".format(self.n1, self.budget_left, self.orig_graph.snap.GetNodes()))
        self.method_name = ""  # will be used in names of files

    def crawl_multi_seed(self, n1):
        if n1 <= 0:  # if there is no parallel seeds, method do nothing
            return False
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        multi_seeds = [int(node) for node in np.random.choice(graph_nodes, n1)]
        print("seeds for multi crawl:", multi_seeds)
        for seed in multi_seeds:
            self.crawl(seed)
            self.sequence_append(seed)

        print("observed set", list(self.observed_set))

    def crawl(self, seed):
        """
        Crawls given seed
        Decrease self.budget_left only if crawl is successful
        """
        if super().crawl(seed):
            self.budget_left -= 1
            logging.debug("{}.-- seed {}, crawled #{}: {}, observed #{},".format(len(self.seed_sequence_), seed,
                                                                                 len(self.crawled_set),
                                                                                 self.crawled_set,
                                                                                 len(self.observed_set),
                                                                                 self.observed_set))
            return True
        else:
            logging.debug("{}. budget ={}, seed = {}, in crawled set={}, observed ={}".format(len(self.seed_sequence_),
                                                                                              self.budget_left, seed,
                                                                                              self.crawled_set,
                                                                                              self.observed_set))
            return False

    def crawl_budget(self, budget, p=0, file=False):
        """
        Crawl until done budget
        :param p: probability to jump into one of self.initial_seed nodes
        :param budget: how many iterations left
        :param file: - if you need to
        :return:
        """
        self.budget_left = min(budget, self.orig_graph.snap.GetNodes() - 1)
        if random.randint(0, 100, 1) < p * 100:
            print("variety play")
            self.crawl(int(np.random.choice(self.initial_seeds, 1)[0]))
            self.budget_left -= 1

        while (self.budget_left > 0) and (len(self.observed_set) > 0) \
                and (self.observed_graph.snap.GetNodes() <= self.orig_graph.snap.GetNodes()):
            seed = self.next_seed()
            self.crawl(seed)

            if file:
                self.sequence_append(seed)
            # logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
            #               (seed, self.crawled_set, self.observed_set, self.nodes_set))

    def sequence_append(self, seed):
        self.seed_sequence_.append(int(seed))  # TODO: need to be optimized in outer scope
        with open("./data/crawler_history/crawled{}{}.json".format(self.method_name,
                                                                   str(len(self.seed_sequence_)).zfill(6)),
                  'w') as cr_file:
            json.dump(list(self.crawled_set), cr_file)
        with open("./data/crawler_history/observed{}{}.json".format(self.method_name,
                                                                    str(len(self.seed_sequence_)).zfill(6)),
                  'w') as ob_file:
            json.dump(list(self.observed_set), ob_file)


class RandomWalk(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.prev_seed = 1  # previous node, that was already crawled
        self.method_name = 'RW_'

    def next_seed(self):
        node_neighbours = self.observed_graph.neighbors(self.prev_seed)
        # for walking we need to step on already crawled nodes too
        if len(node_neighbours) == 0:
            node_neighbours = tuple(self.observed_set)
        return int(random.choice(node_neighbours, 1)[0])

    def crawl(self, seed):
        super().crawl(seed)
        self.prev_seed = seed


class BreadthFirstSearchCrawler(MultiSeedCrawler):  # в ширину
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.bfs_queue = []
        self.method_name = 'BFS'

    def next_seed(self):
        while self.bfs_queue[0] not in self.observed_set:
            self.bfs_queue.pop(0)
        return self.bfs_queue[0]

    def crawl(self, seed):
        for n in self.orig_graph.neighbors(seed):
            self.bfs_queue.append(n)
        return super(BreadthFirstSearchCrawler, self).crawl(seed)


class DepthFirstSearchCrawler(MultiSeedCrawler):  # TODO
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.dfs_queue = []
        self.dfs_counter = 0
        self.method_name = 'DFS'
        # int(random.choice(tuple(self.observed_set)))

    def next_seed(self):
        while self.dfs_queue[0] not in self.observed_set:
            self.dfs_queue.pop(0)
        return self.dfs_queue[0]

    def crawl(self, seed):
        self.dfs_counter = 0
        for n in self.orig_graph.neighbors(seed):
            self.dfs_counter += 1
            self.dfs_queue.insert(self.dfs_counter, n)
        return super(DepthFirstSearchCrawler, self).crawl(seed)


class RandomCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.method_name = 'RC_'

    def next_seed(self):
        return random.choice(tuple(self.observed_set))


class MaximumObservedDegreeCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, top_k_mod=1):
        super().__init__(orig_graph)
        self.mod_queue = []
        self.top_k_mod = top_k_mod
        self.method_name = 'MOD'

    def next_seed(self):
        if len(self.mod_queue) == 0:  # making array of topk degrees
            deg_dict = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                        if node.GetId() not in self.crawled_set}

            heap = [(-value, key) for key, value in deg_dict.items()]
            min_iter = min(self.top_k_mod, len(deg_dict))
            self.mod_queue = [heapq.nsmallest(self.top_k_mod, heap)[i][1] for i in range(min_iter)]
        return self.mod_queue.pop(0)


class AvrachenkovCrawler(Crawler, ABC):
    """
    Algorithm from paper "Quick Detection of High-degree Entities in Large Directed Networks" (2014)
    https://arxiv.org/pdf/1410.0571.pdf
    """

    def __init__(self, graph, n=1000, n1=500, k=100):
        super().__init__(graph)
        print(n1, n, self.orig_graph.snap.GetNodes())
        assert n1 <= n <= self.orig_graph.snap.GetNodes()
        assert k <= n - n1
        self.n1 = n1
        self.n = n
        self.k = k
        self.method_name = 'AVR'

    def first_step(self):
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        size = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[np.random.randint(size)]
            if seed in self.crawled_set:
                continue
            self.crawl(seed)
            i += 1
            if i == self.n1:
                break

    def second_step(self):
        observed_only = self.observed_set

        # Get n2 max degree observed nodes
        obs_deg = []
        g = self.observed_graph.snap
        for o_id in observed_only:
            deg = g.GetNI(o_id).GetDeg()
            obs_deg.append((o_id, deg))

        max_degs = sorted(obs_deg, key=itemgetter(1), reverse=True)[:self.n - self.n1]

        # Crawl chosen nodes
        [self.crawl(n) for n, _ in max_degs]

        # assert len(self.crawled) == self.n
        # Take top-k of degrees
        hubs_detected = get_top_centrality_nodes(self.observed_graph, 'degree', self.k)
        return hubs_detected


def test_graph():
    g = snap.TUNGraph.New()

    for i in range(17):
        g.AddNode(i)

    g.AddEdge(1, 2)
    g.AddEdge(2, 3)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    g.AddEdge(1, 6)
    g.AddEdge(6, 7)
    g.AddEdge(8, 7)
    g.AddEdge(8, 16)
    g.AddEdge(8, 9)
    g.AddEdge(8, 10)
    g.AddEdge(8, 7)
    g.AddEdge(0, 10)
    g.AddEdge(0, 9)

    g.AddEdge(11, 12)
    g.AddEdge(12, 13)
    g.AddEdge(14, 12)
    g.AddEdge(14, 13)
    g.AddEdge(15, 13)
    g.AddEdge(15, 5)
    g.AddEdge(11, 16)

    # for dfs

    g.AddEdge(14, 0)
    g.AddEdge(3, 0)

    return g


# if __name__ == '__main__':
file_path = "./data/crawler_history/"
if os.path.exists(file_path):
    for file in glob.glob("./data/crawler_history/*.json"):
        os.remove(file)
else:
    os.makedirs(file_path)

file_path = "./data/gif_files/"
if os.path.exists(file_path):
    for file in glob.glob("./data/gif_files/*.png"):
        os.remove(file)
else:
    os.makedirs(file_path)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

#


Graph = MyGraph.new_snap(name='test', directed=False)
Graph._snap_graph = test_graph()  # GraphCollections.get('dolphins').snap  #
print("N=%s E=%s" % (Graph.snap.GetNodes(), Graph.snap.GetEdges()))

total_budget = 100  # min(100,Graph.snap.GetNodes())
n1 = 1
top_k_mod = 5
k = 4
pos = None
crawlers_dictionary = {'MOD': MaximumObservedDegreeCrawler,
                       'DFS': DepthFirstSearchCrawler,
                       'RW_': RandomWalk,
                       'BFS': BreadthFirstSearchCrawler,
                       'RC_': RandomCrawler,
                       }

crawlers = [(name, crawlers_dictionary[name]) for name in crawlers_dictionary]

for method_name, Crawler_class in crawlers:
    print("Running {} with budget={}".format(method_name, total_budget))
    if method_name == 'MOD':
        crawler = Crawler_class(Graph, top_k_mod=top_k_mod)
        crawler.crawl_multi_seed(n1=n1)
    else:
        crawler = Crawler_class(Graph)
        crawler.crawl_multi_seed(n1=n1)

    crawler.crawl_budget(total_budget, file=True)
    # print("after first: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
    #       " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))
    # print("after second: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
    #       " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))
    # print("Total iterations:", len(crawler.seed_sequence_))
    # print("sequence of seeds:", crawler.seed_sequence_)

    with open("./data/crawler_history/sequence.json", 'w') as f:
        json.dump(crawler.seed_sequence_, f)

    crawler.orig_graph.path = './data/crawler_history/orig_graph.txt'
    crawler.orig_graph.save_snap_edge_list()
    print('saved')

    print("drawing")
    graph_path = './data/crawler_history/'
    crawler_history_path = "./data/crawler_history/"
    pngs_path = "./data/gif_files/"
    gif_export_path = './data/graph_traversal.gif'

    pos = drawing_graph.make_png_history(crawler.orig_graph.snap, pngs_path, crawler_history_path,
                                         method_name, pos=pos, labels=True)

# if method_name == 'AVR':
#     crawler = Crawler_class(Graph, n1=n1, k=k, n=total_budget)
#     crawler.first_step()
#     crawler.second_step()


# 'AVR':AvrachenkovCrawler,
