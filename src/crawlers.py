import glob
import heapq
import json
import logging
import os
import random
from abc import ABC
from operator import itemgetter

import numpy as np
import snap
from scipy import stats
from tqdm import tqdm

from centralities import get_top_centrality_nodes
from experiments import drawing_graph
from graph_io import GraphCollections
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
        self.initial_seeds = []  # store n1 seeds # TODO maybe use it not only in Random Walk (Frontier Sampling)
        self.budget_left = 1  # how many iterations left. stops when 0
        # print("n1={}, budget={}, nodes={}".format(self.n1, self.budget_left, self.orig_graph.snap.GetNodes()))
        self.crawler_name = ""  # will be used in names of files
        # self.components_current_seeds

    def crawl_multi_seed(self, n1):
        if n1 <= 0:  # if there is no parallel seeds, method do nothing
            return False
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        # TODO do something if there are doublicates in initial seeds
        self.initial_seeds = [int(node) for node in np.random.choice(graph_nodes, n1)]
        for seed in self.initial_seeds:
            self.crawl(seed)
        print("observed set after multiseed", list(self.observed_set))

    def crawl(self, seed):
        """
        Crawls given seed
        Decrease self.budget_left only if crawl is successful
        """
        # http://conferences.sigcomm.org/imc/2010/papers/p390.pdf
        # TODO implement frontier choosing of component (one from m)
        self.seed_sequence_.append(seed)  # updates every TRY of crawling
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
        :param p: probability to jump into one of self.initial_seed nodes  # TODO do something with it. Mb delete?
        :param budget: how many iterations left
        :param file: - if you need to
        :return:
        """
        self.budget_left = min(budget, self.observed_graph.snap.GetNodes() - 1)
        if np.random.randint(0, 100, 1) < p * 100:  # TODO to play with this dead staff
            print("variety play")
            self.crawl(int(np.random.choice(self.initial_seeds, 1)[0]))
            self.budget_left -= 1

        while (self.budget_left > 0) and (len(self.observed_set) > 0) \
                and (self.observed_graph.snap.GetNodes() <= self.orig_graph.snap.GetNodes()):
            seed = self.next_seed()
            self.crawl(seed)

            # if file:
            logging.debug("seed:%s. crawled:%s, observed:%s, all:%s" %
                          (seed, self.crawled_set, self.observed_set, self.nodes_set))


class RandomWalkCrawler(MultiSeedCrawler):
    """
    Normal random work if n1=1. Otherwise it is Frontier Crawling, that chooses from self.initial_seeds ~ degree
    """

    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.prev_seed = 1  # previous node, that was already crawled
        self.crawler_name = 'RW_'

    def next_seed(self):
        # step 4 from paper about Frontier Sampling.  taken from POD.next_seed
        prob_func = {node: self.observed_graph.snap.GetNI(node).GetDeg() for node in self.initial_seeds}
        keys, values = zip(*prob_func.items())
        values = np.array(values) / sum(values)
        self.discrete_distribution = stats.rv_discrete(values=(keys, values))
        self.prev_seed = [node for node in self.discrete_distribution.rvs(size=1)].pop(0)
        # print("node degrees (probabilities)", prob_func)

        # original Random Walk
        node_neighbours = self.observed_graph.neighbors(self.prev_seed)
        # for walking we need to step on already crawled nodes too
        if len(node_neighbours) == 0:
            node_neighbours = tuple(self.observed_set)
        next_seed = int(np.random.choice(node_neighbours, 1)[0])
        self.initial_seeds[self.initial_seeds.index(self.prev_seed)] = next_seed
        return next_seed

    def crawl(self, seed):
        super().crawl(seed)
        self.prev_seed = seed


class BreadthFirstSearchCrawler(MultiSeedCrawler):  # в ширину
    def __init__(self, orig_graph: MyGraph):
        super().__init__(orig_graph)
        self.bfs_queue = []
        self.crawler_name = 'BFS'

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
        self.crawler_name = 'DFS'

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
        self.crawler_name = 'RC_'

    def next_seed(self):
        return int(np.random.choice(tuple(self.observed_set)))


class MaximumObservedDegreeCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, top_k=1):
        super().__init__(orig_graph)
        self.mod_queue = []
        self.top_k = top_k  # crawling by batches if > 1 # TODO make another MOD crawler with batches
        self.crawler_name = 'MOD'

    def next_seed(self):
        if len(self.mod_queue) == 0:  # making array of topk degrees
            deg_dict = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                        if node.GetId() not in self.crawled_set}

            heap = [(-value, key) for key, value in deg_dict.items()]
            min_iter = min(self.top_k, len(deg_dict))
            self.mod_queue = [heapq.nsmallest(self.top_k, heap)[i][1] for i in range(min_iter)]
        return self.mod_queue.pop(0)


class PreferentialObservedDegreeCrawler(MultiSeedCrawler):
    def __init__(self, orig_graph: MyGraph, top_k=1):
        super().__init__(orig_graph)
        self.discrete_distribution = None
        self.pod_queue = []  # queue of nodes to proceed in batch
        self.top_k = top_k  # crawling by batches if > 1 #
        self.crawler_name = 'POD'

    def next_seed(self):
        if len(self.pod_queue) == 0:  # when batch ends, we create another one
            prob_func = {node.GetId(): node.GetDeg() for node in self.observed_graph.snap.Nodes()
                         if node.GetId() not in self.crawled_set}
            keys, values = zip(*prob_func.items())
            values = np.array(values) / sum(values)

            self.discrete_distribution = stats.rv_discrete(values=(keys, values))
            self.pod_queue = [node for node in self.discrete_distribution.rvs(size=self.top_k)]

        # print("node degrees (probabilities)", prob_func)
        return self.pod_queue.pop(0)


class ForestFireCrawler(BreadthFirstSearchCrawler):
    """Algorythm from https://dl.acm.org/doi/abs/10.1145/1081870.1081893
    with my little modification - stuck_ends, it is like illegitimate son of BFS and RC
    :param p - forward burning probability of algorythm
    :param stuck_ends - if true, finishes when queue is empty, otherwise crawl random from observed
    """

    def __init__(self, orig_graph: MyGraph, p=0.35, stuck_ends=False):
        super().__init__(orig_graph)
        self.crawler_name = 'FFC'
        self.p = p

    # next_seed is the same with BFS, just choosing ambassador node w=seed, except empty queue
    def next_seed(self):
        while self.bfs_queue[0] not in self.observed_set:
            self.bfs_queue.pop(0)
            if len(self.bfs_queue) == 0:  # if we get stucked, choosing random from observed
                return int(np.random.choice(tuple(self.observed_set)))
        return self.bfs_queue[0]

    def crawl(self, seed):
        degree = self.orig_graph.snap.GetNI(seed).GetDeg()
        # computing x - number of friends to add
        # print("seed", seed, self.p, degree, (1 - self.p) ** (-1) / degree )
        # in paper (1-p)**(-1) == E == degree * bin_prob
        x = np.random.binomial(degree, self.p)  # (1 - self.p) ** (-1) / degree)
        x = max(1, min(x, len(self.orig_graph.neighbors(seed))))
        intersection = [n for n in self.orig_graph.neighbors(seed)]
        burning = [int(n) for n in random.sample(intersection, x)]
        # print("FF: queue:{},obs:{},crawl:{},x1={},burn={}".format(self.bfs_queue, self.observed_set,self.crawled_set,x, burning))
        for node in burning:
            self.bfs_queue.append(node)

        return super(BreadthFirstSearchCrawler, self).crawl(seed)


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
        self.crawler_name = 'AVR'

    def first_step(self):
        graph_nodes = [n.GetId() for n in self.orig_graph.snap.Nodes()]
        size = len(graph_nodes)

        i = 0
        while True:
            seed = graph_nodes[int(np.random.randint(size))]
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


def Crawler_Runner(Graph: MyGraph, crawler_name: str, total_budget=1, n1=1,
                   top_set=False, jsons=False, gif=False, ending_sets=False):
    """   # TODO other arguments like top_k=False,

    The core function that takes crawler and does everything with it (crawling budget, export results....)
    :param crawler_name: example of Crawler class (preferably MultiSeedCrawler class )  
    :param total_budget:  how many crawling operations we could make
    :param n1:    number of random crawled nodes at the beggining. n1<2 if dont need to.
    :param top_set: dictionary of needed sets of nodes to intersect in history (ex {degree: {1,2,3,},kcore: {1,3,6,}}...)
    :param jsons: if need to export final dumps of crawled and observed sets after ending
    :param pngs:  if need to export pngs  in
    :param gif:   if need to make gif from traversal history every #gif times (ex gif=2 - plot every two iterations
    :param ending_sets: if need to make json files of crawled and observed sets at the end of crawling
    :return:
    """

    if crawler_name in ('MOD', 'POD'):
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph, top_k=top_k)
    else:
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph)

    if gif:  # TODO need to replace directories in utils.py after all and make normal directions
        graph_path = './data/crawler_history/'
        crawler_history_path = "./data/crawler_history/"
        pngs_path = "./data/gif_files/"
        gif_export_path = './data/graph_traversal.gif'
        gif_counter = 0
        # for drawing
        from matplotlib import pyplot as plt
        plt.figure(figsize=(14, 6))
        # print(layout_pos)
        # print([i for i in layout_pos])
        # fig, ax = plt.subplots()
        # print("lp",[layout_pos[v][0] for v in layout_pos])
        # axis_x_coords = [layout_pos[v][0] for v in layout_pos]
        # axis_y_coords = [layout_pos[v][1] for v in layout_pos]

        # file preparing
        if os.path.exists(pngs_path):
            for file in glob.glob(pngs_path + crawler_name + "*.png"):
                os.remove(file)
        else:
            os.makedirs(pngs_path)

        # with open(crawler_history_path + 'sequence.json', 'r') as f:
        #    sequence = json.load(f)

    if top_set:  # TODO need to make plots for every centrality (utils.py CENTRALITIES)
        top_set = dict()
        centralities = top_set.keys()
        history_plots = {centr: [] for centr in centralities}  # list of numbers of intersections for every step
        # plot of crawled history will be just plotting this graph plt.plot( history_plots.)
        # TODO + all node set. it was 'nodes' in our paper

    if n1:  # crawl first n1 seeds
        print('RUNNER: crawl_multi_seed{}'.format(n1))
        crawler.crawl_multi_seed(n1=n1)

    for iterator in tqdm(range(total_budget)):
        # print('RUNNER: iteration:{}/{}'.format(iterator,total_budget))
        crawler.crawl_budget(1)  # file=True)  # TODO return crawling export in files

        if (gif) and (iterator % gif == 0):
            from networkx import draw
            # TODO some strange things coulf happen, because need to give @pos back
            last_seed = crawler.seed_sequence_[-1]
            networkx_graph = crawler.orig_graph.networkx_graph

            # coloring nodes
            s = [n.GetId() for n in crawler.orig_graph.snap.Nodes()]
            s.sort()
            gen_node_color = ['gray'] * (max(s) + 1)
            for node in crawler.observed_set:
                #    print(node, gen_node_color)
                gen_node_color[node] = 'y'
            for node in crawler.crawled_set:
                gen_node_color[node] = 'cyan'
            gen_node_color[last_seed] = 'red'

            plt.title(crawler_name + " " + str(iterator) + '  ' + "current node:" + str(last_seed))
            draw(networkx_graph, pos=layout_pos, with_labels=True, node_size=100,
                 node_color=[gen_node_color[node] for node in networkx_graph.nodes]
                 # if node in networkx_graph.nodes()]
                 )
            # plt.xlim(min(axis_x_coords) - 1, max(axis_x_coords) + 1)
            # plt.ylim(min(axis_y_coords) - 1, max(axis_y_coords) + 1)

            plt.savefig(pngs_path + '/gif{}{}.png'.format(crawler_name, str(iterator).zfill(3)))
            plt.clf()

        # finishes when see all nodes. it means even if they are only obsered, we know most part of degrees
        if len(crawler.observed_set) + len(crawler.crawled_set) == crawler.orig_graph.snap.GetNodes():
            print("Finished coz crawled+observed all")
            break

    # drawing_graph.draw_new_png(crawler.observed_graph, crawler.observed_set, crawler.crawled_set,
    #                          iterator, last_seed=crawler.seed_sequence_[-1], pngs_path=pngs_path,
    #                           crawler_name = crawler_name, labels = False)
    # if top_set: # TODO every (successful) crawling iteration need to calculate intersection and write in history_plots

    # TODO uncomment this or do something with empty node numbers (ex. nodes starts from 1, or does not exist)
    # nx_graph = crawler.observed_graph.networkx_graph  # snap_to_nx_graph(snap_graph)
    # # node_list = list(nx_graph.nodes())
    # print("----", crawler.observed_set, crawler.crawled_set)
    # for node in range(max(nx_graph.nodes())):  # filling empty indexes in graph
    #     if node not in nx_graph.nodes():
    #         nx_graph.add_node(1)
    #         print('i added node', node)

    if gif:
        drawing_graph.make_gif(crawler_name=crawler_name, pngs_path=pngs_path)

    print(crawler_name + ": after first: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))

    # dumps final versions of crawled_set and observed_set after finishing crawling all budget.
    if ending_sets:
        with open("./data/crawler_history/crawled{}{}.json".format(crawler.crawler_name,
                                                                   str(len(crawler.seed_sequence_)).zfill(6)),
                  'a') as cr_file:
            json.dump(list(crawler.crawled_set), cr_file)
        with open("./data/crawler_history/observed{}{}.json".format(crawler.crawler_name,
                                                                    str(len(crawler.seed_sequence_)).zfill(6)),
                  'a') as ob_file:
            json.dump(list(crawler.observed_set), ob_file)

    return crawler


# using simple dictionary to work with crawlers. aAlso .keys of it are names of crawlers for files
CRAWLERS_DICTIONARY = {'POD': PreferentialObservedDegreeCrawler,
                       'MOD': MaximumObservedDegreeCrawler,
                       'DFS': DepthFirstSearchCrawler,
                       'BFS': BreadthFirstSearchCrawler,
                       'RWC': RandomWalkCrawler,
                       'RC_': RandomCrawler,
                       'FFC': ForestFireCrawler,
                       'AVR': AvrachenkovCrawler,  # dont use, it iscompletely different
                       }


def test_graph():
    g = snap.TUNGraph.New()

    for i in range(19):
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
    g.AddEdge(1, 17)
    g.AddEdge(7, 17)
    g.AddEdge(1, 18)
    g.AddEdge(7, 18)
    g.AddEdge(11, 15)

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


def test_crawlers():
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

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    Graph = MyGraph.new_snap(name='test', directed=False)
    Graph._snap_graph = GraphCollections.get('dolphins').snap  # test_graph()  #
    # Graph.snap.AddNode(0)
    # Graph.snap.AddEdge(0, 1)
    layout_pos = Graph.graph_layout_pos

    print("N=%s E=%s" % (Graph.snap.GetNodes(), Graph.snap.GetEdges()))

    # min(100,Graph.snap.GetNodes())
    n1 = 2
    total_budget = min(100 - n1, Graph.snap.GetNodes())
    top_k = 5
    k = 4
    # pos = None  # position layout for drawing similar graphs (with nodes on same positions). updates at the end

    # crawlers = [(name, crawlers_dictionary[name]) for name in crawlers_dictionary]
    crawlers = ['DFS', 'BFS', 'RWC', 'RC_']  # 'FFC','MOD', 'POD',

    for crawler_name in crawlers:
        print("Running {} with budget={}, n1={}".format(crawler_name, total_budget, n1))
        crawler = Crawler_Runner(Graph, crawler_name, total_budget=total_budget, n1=n1,
                                 gif=1)
        print("Seed sequence due crawling:", crawler.seed_sequence_)
    # with open("./data/crawler_history/sequence.json", 'w') as f:
    #     json.dump(crawler.seed_sequence_, f)


if __name__ == '__main__':
    # test_graph()
    test_crawlers()
