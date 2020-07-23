from graph_io import GraphCollections
from running.animated_runner import Metric, AnimatedCrawlerRunner
from statistics import Stat, get_top_centrality_nodes
from utils import USE_NETWORKIT, PICS_DIR

if USE_NETWORKIT:
    from networkit.community import PLM, Modularity

from crawlers.cbasic import MaximumObservedDegreeCrawler, Crawler, BreadthFirstSearchCrawler, RandomCrawler, PreferentialObservedDegreeCrawler
from crawlers.multiseed import MultiInstanceCrawler


def test_plm(g):
    print(g['NODES'], g['EDGES'], g[Stat.AVG_DEGREE])
    m = g['EDGES']
    nk = g.networkit()
    plm = PLM(nk, refine=False, gamma=1)
    plm.run()
    partition = plm.getPartition()
    # for p in partition:
    len_num = {}
    exp_q = 0
    for i in range(partition.numberOfSubsets()):
        comm = partition.getMembers(i)
        for i in comm:
            exp_q += (g.deg(i)) ** 2

        # print(len(comm), comm)
        size = len(comm)
        if size not in len_num:
            len_num[size] = 0
        len_num[size] += 1
        if size > 1:
            print(size, list(comm)[:10])

    print(exp_q, 4*m*m)
    print(len_num)
    print('total', sum(k * v for k, v in len_num.items()))
    # print(partition.getMembers(0))
    # print(partition.getMembers(1))
    # print(partition.getMembers(2))
    mod = Modularity().getQuality(partition, nk)
    print(mod)


class CrawlerWithComms(Crawler):
    def __init__(self, graph, num_comms, clas, **clas_kwargs):
        self.crawler = clas(graph, **clas_kwargs)
        super().__init__(graph, name=self.crawler.name)
        # self.crawled_set = self.crawler.crawled_set

        # Enumerate comms in desc order
        comms = graph[Stat.PLM_COMMUNITIES]
        self.id_comm = {}  # node_id -> comm_id
        self.comm_size = {}  # comm_id -> size
        comms = sorted(comms, key=len, reverse=True)
        for c_id, comm in enumerate(comms):
            # print(len(comm))
            self.comm_size[c_id] = len(comm)
            for n in comm:
                self.id_comm[n] = c_id

        self.num_comms = num_comms
        self.counters = [0 for _ in range(self.num_comms)]

    @property
    def crawled_set(self):
        return self.crawler.crawled_set

    @property
    def observed_set(self):
        return self.crawler.observed_set

    @property
    def nodes_set(self):
        return self.crawler.nodes_set

    def next_seed(self) -> int:
        return self.crawler.next_seed()

    def crawl(self, seed: int) -> set:
        res = self.crawler.crawl(seed)
        comm_id = self.id_comm[seed]
        if comm_id < self.num_comms:
            self.counters[comm_id] += 1  # / self.comm_size[comm_id]
        return res

    def crawl_budget(self, budget: int, *args):
        for _ in range(budget):
            self.crawl(self.next_seed())


def test_crawler_comms(g):
    # Measure comm detecting rate
    p = 0.1
    count = int(p*g.nodes())
    target_set = set(get_top_centrality_nodes(g, Stat.DEGREE_DISTR, count))

    num = 16
    random_seeds = g.random_nodes(100)
    crawlers = [
        CrawlerWithComms(g, num, clas=PreferentialObservedDegreeCrawler, batch=1, initial_seed=1),

        CrawlerWithComms(g, num, clas=MultiInstanceCrawler, crawlers=[
            MaximumObservedDegreeCrawler(g, initial_seed=random_seeds[i]) for i in range(10)
        ]),

        CrawlerWithComms(g, num, clas=MaximumObservedDegreeCrawler, initial_seed=1),
        CrawlerWithComms(g, num, clas=BreadthFirstSearchCrawler, initial_seed=1),
        # CrawlerWithComms(g, num, clas=RandomCrawler, initial_seed=1),
    ]

    metrics = [
        Metric("Re (crawled)", lambda crawler: len(crawler.crawled_set.intersection(target_set))/len(target_set)),
        Metric("Re (all)", lambda crawler: len(crawler.nodes_set.intersection(target_set)) / len(target_set)),
    ]
    # metrics = [
    #     Metric("comm%s (%s)" % (i, crawlers[0].comm_size[i]),
    #            lambda crawler, i: crawler.counters[i],
    #            i=i) for i in range(num)
    # ]

    budget = 10000  # int(g.nodes() / 10)
    step = max(1, int(budget / 30))
    acr = AnimatedCrawlerRunner(g, crawlers, metrics, budget=budget, step=step)
    acr.run(swap_coloring_scheme=False,
            save_to_file=PICS_DIR + "/comms_%s_%s.png" % (g.name, "_".join([c.name for c in crawlers])))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # a = [1003, 924, 930, 1674, 720, 342, 1482, 804, 252, 708, 876, 1410, 1374, 1119, 1104, 1392, 351, 939]
    # print(sum(a))

    name = 'petster-hamster'
    # name = 'digg-friends'
    name = 'Pokec'
    g = GraphCollections.get(name, 'konect')

    # name = 'uk-2005'  # netrepo
    # name = 'github'  # netrepo
    # name = 'sc-shipsec5'  # netrepo  Q=0.8995
    # name = 'socfb-Bingham82'  # netrepo  Q=0.4597
    # name = 'tech-p2p-gnutella'  # netrepo  Q=0.5022
    # name = 'karate'  # netrepo
    # name = 'rec-amazon'  # netrepo Q=0.9898
    # name = 'soc-slashdot'  # netrepo Q=0.362
    # name = 'socfb-Bingham82'  # netrepo Q=0.4619
    # name = 'ca-MathSciNet'  # netrepo Q=

    # g = GraphCollections.get(name, 'netrepo')

    # test_plm(g)
    test_crawler_comms(g)

