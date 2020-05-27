import logging

from utils import USE_CYTHON_CRAWLERS

if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from base.cbasic import MaximumObservedDegreeCrawler
    from base.cmultiseed import MultiCrawler
else:
    from base.graph import MyGraph
    from crawlers.basic import MaximumObservedDegreeCrawler
    from crawlers.multiseed import MultiCrawler

from graph_io import GraphCollections
from runners.animated_runner import Metric, AnimatedCrawlerRunner
from runners.crawler_runner import CrawlerRunner
from statistics import get_top_centrality_nodes, Stat
import multiprocessing


def test_runner(graph, animated=False, statistics: list = None, layout_pos=None, tqdm_info=''):
    import random
    # initial_seed = random.sample([n.GetId() for n in graph.snap.Nodes()], 1)[0]
    initial_seed = graph.random_nodes(1000)
    crawlers = [  ## ForestFireCrawler(graph, initial_seed=initial_seed), # FIXME fix and rewrite
        # RandomWalkCrawler(graph, initial_seed=initial_seed),
        # RandomCrawler(graph, initial_seed=initial_seed),
        #
        # DepthFirstSearchCrawler(graph, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.1, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.1, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.25, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.5, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.75, initial_seed=initial_seed),
        # SnowBallCrawler(graph, p=0.9, initial_seed=initial_seed),
        # BreadthFirstSearchCrawler(graph, initial_seed=initial_seed),  # is like take SBS with p=1

        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
        # MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
        # MaximumObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
        # MaximumObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed[0]),
        # MaximumObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed[0]),

        # PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed),

        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(2)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(3)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(4)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(5)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(10)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in
            range(20)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in
            range(50)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(100)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in
            range(500)]),
        MultiCrawler(graph, crawlers=[
            MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(1000)]),
        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, skl_mode=True, batch=1, initial_seed=initial_seed[i]) for i in
        #     range(10000)]),
    ]
    logging.info([c.name for c in crawlers])
    metrics = []
    target_set = {}  # {i.name: set() for i in statistics}
    for target_statistics in statistics:
        target_set = set(get_top_centrality_nodes(graph, target_statistics, count=int(0.1 * graph[Stat.NODES])))
        # creating metrics and giving callable function to it (here - total fraction of nodes)
        # metrics.append(Metric(r'observed' + target_statistics.name, lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]))
        metrics.append(Metric(r'crawled_' + target_statistics.name,  # TODO rename crawled to observed
                              lambda crawler, t: len(t.intersection(crawler.crawled_set)) / len(t), t=target_set
                              ))

        # print(metrics[-1], target_set)
    if animated == True:
        ci = AnimatedCrawlerRunner(graph,
                                   crawlers,
                                   metrics,
                                   budget=10000,
                                   step=10)
    else:
        ci = CrawlerRunner(graph,
                           crawlers,
                           metrics,
                           budget=0,
                           # step=ceil(10 ** (len(str(graph.nodes())) - 3)),
                           tqdm_info=tqdm_info,
                           # if 5*10^5 then step = 10**2,if 10^7 => step=10^4
                           # batches_per_pic=10,
                           # draw_mod='traversal', layout_pos=layout_pos,
                           )  # if you want gifs, draw_mod='traversal'. else: 'metric'
    ci.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # graph_name = 'digg-friends'       # with 261489 nodes and 1536577 edges
    # graph_name = 'douban'             # with 154908 nodes and  327162 edges
    # graph_name = 'facebook-wosn-links'# with  63392 nodes and  816831 edges
    # graph_name = 'slashdot-threads'   # with  51083 nodes and  116573 edges
    # graph_name = 'ego-gplus'          # with  23613 nodes and   39182 edges
    # graph_name = 'petster-hamster'    # with   2000 nodes and   16098 edges
    for graph_name in ['petster-friendships-dog', 'munmun_twitter_social', 'com-youtube',
                       'soc-pokec-relationships', 'flixster', 'youtube-u-growth', 'petster-friendships-cat', ]:
        g = GraphCollections.get(graph_name, giant_only=True)

    # graph_name = 'mipt'                 #  with 14313 nodes and 488852 edges
    # g = GraphCollections.get(graph_name, 'other', giant_only=True)

    # g._snap_graph = snap.GetMxWcc(g.snap)  # Taking only giant component
    print('Graph {} with {} nodes and {} edges'.format(graph_name, g.nodes(), g.edges()))
    # from crawlers.multiseed import test_carpet_graph, MultiCrawler
    # x,y = 7,7
    # graph_name = 'carpet_graph_'+str(x)+'_'+str(y)
    # g, layout_pos = test_carpet_graph(x,y)  # GraphCollections.get(name)

    import time

    start_time = time.time()
    processes = []
    iterations = multiprocessing.cpu_count() - 1  # making parallel itarations. Number of processes
    for exp in range(iterations):
        logging.info('Running iteration {}/{}'.format(exp, iterations))
        # little multiprocessing magic, that calculates several iterations in parallel
        p = multiprocessing.Process(target=test_runner, args=(g,),
                                    kwargs={'animated': False,
                                            'statistics': ["DEGREE_DISTR", "PAGERANK_DISTR", "K_CORENESS_DISTR"]
                                            # [s for s in Stat if 'DISTR' in s.name],
                                            # 'layout_pos':layout_pos,
                                            'tqdm_info': 'core-' + str(exp)
                                            })
        p.start()

    p.join()
    print("time elapsed: {:.2f}s, {}".format(time.time() - start_time, processes))

    from experiments.merger import merge

    # merge(graph_name,
    #       show=True,
    #       filter_only='MOD', )

# 'degree', 'betweenness', 'eccentricity', 'k-coreness',  'pagerank', 'clustering'
