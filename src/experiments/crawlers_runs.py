import logging

from utils import USE_CYTHON_CRAWLERS
import time
if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from base.cbasic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, \
        SnowBallCrawler, PreferentialObservedDegreeCrawler, RandomCrawler, RandomWalkCrawler
    from base.cmultiseed import MultiCrawler
else:
    from base.graph import MyGraph
    from crawlers.basic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, \
        SnowBallCrawler, PreferentialObservedDegreeCrawler, RandomCrawler, RandomWalkCrawler
    from crawlers.multiseed import MultiCrawler

from graph_io import GraphCollections
from runners.animated_runner import Metric, AnimatedCrawlerRunner
from runners.crawler_runner import CrawlerRunner
from statistics import get_top_centrality_nodes, Stat
import multiprocessing


def test_runner(graph, animated=False, statistics: list = None, top_k_percent=0.1, layout_pos=None, tqdm_info=''):
    import random
    # initial_seed = random.sample([n.GetId() for n in graph.snap.Nodes()], 1)[0]
    print([stat_name.name for stat_name in statistics])
    initial_seed = graph.random_nodes(1000)
    ranges = [2, 3, 4, 5, 10, 30, 100, 1000]
    crawlers = [  ## ForestFireCrawler(graph, initial_seed=initial_seed), # FIXME fix and rewrite
                   RandomWalkCrawler(graph, initial_seed=initial_seed[0]),
                   RandomCrawler(graph, initial_seed=initial_seed[0]),

                   DepthFirstSearchCrawler(graph, initial_seed=initial_seed[0]),
                   SnowBallCrawler(graph, p=0.1, initial_seed=initial_seed[0]),
                   SnowBallCrawler(graph, p=0.25, initial_seed=initial_seed[0]),
                   SnowBallCrawler(graph, p=0.5, initial_seed=initial_seed[0]),
                   SnowBallCrawler(graph, p=0.75, initial_seed=initial_seed[0]),
                   SnowBallCrawler(graph, p=0.9, initial_seed=initial_seed[0]),
                   BreadthFirstSearchCrawler(graph, initial_seed=initial_seed[0]),  # is like take SBS with p=1

                   MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
                   MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
                   MaximumObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
                   MaximumObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed[0]),
                   MaximumObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed[0]),

                   PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
                   PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
                   PreferentialObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
                   PreferentialObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed[0]),
                   PreferentialObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed[0]),
               ] \
               + [
                   MultiCrawler(graph, crawlers=[
                       PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in
                       range(range_i)])
                   for range_i in ranges \
                   #
               ] + [MultiCrawler(graph, crawlers=[
        BreadthFirstSearchCrawler(graph, initial_seed=initial_seed[i]) for i in range(range_i)])
                    for range_i in ranges \
                    #
                    ] + [
                   MultiCrawler(graph, crawlers=[
                       MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in
                       range(range_i)])
                   for range_i in ranges \
 \
                   ]
    logging.info([c.name for c in crawlers])
    metrics = []
    target_set = {}  # {i.name: set() for i in statistics}

    for target_statistics in statistics:
        target_set = set(
            get_top_centrality_nodes(graph, target_statistics, count=int(top_k_percent * graph[Stat.NODES])))
        # creating metrics and giving callable function to it (here - total fraction of nodes)
        # metrics.append(Metric(r'observed' + target_statistics.name, lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]))
        metrics.append(Metric(r'crawled_' + target_statistics.name,  # TODO rename crawled to observed
                              lambda crawler, t: len(t.intersection(crawler.crawled_set)) / len(t), t=target_set
                              ))

        # print(metrics[-1], target_set)
    if animated == True:
        ci = 1
        # AnimatedCrawlerRunner(graph,
        #                            crawlers,
        #                            metrics,
        #                            budget=10000,
        #                            step=10)
    else:
        ci = CrawlerRunner(graph,
                           crawlers,
                           metrics,
                           budget=0,
                           top_k_percent=top_k_percent,
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

    graphs = [
        # # # # 'livejournal-links', toooo large need all metrics
        'youtube-u-growth',  # with 3216075 nodes and  9369874 edges, davg= 5.83     no ecc
        'flixster',  # with 2523386 nodes and  7918801 edges, davg= 6.28     no ecc
        'soc-pokec-relationships',  # with 1632803 nodes and 22301964 edges, davg=27.32     no ecc
        'com-youtube',  # with 1134890 nodes and  2987624 edges, davg= 5.27     no cls, no ecc
        'munmun_twitter_social',  # with  465017 nodes and   833540 edges, davg= 3.58
        'petster-friendships-dog',  # with  426485 nodes and  8543321 edges, davg=40.06
        'petster-friendships-cat',  # with  148826 nodes and  5447464 edges, davg=73.21
        # 'digg-friends',           # with  261489 nodes and  1536577 edges, davg=11.75
        # 'douban',                 # with  154908 nodes and   327162 edges, davg= 4.22
        # 'facebook-wosn-links',    # with   63392 nodes and   816831 edges, davg=25.77
        # 'slashdot-threads',       # with   51083 nodes and   116573 edges, davg= 4.56
        # 'ego-gplus',              # with   23613 nodes and    39182 edges, davg= 3.32
        # 'mipt',                   # with   14313 nodes and   488852 edges, davg=68.31
        # 'petster-hamster',        # with    2000 nodes and    16098 edges, davg=16.10
    ]
    big_graphs = ['youtube-u-growth', 'flixster', 'soc-pokec-relationships', 'com-youtube', ]

    for graph_name in graphs[::-1]:
        if graph_name == 'mipt':
            g = GraphCollections.get(graph_name, 'other', giant_only=True)
        else:
            g = GraphCollections.get(graph_name, giant_only=True)
        print('Graph {} with {} nodes and {} edges, davg={:02.2f}'.format(graph_name, g.nodes(), g.edges(),
                                                                          2.0 * g.edges() / g.nodes()))
        if graph_name in big_graphs:
            big_graph_no_ecc = 'ECC'  # FIXME костыль пока не посчитан ECC  у больших
        else:
            big_graph_no_ecc = '----'

        # TODO: to check and download graph before multiprocessing
        iterations = multiprocessing.cpu_count() - 2
        for iter in range(int(8 // iterations)):
            start_time = time.time()
            processes = []
            # making parallel itarations. Number of processes
            for exp in range(iterations):
                logging.info('Running iteration {}/{}'.format(exp, iterations))
                # little multiprocessing magic, that calculates several iterations in parallel
                p = multiprocessing.Process(target=test_runner, args=(g,),
                                            kwargs={'animated': False,
                                                    'statistics': [s for s in Stat if 'DISTR' in s.name
                                                                   if big_graph_no_ecc not in s.name
                                                                   ],
                                                    'top_k_percent': 0.01,
                                                    # 'layout_pos':layout_pos,
                                                    'tqdm_info': 'core-' + str(exp + 1)
                                                    })
                p.start()

            p.join()
            print("Completed graph {} with {} nodes and {} edges".format(graph_name, g.nodes(), g.edges()),
                  "time elapsed: {:.2f}s, {}".format(time.time() - start_time, processes))

    # from experiments.merger import merge
    # merge(graph_name,
    #       show=True,
    #       filter_only='MOD', )
