from utils import USE_CYTHON_CRAWLERS
import time
if USE_CYTHON_CRAWLERS:
    from base.cgraph import seed_random
    from crawlers.cadvanced import DE_Crawler
else:
    pass

from crawlers.cbasic import filename_to_definition
from runners.merger import CrawlerRunsMerger
from runners.metric_runner import TopCentralityMetric
from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections, konect_names, netrepo_names
from runners.animated_runner import Metric
from runners.history_runner import CrawlerHistoryRunner
from statistics import get_top_centrality_nodes, Stat
import multiprocessing


# def start_runner(graph, animated=False, statistics: list = None, top_k_percent=0.1, layout_pos=None, tqdm_info=''):
#     # initial_seed = random.sample([n.GetId() for n in graph.snap.Nodes()], 1)[0]
#     print([stat_name.name for stat_name in statistics])
#     seed_random(int(time.time() * 1e7 % 1e9))  # to randomize t_random in parallel processes
#
#     initial_seed = graph.random_nodes(1000)
#     print(initial_seed[:10])
#     ranges = [2, 3, 4, 5, 10, 30, 100, 1000]
#     crawlers = [  ## ForestFireCrawler(graph, initial_seed=initial_seed), # FIXME fix and rewrite
#                    # DepthFirstSearchCrawler(graph, initial_seed=initial_seed[0]),
#                    # SnowBallCrawler(graph, p=0.1, initial_seed=initial_seed[0]),
#                    # SnowBallCrawler(graph, p=0.25, initial_seed=initial_seed[0]),
#                    # SnowBallCrawler(graph, p=0.5, initial_seed=initial_seed[0]),
#                    # SnowBallCrawler(graph, p=0.75, initial_seed=initial_seed[0]),
#                    # SnowBallCrawler(graph, p=0.9, initial_seed=initial_seed[0]),
#                    # BreadthFirstSearchCrawler(graph, initial_seed=initial_seed[0]),  # is like take SBS with p=1
#                    #
#                    # RandomWalkCrawler(graph, initial_seed=initial_seed[0]),
#                    # RandomCrawler(graph, initial_seed=initial_seed[0]),
#                    #
#                    # MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
#                    # MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
#                    # MaximumObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
#                    # MaximumObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed[0]),
#                    # MaximumObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed[0]),
#                    #
#                    # PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
#                    # PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
#                    # PreferentialObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
#                    # PreferentialObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed[0]),
#                    # PreferentialObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed[0]),
#                    # DE_Crawler(graph, initial_seed=initial_seed[0]),
#                    MultiInstanceCrawler(graph, crawlers=[DE_Crawler(graph, initial_seed=initial_seed[i]) for i in range(range_i)]) for range_i in ranges
#                # ] + [
#         # MultiInstanceCrawler(graph, crawlers=[PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(range_i)]) for range_i in ranges
#     # ] + [
#     #     MultiInstanceCrawler(graph, crawlers=[BreadthFirstSearchCrawler(graph, initial_seed=initial_seed[i]) for i in range(range_i)]) for range_i in ranges
#     # ] + [
#     #     MultiInstanceCrawler(graph, crawlers=[MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[i]) for i in range(range_i)]) for range_i in ranges
#     ]
#     logging.info([c.name for c in crawlers])
#     metrics = []
#     target_set = {}  # {i.name: set() for i in statistics}
#
#     for target_statistics in statistics:
#         target_set = set(
#             get_top_centrality_nodes(graph, target_statistics, count=int(top_k_percent * graph[Stat.NODES])))
#         # creating metrics and giving callable function to it (here - total fraction of nodes)
#         # metrics.append(Metric(r'observed' + target_statistics.name, lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]))
#         metrics.append(Metric(r'crawled_' + target_statistics.name,  # TODO rename crawled to observed
#                               lambda crawler, t: len(t.intersection(crawler.crawled_set)) / len(t), t=target_set
#                               ))
#
#         # print(metrics[-1], target_set)
#     if animated == True:
#         ci = 1
#         # AnimatedCrawlerRunner(graph,
#         #                            crawlers,
#         #                            metrics,
#         #                            budget=10000,
#         #                            step=10)
#     else:
#         ci = TopKCrawlerRunner(graph,
#                                crawlers,
#                                metrics,
#                                budget=0,
#                                top_k_percent=top_k_percent,
#                                # step=ceil(10 ** (len(str(graph.nodes())) - 3)),
#                                tqdm_info=tqdm_info,
#                                # if 5*10^5 then step = 10**2,if 10^7 => step=10^4
#                                # batches_per_pic=10,
#                                # draw_mod='traversal', layout_pos=layout_pos,
#                                )  # if you want gifs, draw_mod='traversal'. else: 'metric'
#     ci.run()
#
#
# def big_run():
#
#     logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
#     logging.getLogger().setLevel(logging.INFO)
#     # graph_name = 'digg-friends'       # with 261489 nodes and 1536577 edges
#     # graph_name = 'douban'             # with 154908 nodes and  327162 edges
#     # graph_name = 'facebook-wosn-links'# with  63392 nodes and  816831 edges
#     # graph_name = 'slashdot-threads'   # with  51083 nodes and  116573 edges
#     # graph_name = 'ego-gplus'          # with  23613 nodes and   39182 edges
#     # graph_name = 'petster-hamster'    # with   2000 nodes and   16098 edges
#
#     graphs = [
#         # # # # 'livejournal-links', toooo large need all metrics
#         # 'soc-pokec-relationships',  # with 1632803 nodes and 22301964 edges, davg=27.32  1-2 all but POD,Multi
#         # 6x all but POD,Multi - cloud1.  F after 30+ hours
#
#         # 'youtube-u-growth',         # with 3216075 nodes and  9369874 edges, davg= 5.83     no ecc
#         # 'petster-friendships-dog',  # with  426485 nodes and  8543321 edges, davg=40.06  10/10
#
#         # 'flixster',                 # with 2523386 nodes and  7918801 edges, davg= 6.28  fails   no ecc
#         # 'com-youtube',              # with 1134890 nodes and  2987624 edges, davg= 5.27  2+ all
#
#         # 'munmun_twitter_social',    # with  465017 nodes and   833540 edges, davg= 3.58  10/10
#         # 'petster-friendships-cat',  # with  148826 nodes and  5447464 edges, davg=73.21 10/10
#         # 'digg-friends',           # with  261489 nodes and  1536577 edges, davg=11.75
#         # 'douban',                 # with  154908 nodes and   327162 edges, davg= 4.22
#         # 'facebook-wosn-links',    # with   63392 nodes and   816831 edges, davg=25.77
#         'slashdot-threads',       # with   51083 nodes and   116573 edges, davg= 4.56
#         # 'ego-gplus',              # with   23613 nodes and    39182 edges, davg= 3.32
#         # 'mipt',                   # with   14313 nodes and   488852 edges, davg=68.31
#         # 'petster-hamster',        # with    2000 nodes and    16098 edges, davg=16.10
#         # 4x Multi DE - cloud1
#
#
#         # # netrepo from Guidelines
#         # #
#         # 'socfb-Bingham82',     # N=10001,  E=362892,   d_avg=72.57
#         # 'soc-brightkite',      # N=56739,  E=212945,   d_avg=7.51
#         # 'ca-citeseer',         # N=227320, E=814134,   d_avg=7.16
#         # 'ca-dblp-2010',        # N=226413, E=716460,   d_avg=6.33
#         # 'ca-dblp-2012',        # N=317080, E=1049866,  d_avg=6.62
#         # 'web-arabic-2005',     # N=163598, E=1747269,  d_avg=21.36
#         # 'web-italycnr-2000',   # N=325557, E=2738969,  d_avg=16.83
#         # 'socfb-Penn94',        # N=41536,  E=1362220,  d_avg=65.59
#         # 'ca-MathSciNet',       # N=332689, E=820644,   d_avg=4.93
#         # 'socfb-wosn-friends',  # N=63392,  E=816886,   d_avg=25.77
#         # 'tech-RL-caida',       # N=190914, E=607610,   d_avg=6.37
#         # 'rec-github',          # N=121331, E=439642,   d_avg=7.25
#         # 'web-sk-2005',         # N=121422, E=334419,   d_avg=5.51
#         # 'tech-p2p-gnutella',   # N=62561,  E=147878,   d_avg=4.73
#         # 'sc-pkustk13',         # N=94893,  E=3260967,  d_avg=68.73
#         # 'sc-pwtk',             # N=217883, E=5653217,  d_avg=51.89
#         # 'sc-shipsec1',         # N=139995, E=1705212,  d_avg=24.36
#         # 'sc-shipsec5',         # N=178573, E=2197367,  d_avg=24.61
#         # 'rec-amazon',          # N=91813,  E=125704,   d_avg=2.74
#         # 'soc-slashdot',        # N=70068,  E=358647,   d_avg=10.24
#         # 'soc-BlogCatalog',     # N=88784,  E=2093195,  d_avg=47.15
#         # 'web-uk-2005',         # N=129632, E=11744049, d_avg=181.19
#         # 'soc-themarker',       #?N=69317,  E=1644794,  d_avg=47.46
#         # 8x Multi DE - cloud2
#
#     ]
#
#     for graph_name in graphs:
#         msg = "Did not finish"
#
#         g = GraphCollections.get(graph_name, giant_only=True)
#         iterations = 8  # multiprocessing.cpu_count() - 2
#         max_threads = 4
#
#         for iter in range(iterations // max_threads):
#             start_time = time.time()
#             processes = []
#             # making parallel iterations. Number of processes
#             for thread in range(max_threads):
#                 logging.info('Running iteration %s/%s' % (thread+1, max_threads))
#                 # little multiprocessing magic, that calculates several iterations in parallel
#                 p = multiprocessing.Process(target=start_runner, args=(g,),
#                                             kwargs={'animated': False,
#                                                     # 'statistics': [s for s in Stat if 'DISTR' in s.name],
#                                                     'statistics': [Stat.ECCENTRICITY_DISTR],
#                                                     'top_k_percent': 0.01,
#                                                     'tqdm_info': 'core-' + str(thread + 1)
#                                                     })
#                 p.start()
#
#             p.join()
#
#             msg = "Completed graph {} with {} nodes and {} edges. time elapsed: {:.2f}s, {}". \
#                 format(graph_name, g.nodes(), g.edges(), time.time() - start_time, processes)
#             # except Exception as e:
#             #     msg = "Failed graph %s after %.2fs with error %s" % (graph_name, time.time() - start_time, e)
#
#             print(msg)
#
#         # send to my vk
#         import os
#         from utils import rel_dir
#         bot_path = os.path.join(rel_dir, "src", "experiments", "vk_signal.py")
#         command = "python3 %s -m '%s'" % (bot_path, msg)
#         exit_code = os.system(command)
#
#     # from experiments.merger import merge
#     # merge(graph_name,
#     #       show=True,
#     #       filter_only='MOD', )


def run_missing(max_cpus: int=multiprocessing.cpu_count(), max_memory: float=6):
    """
    :param max_cpus: max number of CPUs to use for computation, all by default
    :param max_memory: max Mbytes of operative memory to use for computation, 6Gb by default
    :return:
    """
    graphs = [
        # 'socfb-Bingham82',
        # 'ego-gplus',
        # 'web-sk-2005',
        # 'digg-friends',
        # 'petster-hamster',
        # 'ego-gplus',
        # g for g in netrepo_names
    # ] + [
        g for g in konect_names
    ]
    p = 0.01
    crawler_defs = [
        # (ThreeStageCrawler, {'s': 699, 'n': 1398, 'p': p}),
        # (ThreeStageMODCrawler, {'s': 699, 'n': 1398, 'p': p, 'b': 10}),
        filename_to_definition('RW()'),
        filename_to_definition('RC()'),
        filename_to_definition('BFS()'),
        filename_to_definition('DFS()'),
        filename_to_definition('SBS(p=0.1)'),
        filename_to_definition('SBS(p=0.25)'),
        filename_to_definition('SBS(p=0.75)'),
        filename_to_definition('SBS(p=0.89)'),
        filename_to_definition('SBS(p=0.5)'),
        filename_to_definition('MOD(batch=10000)'),
        filename_to_definition('MOD(batch=1000)'),
        filename_to_definition('MOD(batch=100)'),
        filename_to_definition('MOD(batch=10)'),
        filename_to_definition('MOD(batch=1)'),
        filename_to_definition('POD(batch=10000)'),
        filename_to_definition('POD(batch=1000)'),
        filename_to_definition('POD(batch=100)'),
        filename_to_definition('POD(batch=10)'),
        filename_to_definition('POD(batch=1)'),
        filename_to_definition('DE(initial_budget=10)'),
        filename_to_definition('MultiInstance(count=2,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=2,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=2,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=2,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=3,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=3,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=3,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=3,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=4,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=4,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=4,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=4,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=5,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=5,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=5,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=5,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=10,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=10,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=10,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=10,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=30,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=30,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=30,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=30,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=100,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=100,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=100,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=100,crawler_def=DE(initial_budget=10))'),
        filename_to_definition('MultiInstance(count=1000,crawler_def=BFS())'),
        filename_to_definition('MultiInstance(count=1000,crawler_def=MOD(batch=1))'),
        filename_to_definition('MultiInstance(count=1000,crawler_def=POD(batch=1))'),
        filename_to_definition('MultiInstance(count=1000,crawler_def=DE(initial_budget=10))'),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]

    # Get missing combinations
    crm = CrawlerRunsMerger(graphs, crawler_defs, metric_defs, n_instances=6)
    missing = crm.missing_instances()
    import json
    print(json.dumps(missing, indent=2))

    for graph_name, cmi in missing.items():
        crawler_defs = [filename_to_definition(c) for c in cmi.keys()]
        max_count = 0
        for crawler_name, mi in cmi.items():
            max_count = max(max_count, max(mi.values()))

        print("Will run x%s missing crawlers for %s graph: %s" % (max_count, graph_name, list(cmi.keys())))
        g = GraphCollections.get(graph_name)

        cr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
        # [cr.run() for _ in range(max_count)]

        # Parallel run with adaptive number of CPUs
        memory = (0.25 * g['NODES']/1000 + 2.5)/1024 * len(crawler_defs)  # Gbytes of operative memory per instance
        max_cpus = min(max_cpus, max_memory // memory)
        while max_count > 0:
            num = min(max_cpus, max_count)
            max_count -= num
            msg = cr.run_parallel(num)

            # send to my vk
            import os
            from utils import rel_dir
            bot_path = os.path.join(rel_dir, "src", "experiments", "vk_signal.py")
            os.system("python3 %s -m '%s'" % (bot_path, msg))
        print('\n\n')


def big_run(max_cpus: int=multiprocessing.cpu_count(), max_memory: float=6):
    """
    :param max_cpus: max number of CPUs to use for computation, all by default
    :param max_memory: max Mbytes of operative memory to use for computation, 6Gb by default
    :return:
    """
    graphs = [
        'socfb-Bingham82',
        # 'web-sk-2005',
        # 'digg-friends',
        'petster-hamster',
        'ego-gplus',
        'slashdot-threads',
        'facebook-wosn-links',
    ]
    p = 0.01
    crawler_defs = [
        # filename_to_definition('RW()'),
        # filename_to_definition('RC()'),
        # filename_to_definition('BFS()'),
        # filename_to_definition('DFS()'),
        # filename_to_definition('SBS(p=0.1)'),
        # filename_to_definition('SBS(p=0.25)'),
        # filename_to_definition('SBS(p=0.75)'),
        # filename_to_definition('SBS(p=0.89)'),
        # filename_to_definition('SBS(p=0.5)'),
        # filename_to_definition('MOD(batch=10000)'),
        # filename_to_definition('MOD(batch=1000)'),
        # filename_to_definition('MOD(batch=100)'),
        # filename_to_definition('MOD(batch=10)'),
        # filename_to_definition('MOD(batch=1)'),
        # filename_to_definition('POD(batch=10000)'),
        # filename_to_definition('POD(batch=1000)'),
        # filename_to_definition('POD(batch=100)'),
        # filename_to_definition('POD(batch=10)'),
        # filename_to_definition('POD(batch=1)'),
        (DE_Crawler, {}),
        (DE_Crawler, {'initial_budget': 10}),
        # (DE_Crawler, {'initial_budget': 30}),
        # (DE_Crawler, {'initial_budget': 100}),
        # filename_to_definition('MultiInstance(count=2,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=DE(initial_budget=10))'),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]

    # Get missing combinations
    crm = CrawlerRunsMerger(graphs, crawler_defs, metric_defs, n_instances=6)
    missing = crm.missing_instances()
    import json
    print(json.dumps(missing, indent=2))

    for graph_name, cmi in missing.items():
        crawler_defs = [filename_to_definition(c) for c in cmi.keys()]
        max_count = 0
        for crawler_name, mi in cmi.items():
            max_count = max(max_count, max(mi.values()))

        print("Will run x%s missing crawlers for %s graph: %s" % (max_count, graph_name, list(cmi.keys())))
        g = GraphCollections.get(graph_name)

        cr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
        # [cr.run() for _ in range(max_count)]

        # Parallel run with adaptive number of CPUs
        memory = (0.25 * g['NODES']/1000 + 2.5)/1024 * len(crawler_defs)  # Gbytes of operative memory per instance
        max_cpus = min(max_cpus, max_memory // memory)
        while max_count > 0:
            num = min(max_cpus, max_count)
            max_count -= num
            msg = cr.run_parallel(num)

            # send to my vk
            import os
            from utils import rel_dir
            bot_path = os.path.join(rel_dir, "src", "experiments", "vk_signal.py")
            os.system("python3 %s -m '%s'" % (bot_path, msg))
        print('\n\n')

    # crm.draw_by_crawler()
    crm.draw_winners('AUCC')


def cloud_manager():
    import subprocess, sys

    cloud1 = 'ubuntu@83.149.198.220'
    cloud2 = 'ubuntu@83.149.198.231'

    local_dir = '/home/misha/workspace/crawling'
    remote_dir = '/home/ubuntu/workspace/crawling'
    ssh_key = '~/.ssh/drobyshevsky_home_key.pem'

    # Copy stats to remote

    names = ['ca-citeseer', 'ca-dblp-2010', 'rec-amazon', 'rec-github', 'sc-pkustk13', 'soc-BlogCatalog',
             'soc-brightkite', 'soc-slashdot', 'soc-themarker', 'socfb-Bingham82', 'socfb-OR',
             'socfb-Penn94', 'socfb-wosn-friends', 'tech-RL-caida', 'tech-p2p-gnutella', 'web-arabic-2005']
    cloud = cloud1
    collection = 'konect'
    for name in ['slashdot-threads']:
    # for name in ['web-uk-2005', 'web-italycnr-2000', 'ca-dblp-2012', 'sc-pwtk']:
        # if not os.path.exists('%s/data/%s/%s.ij_stats/EccDistr' % (local_dir, collection, name)):
        #     continue

        # loc2rem_copy_command = 'scp -i %s -r %s/data/%s/%s.ij_stats/ %s:%s/data/%s/' % (
        #     ssh_key, local_dir, collection, name, cloud, remote_dir, collection)
        # loc2rem_copy_command = 'scp -i %s -r %s/results/k=0.01/%s/ %s:%s/results/k=0.01/' % (
        #     ssh_key, local_dir, name, cloud, remote_dir, )

        # rem2loc_copy_command = 'scp -i %s -r %s:%s/data/%s/%s.ij_stats/EccDistr %s/data/%s/%s.ij_stats/' % (
        #     ssh_key, cloud, remote_dir, collection, name, local_dir, collection, name)
        rem2loc_copy_command = 'scp -i %s -r %s:%s/results/k=0.01/ %s/%s/' % (
            ssh_key, cloud, remote_dir, local_dir, cloud)

        command = rem2loc_copy_command

        logging.info("Executing command: '%s' ..." % command)
        retcode = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
        if retcode != 0:
            logging.error("returned code = %s" % retcode)
            raise RuntimeError("unsuccessful: '%s'" % command)
        else:
            logging.info("OK")


    # # Run script
    # cloud = cloud2
    # connect_command = 'ssh %s -i %s' % (cloud2, ssh_key)
    # logging.info("Executing command: '%s' ..." % connect_command)
    # retcode = subprocess.Popen(connect_command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
    #
    # run_command = 'cd workspace/crawling/; PYTHONPATH=/home/ubuntu/workspace/crawling/src python3 src/experiments/crawlers_runs.py'
    # logging.info("Executing command: '%s' ..." % run_command)
    # retcode = subprocess.Popen(run_command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()


def prepare_graphs():
    for name in konect_names + netrepo_names + ['mipt']:
        g = GraphCollections.get(name, giant_only=True, not_load=True)
        print("%s N=%s, E=%s, d_avg=%.2f" % (name, g['NODES'], g['EDGES'], g[Stat.AVG_DEGREE]))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # # Write logs to file
    # from datetime import datetime
    # fh_info = logging.FileHandler("%s.log" % (datetime.now()))
    # fh_info.setLevel(logging.DEBUG)
    # fh_info.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s:%(message)s'))
    # logger = logging.getLogger('__main__')
    # logger.addHandler(fh_info)

    import sys
    # sys.stdout = open('logs', 'w')
    # sys.stderr = open('logs', 'w')

    # run_missing(max_cpus=6, max_memory=24)
    big_run()
    # prepare_graphs()
    # cloud_manager()
