import numpy as np

from graph_io import GraphCollections
from runners.animated_runner import Metric, AnimatedCrawlerRunner
from statistics import get_top_centrality_nodes, Stat

from base.cgraph import MyGraph
from crawlers.cbasic import RandomWalkCrawler, RandomCrawler, BreadthFirstSearchCrawler, \
    MaximumObservedDegreeCrawler, SnowBallCrawler, PreferentialObservedDegreeCrawler

from crawlers.multiseed import MultiInstanceCrawler


def test_multi_mod(graph: MyGraph):
    # graph.random_node()
    init_nodes = graph.random_nodes(1000)
    crawlers = [
        # BreadthFirstSearchCrawler(graph, initial_seed=int(init_nodes[0])),
        # RandomWalkCrawler(graph, initial_seed=int(init_nodes[1])),
        # RandomCrawler(graph, initial_seed=int(init_nodes[2])),
        # SnowBallCrawler(graph, initial_seed=int(init_nodes[3])),
        MaximumObservedDegreeCrawler(graph, batch=1),
        MaximumObservedDegreeCrawler(graph, batch=10),
        MultiInstanceCrawler(graph, count=100, crawler_def=(MaximumObservedDegreeCrawler, {'batch': 1})),
        # MultiInstanceCrawler(graph, [
        #     PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=int(init_nodes[i+10])) for i in range(100)
        # ]),
        # MultiInstanceCrawler(graph, [
        #     PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=int(init_nodes[i+10])) for i in range(100)
        # ]),
    ]

    target_set = set(get_top_centrality_nodes(graph, Stat.DEGREE_DISTR, count=int(0.1 * graph[Stat.NODES])))
    metrics = [
        Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # Metric(r'$|V_{all} \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    acr = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=5000, step=100)
    acr.run(ylims=(0, 1))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'youtube-u-growth'
    # name = 'flixster'
    # name = 'flickr-links'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'

    g = GraphCollections.get(name)

    test_multi_mod(g)
