from graph_io import MyGraph, GraphCollections
from runners.animated_runner import Metric, AnimatedCrawlerRunner
from statistics import get_top_centrality_nodes, Stat
from crawlers.advanced import CrawlerWithAnswer
from crawlers.basic import RandomWalkCrawler, RandomCrawler, BreadthFirstSearchCrawler, \
    MaximumObservedDegreeCrawler
from crawlers.multiseed import MultiCrawler


def test_multi_mod(graph: MyGraph):
    crawlers = [
        # MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=1),
        MaximumObservedDegreeCrawler(graph, batch=1),
        # MultiCrawler(graph, [
        #     MaximumObservedDegreeCrawler(graph, batch=1) for _ in range(10)
        # ])
    ]

    target_set = set(get_top_centrality_nodes(graph, Stat.DEGREE_DISTR, count=int(0.1 * graph[Stat.NODES])))
    metrics = [
        Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # Metric(r'$|V_{all} \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    acr = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=5000, step=100)
    acr.run()


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'

    g = GraphCollections.get(name)

    test_multi_mod(g)
