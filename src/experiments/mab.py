import logging

from crawlers.cbasic import MaximumObservedDegreeCrawler
from graph_io import GraphCollections
from crawlers.knn_ucb import KNN_UCB_Crawler
from running.animated_runner import AnimatedCrawlerRunner
from running.metrics_and_runner import TopCentralityMetric
from statistics import Stat


def test_knnucb():
    # g = GraphCollections.get('dolphins')
    # g = GraphCollections.get('Pokec')
    # g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('socfb-Bingham82')
    g = GraphCollections.get('soc-brightkite')

    p = 1
    # budget = int(0.005 * g.nodes())
    # s = int(budget / 2)

    crawler_defs = [
        (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        (MaximumObservedDegreeCrawler, {'initial_seed': 1}),
        # (KNN_UCB_Crawler, {}),
        # (MaximumObservedDegreeCrawler, {}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]
    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=1000)
    acr.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    test_knnucb()
