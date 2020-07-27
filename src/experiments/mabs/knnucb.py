import logging

from crawlers.cadvanced import DE_Crawler
from crawlers.cbasic import MaximumObservedDegreeCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler
from experiments.three_stage import social_names
from graph_io import GraphCollections, konect_names
from crawlers.knn_ucb import KNN_UCB_Crawler
from running.animated_runner import AnimatedCrawlerRunner
from running.metrics_and_runner import TopCentralityMetric, Metric
from running.merger import ResultsMerger
from running.history_runner import CrawlerHistoryRunner
from statistics import Stat


def test_knnucb():
    # g = GraphCollections.get('dolphins')
    # g = GraphCollections.get('Pokec')
    g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('socfb-Bingham82')
    # g = GraphCollections.get('soc-brightkite')

    p = 1
    # budget = int(0.005 * g.nodes())
    # s = int(budget / 2)

    crawler_defs = [
        # (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        # (MaximumObservedDegreeCrawler, {'initial_seed': 1}),
        (KNN_UCB_Crawler, {'initial_seed': 2, 'n0': 0}),
        # (MaximumObservedDegreeCrawler, {'initial_seed': 2}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=1000, step=1000)
    acr.run()


def run_comparison():
    p = 0.01

    crawler_defs = [
        # (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        # (MaximumObservedDegreeCrawler, {'name': 'MOD'}),
        (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 30, 'n_features': 1, 'n0': 0}),
        # (DE_Crawler, {'name': 'DE'}),
    ] + [
        # (KNN_UCB_Crawler, {'alpha': a, 'k': k}) for a in [0.2, 0.5, 1.0, 5.0] for k in [3, 10, 30]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 10, 'n0': 0})
        # (KNN_UCB_Crawler, {'alpha': a, 'k': 30}) for a in [0.2, 0.5, 1.0, 5.0]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 30, 'n_features': f, 'n0': 30}) for f in [1, 2, 3, 4]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 30, 'n_features': 1, 'n0': n0}) for n0 in [0]
        # (MaximumObservedDegreeCrawler, {}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'nodes'}),
    ]

    n_instances = 8
    graph_names = social_names
    for graph_name in graph_names:
        g = GraphCollections.get(graph_name)
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs, budget=1000)
        chr.run_missing(n_instances, max_cpus=8, max_memory=30)

    # crm = ResultsMerger(graph_names, crawler_defs, metric_defs, n_instances)
    # crm.draw_by_crawler(x_normalize=False, draw_error=False, scale=3)
    # crm.draw_aucc()
    # crm.draw_winners('AUCC', scale=3)
    # crm.draw_winners('wAUCC', scale=3)


def run_original_knnucb():
    # git clone https://bitbucket.org/kau_mad/bandits.git
    # needs pip3 install pymc3
    command = "PYTHONPATH=/home/misha/soft/bandits/:/home/misha/soft/mab_explorer/mab_explorer python3 mab_explorer/sampling.py data/CA-AstroPh.txt -s 0.05 -b 100 -m rn -e 1 -plot ./results"


def reproduce_paper():
    """ Try to reproduce experiments from "A multi-armed bandit approach for exploring partially observed networks"
    https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    # g = GraphCollections.get('ca-dblp-2012')
    g = GraphCollections.get('ca-AstroPh', 'netrepo')
    n = g[Stat.NODES]

    # Sample = 5% of nodes
    bfs = BreadthFirstSearchCrawler(g, initial_seed=1)
    bfs.crawl_budget(int(0.05 * n))
    observed_graph = bfs._observed_graph
    node_set = bfs.nodes_set
    initial_size = len(node_set)
    print(initial_size)

    # Metric which counts newly observed nodes
    class AMetric(Metric):
        def __init__(self, graph):
            super().__init__(name='new obs nodes', callback=lambda crawler: len(crawler.nodes_set) - initial_size)

    crawler_defs = [
        (MaximumObservedDegreeCrawler, {'observed_graph': observed_graph.copy(), 'observed_set': set(node_set)}),
        (KNN_UCB_Crawler, {'n0': 20, 'n_features': 4, 'observed_graph': observed_graph.copy(), 'observed_set': set(node_set)}),
    ]
    metric_defs = [
        (AMetric, {}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=1000, step=20)
    acr.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # test_knnucb()
    run_comparison()
    # reproduce_paper()
