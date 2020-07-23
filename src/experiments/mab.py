import logging

from crawlers.cbasic import MaximumObservedDegreeCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler
from experiments.three_stage import social_names
from graph_io import GraphCollections, konect_names
from crawlers.knn_ucb import KNN_UCB_Crawler
from running.animated_runner import AnimatedCrawlerRunner
from running.metrics_and_runner import TopCentralityMetric
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
    p = 1

    crawler_defs = [
        # (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        # (MaximumObservedDegreeCrawler, {'initial_seed': 1}),
        (KNN_UCB_Crawler, {'alpha': a, 'k': k}) for a in [0.2, 0.5, 1.0, 5.0] for k in [3, 10, 30]
        # (MaximumObservedDegreeCrawler, {}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    n_instances = 8
    graph_names = social_names
    for graph_name in graph_names:
        g = GraphCollections.get(graph_name)
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs, budget=1000)
        chr.run_missing(n_instances, max_cpus=8, max_memory=30)

    # crm = ResultsMerger(graph_names, crawler_defs, metric_defs, n_instances)
    # crm.draw_by_crawler()


def run_original_knnucb():
    command = "PYTHONPATH=/home/misha/soft/bandits/:/home/misha/soft/mab_explorer/mab_explorer python3 mab_explorer/sampling.py data/CA-AstroPh.txt -s 0.05 -b 100 -m rn -e 1 -plot ./results"


def repeat_paper():
    #
    # git clone https://bitbucket.org/kau_mad/bandits.git
    # needs pip3 install pymc3
    # g = GraphCollections.get('ca-dblp-2012')
    g = GraphCollections.get('ca-AstroPh', 'netrepo')
    n = g[Stat.NODES]

    # Sample = 5% of nodes
    bfs = BreadthFirstSearchCrawler(g, initial_seed=2)
    bfs.crawl_budget(int(0.05 * n))
    observed_graph = bfs._observed_graph
    node_set = bfs.nodes_set

    p = 1
    crawler_defs = [
        (KNN_UCB_Crawler, {'n0': 20, 'n_features': 4, 'observed_graph': observed_graph, 'observed_set': node_set})
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=100)
    acr.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # test_knnucb()
    run_comparison()
    # repeat_paper()
