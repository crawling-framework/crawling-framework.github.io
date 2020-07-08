from crawlers.cadvanced import DE_Crawler
from crawlers.cbasic import MaximumObservedDegreeCrawler, RandomWalkCrawler
from crawlers.community_based import MaximumObservedCommunityDegreeCrawler
from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections, netrepo_names
from models.models import LFR
from running.animated_runner import AnimatedCrawlerRunner
from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric
from statistics import Stat


def test_comm_based():
    # g = GraphCollections.get('petster-hamster')
    # g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('Pokec', not_load=True)
    # g = GraphCollections.get('Infectious')
    # g = GraphCollections.get('LFR(N=400,k=10,maxk=40,mu=0.1,t1=2,t2=2)/0', 'synthetic')
    # g = GraphCollections.get('LFR(N=4000,k=10,maxk=100,mu=0.1,t1=2,t2=2)/0', 'synthetic')
    g = LFR(nodes=4000, avg_deg=10, max_deg=100, mixing=0.1, t1=2, t2=2)
    print('MAX_WCC', g[Stat.MAX_WCC])

    p = 0.1
    # budget = int(0.005 * g.nodes())
    # s = int(budget / 2)

    crawler_defs = [
        (MaximumObservedDegreeCrawler, {'batch': 1, 'initial_seed': 1}),
        (MaximumObservedCommunityDegreeCrawler, {'initial_seed': 1}),
        (DE_Crawler, {'initial_budget': 10}),
        # (RandomWalkCrawler, {}),
        # (MultiInstanceCrawler, {'count': 10, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 1})}),
        # (MaximumObservedDegreeCrawler, {'batch': 10}),
        # (MaximumObservedDegreeCrawler, {'batch': 100}),
    ]
    metric_defs = [
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.DEGREE_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.PAGERANK_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.CLOSENESS_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'part': 'crawled', 'measure': 'F1', 'centrality': Stat.K_CORENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'crawled'}),
        # (TopCentralityMetric, {'top': 1, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'answer'}),

    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=-1)
    acr.run()

    n_instances = 6
    # # # Run missing iterations
    # for graph_name in netrepo_names:
    #     g = GraphCollections.get(graph_name)
    #     chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    #     chr.run_missing(n_instances, max_cpus=2, max_memory=20)
    #     print('\n\n')
    #
    # # Run merger
    # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
    # crm.draw_by_metric_crawler(x_lims=(0, 10000), x_normalize=False, scale=8, swap_coloring_scheme=False, draw_error=True)
    # crm.missing_instances()


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    test_comm_based()
