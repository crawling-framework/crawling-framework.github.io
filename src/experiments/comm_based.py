from crawlers.cadvanced import DE_Crawler
from crawlers.cbasic import MaximumObservedDegreeCrawler
from crawlers.community_based import CommunityMODCrawler
from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections
from running.animated_runner import AnimatedCrawlerRunner
from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric
from statistics import Stat


def test_comm_based():
    # g = GraphCollections.get('petster-hamster')
    g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('Infectious')

    p = 0.1
    budget = int(0.005 * g.nodes())
    s = int(budget / 2)

    crawler_defs = [
        (MaximumObservedDegreeCrawler, {'batch': 1}),
        (CommunityMODCrawler, {'initial_seed': 1}),
        (DE_Crawler, {'initial_budget': 10, 'initial_seed': 1}),
        (MultiInstanceCrawler, {'count': 10, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 1})}),
        # (MaximumObservedDegreeCrawler, {'batch': 10}),
        # (MaximumObservedDegreeCrawler, {'batch': 100}),
    ]
    metric_defs = [
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.DEGREE_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.PAGERANK_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.BETWEENNESS_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.ECCENTRICITY_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.CLOSENESS_DISTR.short).definition,
        # TopCentralityMetric(g, top=p, part='answer', measure='F1', centrality=Stat.K_CORENESS_DISTR.short).definition,
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'crawled'}),
        # (TopCentralityMetric, {'top': 1, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'answer'}),

    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=5000)
    acr.run()

    # n_instances = 6
    # # # Run missing iterations
    # chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    # chr.run_missing(n_instances)
    # #
    # # Run merger
    # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
    # crm.draw_by_metric_crawler(x_lims=(0, budget), x_normalize=False, scale=8, swap_coloring_scheme=True, draw_error=False)
    # crm.missing_instances()


if __name__ == '__main__':
    test_comm_based()
