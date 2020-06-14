import glob
import logging
import os

from utils import USE_CYTHON_CRAWLERS, RESULT_DIR
from graph_io import GraphCollections
from statistics import Stat

if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from base.cadvanced import AvrachenkovCrawler, CrawlerWithAnswer, ThreeStageCrawler, \
    ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs, DE_Crawler
    from base.cbasic import CCrawler as Crawler, CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler, \
        BreadthFirstSearchCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler
    from base.cmultiseed import MultiCrawler
    from base.cmetrics import CMetric as Metric, CoverageTopCentralityMetric
else:
    from base.graph import MyGraph
    from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
        ThreeStageMODCrawler
    from crawlers.basic import CrawlerException, MaximumObservedDegreeCrawler, \
        BreadthFirstSearchCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler
    from crawlers.multiseed import MultiCrawler


class CrawlerRunsMerger:
    def __init__(self, graphs, crawlers, metrics, n_instances=1):
        self.graphs = graphs
        self.crawlers = crawlers
        self.metrics = metrics
        self.read()

    @staticmethod
    def to_path(graph: MyGraph, crawler: Crawler, metric: Metric):
        budget = graph[Stat.NODES]  # TODO as param?
        path = os.path.join(RESULT_DIR, graph.name, budget, crawler.name, metric.name)
        return path

    @staticmethod
    def names_to_path(graph_name: str, crawler_name: str, metric_name: str):
        # Old path
        budget = "budget=%s" % GraphCollections.get(graph_name, not_load=True)[Stat.NODES]
        top = "k=%s" % metric_name.split('_')[-1]
        path = os.path.join(RESULT_DIR, top, graph_name, budget, crawler_name, '?')

        # New path
        # path = os.path.join(RESULT_DIR, graph_name, crawler_name, metric_name)  # FIXME use new path
        return path

    def read(self):
        for g in self.graphs:
            for c in self.crawlers:
                for m in self.metrics:
                    p = CrawlerRunsMerger.names_to_path(g, c, m)
                    print(p)
        # RESULT_DIR_BUDG = RESULT_DIR + '/k={:1.2f}'.format(top_k_percent)
        #
        # budget_path = glob(os.path.join(RESULT_DIR_BUDG, graph_name,
        #                                 'budget=*'))  # TODO take first founded budget, step


def test_merger():

    g = GraphCollections.get('dolphins', not_load=False)
    print(g[Stat.NODES], g[Stat.ASSORTATIVITY])

    graphs = [
        g.name,
    ]
    crawlers = [
        'RW_',
        'BFS',
    ]
    metrics = [
        CoverageTopCentralityMetric.to_string(top=0.01, centrality=Stat.DEGREE_DISTR, part='crawled')
    ]
    crm = CrawlerRunsMerger(graphs, crawlers, metrics, n_instances=1)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    test_merger()
