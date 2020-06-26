from matplotlib import pyplot as plt

from utils import USE_CYTHON_CRAWLERS

if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
    from crawlers.cbasic import CCrawler as Crawler, MaximumObservedDegreeCrawler, \
    PreferentialObservedDegreeCrawler, definition_to_filename
else:
    from base.graph import MyGraph
    from crawlers.basic import Crawler, PreferentialObservedDegreeCrawler, MaximumObservedDegreeCrawler

from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes


class Metric:
    short = 'Metric'

    def __init__(self, name, callback, **kwargs):
        self._callback = callback
        self._kwargs = kwargs
        self._definition = type(self), kwargs
        self.name = name if name else definition_to_filename(self._definition)

    @staticmethod
    def from_definition(graph: MyGraph, definition):  # -> Metric:
        """ Build a Metric instance from its definition """
        _class, kwargs = definition
        return _class(graph, **kwargs)

    @property
    def definition(self):
        return self._definition

    def __call__(self, crawler: Crawler):
        return self._callback(crawler, **self._kwargs)


centrality_by_name = {stat.short: stat for stat in Stat if 'DISTR' in stat.name}


class TopCentralityMetric(Metric):
    short = 'TopK'

    def __init__(self, graph: MyGraph, top: float, centrality: str, measure='F1', part='crawled', name=None):
        # assert 'Distr' in centrality
        assert part in ['crawled', 'observed', 'nodes', 'answer']
        assert measure in ['Pr', 'Re', 'F1']

        target_set = set(get_top_centrality_nodes(graph, centrality_by_name[centrality], count=int(top*graph[Stat.NODES])))
        target_set_size = len(target_set)

        get_part = {
            'crawled': lambda crawler: crawler.crawled_set,
            'observed': lambda crawler: crawler.observed_set,
            'nodes': lambda crawler: crawler.nodes_set,
            'answer': lambda crawler: crawler.answer,
        }[part]

        callback = {
            'Pr': lambda crawler, **kwargs:    len(get_part(crawler).intersection(target_set)) / len(get_part(crawler)),
            'Re': lambda crawler, **kwargs:    len(get_part(crawler).intersection(target_set)) / target_set_size,
            'F1': lambda crawler, **kwargs:  2*len(get_part(crawler).intersection(target_set)) / (target_set_size + len(get_part(crawler))),
        }[measure]

        name = name if name else "%s_%s_%s_%s" % (measure, part, centrality, top)
        super().__init__(name, callback, top=top, centrality=centrality, measure=measure, part=part)

    # @staticmethod
    # def to_string(top: float, centrality: Stat, measure: str, part: str):
    #     return "%s_%s_%s_%s" % (measure, part, centrality, top)


class CrawlerRunner:
    """ Base class to runs crawlers and measure metrics. Details in subclasses
    """

    def __init__(self, graph: MyGraph, crawlers, metrics, budget: int=-1, step: int=1):
        """
        :param graph: graph to run
        :param crawlers: list of crawlers or crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metrics: list of metrics or metric definitions to compute at each step. Metric should
         be callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        self.graph = graph
        self.crawlers = []
        self.crawler_defs = []
        for x in crawlers:
            if isinstance(x, Crawler):
                assert x._orig_graph == self.graph
                self.crawlers.append(x)
            else:
                self.crawler_defs.append(x)

        self.metrics = []
        self.metric_defs = []
        for x in metrics:
            if isinstance(x, Metric):
                # XXX Hope metric was created for the same graph
                self.metrics.append(x)
            else:
                self.metric_defs.append(x)

        self.budget = min(budget, graph[Stat.NODES]) if budget > 0 else graph[Stat.NODES]
        assert step < self.budget
        self.step = step

    def run(self):
        """ Run specified configurations on the given graph
        """
        raise NotImplementedError()
