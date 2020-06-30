from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, definition_to_filename

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

        name = name if name else "%s %s %s %s" % (measure, part, centrality, top)
        super().__init__(name, callback, top=top, centrality=centrality, measure=measure, part=part)


def remap_iter(total=400):
    """ Remapping steps depending on used budget - on first iters step=1, on last it grows ~x^2
    NOTE: this sequence is used in history and should not be changed
    """
    step_budget = 0
    remap_iter_to_step = {}
    for i in range(total):  # for budget less than 100 mln nodes
        remap = int(max(1, step_budget / 20))
        remap_iter_to_step[step_budget] = remap
        step_budget += remap
    return remap_iter_to_step


class CrawlerRunner:
    """ Base class to run crawlers and measure metrics. Details in subclasses
    """

    def __init__(self, graph: MyGraph, crawlers, metrics, budget: int=-1, step: int=-1):
        """
        Setup configuration - crawler definitions and metric definitions.

        :param graph: graph to run
        :param crawlers: list of crawlers (only their definition will be used) or crawler
         definitions to run. Crawler definitions will be initialized when run() is called
        :param metrics: list of metrics (only their definition will be used) or metric definitions
         to compute at each step. Metric should be callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        self.graph = graph
        self.crawler_defs = []
        for x in crawlers:
            if isinstance(x, Crawler):
                # assert x._orig_graph == self.graph
                self.crawler_defs.append(x.definition)
            else:
                self.crawler_defs.append(x)

        self.metric_defs = []
        for x in metrics:
            if isinstance(x, Metric):
                self.metric_defs.append(x.definition)
            else:
                self.metric_defs.append(x)

        self.budget = min(budget, graph[Stat.NODES]) if budget != -1 else graph[Stat.NODES]
        assert step < self.budget
        self.step = max(1, step) if step != -1 else 1  # TODO use remap_iter

    def run(self):
        """ Run specified configurations on the given graph
        """
        # TODO put here iteration and metrics calculation
        raise NotImplementedError()


if __name__ == '__main__':
    i_step = remap_iter(100)
    n = 0
    for i, step in i_step.items():
        n += step
        print(i, n)
