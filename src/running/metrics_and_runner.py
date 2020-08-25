from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, definition_to_filename

from graph_stats import Stat, get_top_centrality_nodes


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
        assert _class != Metric, "Create a subclass to define your own Metric"
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
        """
        Measure crawling result with respect to top fraction of nodes by a specified centrality.

        :param graph: original graph
        :param top: fraction of target nodes (from 0 to 1)
        :param centrality: node centrality to get top (see Stat)
        :param measure: 'Pr' (precision), 'Re' (recall), or 'F1' (F1-score)
        :param part: 'crawled', 'observed', 'nodes' (observed+crawled), 'answer' (crawler must
         support getting an answer, e.g. extend CrawlerWithAnswer)
        :param name: name for plotting
        """
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


def exponential_batch_generator(budget: int = 1e9):
    """ Generator for exponentially growing batches. Yields batches until their sum < budget; the
    last one fills it up to budget exactly.
    NOTE: magic constant 20 is used in history and should not be changed.
    """
    total = 0
    while True:
        batch = max(1, int(total / 20))
        total += batch
        if total < budget:
            yield batch
        else:
            yield budget - total + batch
            break
        # print(total, batch)


def uniform_batch_generator(budget: int, step: int):
    """ Generator for uniform batches. Yields batches until their sum < budget; the last one
    fills it up to budget exactly"""
    assert 1 <= step <= budget
    batch = step
    total = 0
    while True:
        total += batch
        if total < budget:
            yield batch
        else:
            yield budget - total + batch
            break
        # print(total, batch)


class CrawlerRunner:
    """ Base class to run crawlers and measure metrics. Details in subclasses
    """

    def __init__(self, graph: MyGraph, crawler_defs, metric_defs, budget: int = -1, step: int = -1):
        """
        Setup configuration for the graph: crawler definitions, metric definitions, budget, and step.

        :param graph: graph to run
        :param crawler_defs: list of crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metric_defs: list of metric definitions to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps, by default exponential step
        :return:
        """
        self.graph = graph
        self.crawler_defs = crawler_defs
        self.metric_defs = metric_defs

        # Initialize step sequence
        self.budget = min(budget, graph[Stat.NODES]) if budget != -1 else graph[Stat.NODES]
        self.batch_generator_getter = lambda: exponential_batch_generator(self.budget) \
            if step == -1 else uniform_batch_generator(self.budget, step)

    def _init_runner(self, same_initial_seed=False):
        """ Initialize crawlers, metrics, and batch generator.
        """
        # TODO implement same_initial_seed
        crawlers = [Crawler.from_definition(self.graph, d) for d in self.crawler_defs]
        metrics = [Metric.from_definition(self.graph, d) for d in self.metric_defs]

        return crawlers, metrics, self.batch_generator_getter()

    def run(self):
        """ Run specified configurations on the given graph
        """
        raise NotImplementedError("defined in subclasses")


if __name__ == '__main__':
    t = 0
    # for b in exponential_batch_generator(9949):
    for b in uniform_batch_generator(9949, 9949):
        t += b
        print(t, b)
