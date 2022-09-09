from base.cgraph import MyGraph
from crawlers.cbasic import Crawler
from running.metrics import Metric


class CrawlerRunner:
    """
    Base class to run configurations of several crawlers and metrics. Details in subclasses.
    """

    def __init__(self, graph: MyGraph, crawler_decls, metric_decls, budget: int = -1, step: int = -1):
        """
        Setup configuration for the graph: crawler declarations, metric declarations, budget, and step.

        :param graph: graph to run
        :param crawler_decls: list of crawler declarations to run. Crawler declarations will be
         initialized when run() is called
        :param metric_decls: list of metric declarations to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps, by default exponential step
        :return:
        """
        self.graph = graph
        self.crawler_decls = crawler_decls
        self.metric_decls = metric_decls

        # Initialize step sequence
        self.budget = min(budget, graph.nodes()) if budget != -1 else graph.nodes()
        self.batch_generator_getter = lambda: exponential_batch_generator(self.budget) \
            if step == -1 else uniform_batch_generator(self.budget, step)

    def _init_runner(self):
        """ Initialize crawlers, metrics, and batch generator.

        :return: triple with defined objects: (crawlers, metrics, batch generator)
        """
        # TODO implement same_initial_seed
        crawlers = [Crawler.from_declaration(d, graph=self.graph) for d in self.crawler_decls]
        metrics = [Metric.from_declaration(d, graph=self.graph) for d in self.metric_decls]

        return crawlers, metrics, self.batch_generator_getter()

    def run(self):
        """ Run specified configurations on the given graph
        """
        raise NotImplementedError("defined in subclasses")


def exponential_batch_generator(budget: int = 1e9, exponent=0.05):
    """ Generator for exponentially growing batches.
    Yields batches until their sum < budget; the last one fills it up to budget exactly.

    NOTE: magic constant 20 is used in history and should not be changed.
    """
    total = 0
    while True:
        batch = max(1, int(total * exponent))
        total += batch
        if total < budget:
            yield batch
        else:
            yield budget - total + batch
            break


def uniform_batch_generator(budget: int, step: int):
    """ Generator for uniform batches.
    Yields batches until their sum < budget; the last one fills it up to budget exactly.
    """
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


def test_batch_generator():
    t = 0
    # for b in exponential_batch_generator(9949):
    for b in uniform_batch_generator(9949, step=100):
        t += b
        print(t, b)


if __name__ == '__main__':
    test_batch_generator()
