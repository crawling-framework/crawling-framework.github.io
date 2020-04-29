from matplotlib import pyplot as plt

from centralities import get_top_centrality_nodes
from crawlers.basic import MaximumObservedDegreeCrawler, RandomCrawler, Crawler, \
    BreadthFirstSearchCrawler, RandomWalkCrawler
from crawlers.multiseed import MultiCrawler

from graph_io import MyGraph, GraphCollections
from statistics import Stat


class Metric:
    def __init__(self, name, callback):
        self.name = name
        self._callback = callback

    def __call__(self, crawler: Crawler):
        return self._callback(crawler)


class AnimatedCrawlerRunner:
    def __init__(self, graph: MyGraph, crawlers, metrics, budget=-1, step=1):
        """
        :param graph:
        :param crawlers: list of crawlers to run
        :param metrics: list of metrics to compute at each step. Metric should be callable function
         crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        self.graph = graph
        g = self.graph.snap
        for crawler in crawlers:
            assert crawler.orig_graph == graph
        self.crawlers = crawlers
        self.metrics = metrics
        self.budget = budget if budget > 0 else g.GetNodes()
        assert step < self.budget
        self.step = step

        self.nrows = 1
        self.ncols = 1
        scale = 5
        # if len(self.crawlers) > 1:
        #     self.nrows = 2
        # self.ncols = ceil(len(self.crawlers) / self.nrows)

        fig = plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
                         figsize=(1 + scale*self.ncols, scale*self.nrows))

    def run(self):
        linestyles = ['-', '--', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']

        step_seq = []
        crawler_metric_seq = dict([(c, dict([(m, []) for m in self.metrics])) for c in self.crawlers])

        i = 0
        while i < self.budget:
            batch = min(self.step, self.budget-i)
            i += batch
            step_seq.append(i)

            plt.cla()
            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(budget=batch)

                for m, metric in enumerate(self.metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))
                    plt.plot(step_seq, metric_seq, marker='.',
                             linestyle=linestyles[m%len(linestyles)],
                             color=colors[c%len(colors)],
                             label=r'%s, %s' % (crawler.name, metric.name))

            plt.legend()
            plt.ylim((0, 1))
            plt.xlabel('iteration, n')
            plt.ylabel('metric value')
            plt.grid()
            plt.tight_layout()
            plt.pause(0.001)

        plt.show()


def test_runner(graph):
    crawlers = [
        # MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=1),
        # BreadthFirstSearchCrawler(graph, initial_seed=1),
        RandomWalkCrawler(graph, initial_seed=1),
        RandomCrawler(graph, initial_seed=1),
        # MultiCrawler(graph, [
        #     # RandomCrawler(graph, initial_seed=1),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        # ])
    ]

    target_set = set(get_top_centrality_nodes(graph, 'degree', count=int(0.1 * graph[Stat.NODES])))
    metrics = [
        Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        Metric(r'$|V_o \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=500, step=10)
    ci.run()


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    g = GraphCollections.get(name)

    test_runner(g)
