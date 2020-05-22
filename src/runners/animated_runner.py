from matplotlib import pyplot as plt

from utils import USE_CYTHON_CRAWLERS

if USE_CYTHON_CRAWLERS:
    from base.cbasic import CCrawler as Crawler
else:
    from base.graph import MyGraph
    from crawlers.basic import Crawler, PreferentialObservedDegreeCrawler, MaximumObservedDegreeCrawler

from graph_io import GraphCollections
from statistics import Stat


class Metric:
    def __init__(self, name, callback, **kwargs):
        self.name = name
        self._callback = callback
        self._kwargs = kwargs

    def __call__(self, crawler: Crawler):
        return self._callback(crawler, **self._kwargs)


# TODO need to check several statistics / metrics
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
        # for crawler in crawlers:  FIXME
        #     assert crawler.orig_graph == graph
        self.crawlers = crawlers
        self.metrics = metrics
        self.budget = min(budget, graph[Stat.NODES]) if budget > 0 else graph[Stat.NODES]
        assert step < self.budget
        self.step = step
        self.nrows = 1
        self.ncols = 1
        scale = 5
        self.title = "Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, self.graph[Stat.NODES], self.graph[Stat.EDGES], self.graph[Stat.MAX_DEGREE])

        plt.figure(self.title, figsize=(1 + scale * self.ncols, scale * self.nrows))

    def run(self, ylims=None, xlabel='iteration, n', ylabel='metric value'):
        linestyles = ['-', '--', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']

        step_seq = []
        crawler_metric_seq = dict([(c, dict([(m, []) for m in self.metrics])) for c in self.crawlers])

        i = 0
        while i < self.budget:
            batch = min(self.step, self.budget - i)
            i += batch
            step_seq.append(i)

            plt.cla()
            plt.title(self.title)
            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(batch)

                for m, metric in enumerate(self.metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))
                    plt.plot(step_seq, metric_seq, marker='.',
                             linestyle=linestyles[m % len(linestyles)],
                             color=colors[c % len(colors)],
                             label=r'%s, %s' % (crawler.name[:30], metric.name))

            plt.legend()
            if ylims:
                plt.ylim(ylims)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid()
            plt.tight_layout()
            plt.pause(0.001)

        # file_path = os.path.join(RESULT_DIR, self.graph.name, 'crawling_plot')
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # for metric in self.metrics:
        #     file_name = os.path.join(file_path, metric.name + ':' +
        #                              ','.join([crawler.name for crawler in self.crawlers]) + 'animated.png')
        #     logging.info('Saved pic ' + file_name)
        #     plt.savefig(file_name)
        plt.show()


def test_runner(graph):
    from crawlers.basic import Crawler, RandomWalkCrawler, RandomCrawler
    from statistics import Stat, get_top_centrality_nodes

    crawlers = [
        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=1),
        PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=1),
        # BreadthFirstSearchCrawler(graph, initial_seed=1),
        # RandomWalkCrawler(graph, initial_seed=1),
        # RandomCrawler(graph, initial_seed=1),
        # MultiCrawler(graph, [
        #     # RandomCrawler(graph, initial_seed=1),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        #     BreadthFirstSearchCrawler(graph),
        # ])
    ]

    target_set = set(get_top_centrality_nodes(graph, Stat.DEGREE_DISTR, count=int(0.1 * graph[Stat.NODES])))
    metrics = [
        Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        Metric(r'$|V_{all} \cap V^*|/|V^*|$', lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=50000, step=500)
    ci.run(ylims=(0, 1))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    g = GraphCollections.get(name)

    test_runner(g)
