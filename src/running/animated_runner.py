import os
from matplotlib import pyplot as plt

from base.cgraph import MyGraph
from crawlers.advanced import ThreeStageMODCrawler
from running.metrics_and_runner import CrawlerRunner, TopCentralityMetric
from graph_io import GraphCollections
from graph_stats import Stat


class AnimatedCrawlerRunner(CrawlerRunner):
    """
    Runs several crawlers and measures several metrics for a given graph.
    Visualizes measurements in dynamics at one plot.
    """

    def __init__(self, graph: MyGraph, crawler_defs, metric_defs, budget: int = -1, step: int = -1):
        """
        :param graph: graph to run
        :param crawler_defs: list of crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metric_defs: list of metric definitions to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        super().__init__(graph, crawler_defs=crawler_defs, metric_defs=metric_defs, budget=budget, step=step)

        self.nrows = 1
        self.ncols = 1
        scale = 9
        self.title = "Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, self.graph[Stat.NODES], self.graph[Stat.EDGES], self.graph[Stat.MAX_DEGREE])

        plt.figure(self.title, figsize=(1 + scale * self.ncols, scale * self.nrows))

    def run(self, ylims=None, xlabel='iteration, n', ylabel='metric value', swap_coloring_scheme=False, save_to_file=None):
        """
        :param same_initial_seed: use the same initial seed for all crawler instances (NOT IMPLEMENTED) FIXME
        :param ylims: (low, up)
        :param xlabel: by default 'iteration, n'
        :param ylabel: by default 'metric value'
        :param swap_coloring_scheme: by default metrics differ in linestyle, crawlers differ in color. Set True to swap
        :param save_to_file: specify the full path here to save picture
        :return:
        """
        crawlers, metrics, batch_generator = self._init_runner()
        linestyles = ['-', '--', ':', '-.']
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        step = 0
        step_seq = [0]  # starting for point 0
        crawler_metric_seq = dict([(c, dict([(m, [0]) for m in metrics])) for c in crawlers])
        for batch in batch_generator:
            step += batch
            step_seq.append(step)

            plt.cla()
            plt.title(self.title)
            for c, crawler in enumerate(crawlers):
                crawler.crawl_budget(int(batch))

                for m, metric in enumerate(metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))
                    ls, col = (c, m) if swap_coloring_scheme else (m, c)
                    plt.plot(step_seq, metric_seq, marker='.',
                             linestyle=linestyles[ls % len(linestyles)],
                             color=colors[col % len(colors)],
                             label=r'%s, %s' % (crawler.name, metric.name))

            plt.legend()
            if ylims:
                plt.ylim(ylims)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(b=True)
            plt.tight_layout()
            plt.pause(0.001)

        if save_to_file is not None:
            if not os.path.exists(os.path.dirname(save_to_file)):
                os.makedirs(os.path.dirname(save_to_file))
            plt.savefig(save_to_file)
        plt.show()


def test_runner():
    from graph_stats import Stat

    graph = GraphCollections.get('petster-hamster')
    p = 0.01
    budget = int(0.05 * graph.nodes())
    s = int(budget / 2)
    crawler_defs = [
        # MaximumObservedDegreeCrawler(graph, batch=1),
        # (MaximumObservedDegreeCrawler, {'batch': 1, 'initial_seed': 1}),
        # PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=1).definition,
        # BreadthFirstSearchCrawler(graph, initial_seed=1),
        # RandomWalkCrawler(graph, initial_seed=1),
        # RandomCrawler(graph, initial_seed=1),
        # (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 10})}),
        (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 1}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 10}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 30}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p, 'b': 100}),
    ]

    metric_defs = [
        # TopCentralityMetric(graph, top=0.1, centrality=Stat.DEGREE_DISTR, measure='Pr', part='nodes'),
        # Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # TopCentralityMetric(graph, top=0.1, centrality=Stat.DEGREE_DISTR, measure='Re', part='nodes'),
        TopCentralityMetric(graph, top=p, centrality=Stat.DEGREE_DISTR.short, measure='F1', part='answer').definition,
    ]

    ci = AnimatedCrawlerRunner(graph, crawler_defs, metric_defs)
    ci.run(ylims=(0, 1))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    test_runner()
