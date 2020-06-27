from matplotlib import pyplot as plt

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler
from crawlers.advanced import ThreeStageMODCrawler
from crawlers.multiseed import MultiInstanceCrawler
from running.metrics_and_runner import CrawlerRunner, TopCentralityMetric, Metric
from graph_io import GraphCollections
from statistics import Stat, get_top_centrality_nodes


# TODO need to check several statistics / metrics
class AnimatedCrawlerRunner(CrawlerRunner):
    def __init__(self, graph: MyGraph, crawlers, metrics, budget: int=-1, step: int=1):
        """
        :param graph: graph to run
        :param crawlers: list of crawlers or crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metrics: list of metrics to compute at each step. Metric should be callable function
         crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        super().__init__(graph, crawlers=crawlers, metrics=metrics, budget=budget, step=step)

        self.nrows = 1
        self.ncols = 1
        scale = 9
        self.title = "Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, self.graph[Stat.NODES], self.graph[Stat.EDGES], self.graph[Stat.MAX_DEGREE])

        plt.figure(self.title, figsize=(1 + scale * self.ncols, scale * self.nrows))

    def run(self, same_initial_seed=False, ylims=None, xlabel='iteration, n', ylabel='metric value', swap_coloring_scheme=False, save_to_file=None):
        """
        :param same_initial_seed: use the same initial seed for all crawler instances
        :param ylims: (low, up)
        :param xlabel: by default 'iteration, n'
        :param ylabel: by default 'metric value'
        :param swap_coloring_scheme: by default metrics differ in linestyle, crawlers differ in color. Set True to swap
        :param save_to_file: specify the full path here to save picture
        :return:
        """
        linestyles = ['-', '--', ':', '.-']
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        # Initialize crawlers and metrics
        crawlers = self.crawlers + [Crawler.from_definition(self.graph, d) for d in self.crawler_defs]
        metrics = self.metrics + [Metric.from_definition(self.graph, d) for d in self.metric_defs]

        step_seq = []
        crawler_metric_seq = dict([(c, dict([(m, []) for m in metrics])) for c in crawlers])

        i = 0
        while i < self.budget:
            batch = min(self.step, self.budget - i)
            i += batch
            step_seq.append(i)

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

        # file_path = os.path.join(RESULT_DIR, self.graph.name, 'crawling_plot')
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # for metric in metrics:
        #     file_name = os.path.join(file_path, metric.name + ':' +
        #                              ','.join([crawler.name for crawler in self.crawlers]) + 'animated.png')
        #     logging.info('Saved pic ' + file_name)
        #     plt.savefig(file_name)
        if save_to_file is not None:
            plt.savefig(save_to_file)
        plt.show()


def test_runner(graph):
    from statistics import Stat, get_top_centrality_nodes

    p = 0.01
    budget = int(0.05 * g.nodes())
    s = int(budget / 2)
    crawlers = [
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

    metrics = [
        # TopCentralityMetric(graph, top=0.1, centrality=Stat.DEGREE_DISTR, measure='Pr', part='nodes'),
        # Metric(r'$|V_{all}|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        # TopCentralityMetric(graph, top=0.1, centrality=Stat.DEGREE_DISTR, measure='Re', part='nodes'),
        TopCentralityMetric(graph, top=p, centrality=Stat.DEGREE_DISTR.short, measure='F1', part='answer'),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=budget, step=int(budget/100))
    ci.run(ylims=(0, 1))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'

    # name = 'sc-shipsec5'
    g = GraphCollections.get(name)

    test_runner(g)
