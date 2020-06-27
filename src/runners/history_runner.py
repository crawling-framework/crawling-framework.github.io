import datetime
import glob
import json
import multiprocessing
import os
import re
import logging
import imageio
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import PICS_DIR, RESULT_DIR

from base.cgraph import MyGraph, seed_random
from crawlers.cbasic import Crawler, CrawlerUpdatable as CrawlerUpdatable, \
    MaximumObservedDegreeCrawler, definition_to_filename
from crawlers.advanced import ThreeStageMODCrawler, ThreeStageCrawler, AvrachenkovCrawler
from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections
from runners.metric_runner import CrawlerRunner, TopCentralityMetric, Metric
from runners.merger import CrawlerRunsMerger
from statistics import Stat, get_top_centrality_nodes


def remap_iter(total=400):
    """ Remapping steps depending on used budget - on first iters step=1, on last it grows ~x^2
    """
    step_budget = 0
    remap_iter_to_step = {}
    for i in range(total):  # for budget less than 100 mln nodes
        remap = int(max(1, step_budget / 20))
        remap_iter_to_step[step_budget] = remap
        step_budget += remap
    return remap_iter_to_step


def make_gif(crawler_name, duration=1):
    images = []
    # crawler_name = crawler_name.replace('[', '\\[').replace(']', '\\]')
    filenames = glob.glob(PICS_DIR + "{}*_*.png".format(crawler_name))  # [:3] #if is too long
    filenames.sort()
    logging.info("adding")
    logging.info(filenames)
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    file_path = os.path.join(PICS_DIR, "gifs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    name = file_path + "{}.gif".format(crawler_name)
    imageio.mimsave(name, images, duration=0.2 * duration, loop=2)
    logging.info("made gif " + name)


# TODO : create subclasses for various draw_modes
class CrawlerHistoryRunner(CrawlerRunner):
    """ Runs crawlers, measure metrics and save history. Step sequence is logarithmically growing,
    universal for all graphs
    """
    def __init__(self, graph: MyGraph, crawlers, metrics,
                 draw_mode=None, batches_per_pic=1, layout_pos=None):
        """
        :param graph: graph to run
        :param crawlers: list of crawlers or crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metrics: list of metrics to compute at each step. Metric should be callable function
         crawler -> float, and have name
        :param batches_per_pic: save picture every batches_per_pic batches. more -> faster
        :param layout_pos:
        :return:
        """
        super().__init__(graph, crawlers=crawlers, metrics=metrics)

        self.batches_per_pic = batches_per_pic
        self.draw_mode = draw_mode  # 'metric'   # 'traversal' / 'metric' / 'None
        # print('Crawler runner with budget={},step={},iters={}. Draw {}'.format(
        #     self.budget, step, ceil(self.budget / step), draw_mod))
        self.layout_pos = layout_pos
        # if len(self.crawlers) > 1:
        #     self.nrows = 2
        # self.ncols = ceil(len(self.crawlers) / self.nrows)

        logging.info('Crawler runner:\n Graph: %s\n Crawlers %s\n Metrics: %s' % (
            "%s with N=%s E=%s" % (self.graph.name, self.graph.nodes(), self.graph.edges()),
            [c.name if isinstance(c, Crawler) else definition_to_filename(c) for c in crawlers],
            [m.name if isinstance(m, Metric) else definition_to_filename(m) for m in metrics]))

    def _save_history(self, crawler_metric_seq, step_seq):
        pbar = tqdm(total=len(crawler_metric_seq), desc='Saving history')
        for crawler in crawler_metric_seq.keys():
            metric_seq = crawler_metric_seq[crawler]
            for metric in metric_seq.keys():
                metric_value = dict((step_seq[it], value) for it, value in enumerate(metric_seq[metric]))
                path_pattern = CrawlerRunsMerger.names_to_path(self.graph.name,
                    definition_to_filename(crawler.definition), definition_to_filename(metric.definition))
                directory = os.path.dirname(path_pattern)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                ix = 0
                while True:
                    path = path_pattern.replace('*', str(ix))
                    if not os.path.exists(path):  # if name exists, adding number to it
                        break
                    ix += 1

                with open(path, 'w') as f:
                    json.dump(metric_value, f, indent=1)
            pbar.update(1)
        pbar.close()

    def _save_pics(self):  # TODO need to change it somehow
        raise NotImplementedError()
        if self.draw_mode == 'metric':
            plt.legend()
            for metric in self.metrics:
                plt.title(metric.name)
                file_path = os.path.join(RESULT_DIR, 'k={:1.2f}'.format(self.top_k_percent), self.graph.name,
                                         'crawling_plot')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                plt.savefig(os.path.join(file_path, metric.name + ':' +
                                         ','.join([crawler.name for crawler in self.crawlers]) + '.png'))
        elif self.draw_mode == 'traversal':  # if traversal
            for crawler in self.crawlers:
                make_gif(crawler_name=crawler.name, duration=2)
                logging.info('compiled +')

    def run_parallel(self, num_processes=multiprocessing.cpu_count()):
        """ Run in parallel crawlers and measure metrics. In the end, the measurements are saved.
        """
        from time import time

        t = time()
        jobs = []
        for i in range(num_processes):
            logging.info('Start parallel iteration %s of %s' % (i+1, num_processes))
            seed_random(int(time() * 1e7 % 1e9))  # to randomize t_random in parallel processes
            # little multiprocessing magic, that calculates several iterations in parallel
            p = multiprocessing.Process(target=self.run, args=())
            jobs.append(p)
            p.start()
            # FIXME it could be a problem when several runners try to create and write into one directory

        for i, p in enumerate(jobs):
            p.join()
            logging.info('Completed job %s/%s' % (i+1, num_processes))

        msg = 'Completed %s runs for graph %s with N=%s E=%s. Time elapsed %.1fs.' % (
            num_processes, self.graph.name, self.graph.nodes(), self.graph.edges(), time() - t)
        # TODO how to catch exceptions?

        # except Exception as e:
        #     msg = "Failed graph %s after %.2fs with error %s" % (graph_name, time.time() - start_time, e)

        print(msg)
        return msg

    def run(self, same_initial_seed=False, draw_networkx=False):
        """ Run crawlers and measure metrics. In the end, the measurements are saved.

        :param same_initial_seed: use the same initial seed for all crawler instances
        :param draw_networkx: if True draw graphs with colored nodes using networkx and save it as
         gif. Use for small graphs only.
        :return:
        """
        # Initialize crawlers and metrics
        # if same_initial_seed:
        #     initial_seeds = [self.graph.random_node()] * len(self.crawler_defs)
        # else:
        #     initial_seeds = self.graph.random_nodes(len(self.crawler_defs))
        # crawlers = [_class(self.graph, initial_seed=initial_seeds[i], **kwargs)
        #             for i, (_class, kwargs) in enumerate(self.crawler_defs)]
        crawlers = self.crawlers + [Crawler.from_definition(self.graph, d) for d in self.crawler_defs]
        metrics = self.metrics + [Metric.from_definition(self.graph, d) for d in self.metric_defs]

        if draw_networkx:  # if it is traversal
            for crawler in crawlers:  # TODO test this one
                file_path = os.path.join(PICS_DIR, self.graph.name)
                if os.path.exists(file_path):
                    for file in glob.glob(file_path + crawler.name + "*.png"):
                        os.remove(file)
                else:
                    os.makedirs(file_path)
            if self.layout_pos is None:
                logging.info('need to draw layout pos')
                self.layout_pos = nx.spring_layout(self.graph.snap_to_networkx, iterations=40)
            else:
                logging.info('using current layout')
        # else:
        #     self.nrows = 1
        #     self.ncols = 1
        #     scale = 5
        #     plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
        #         self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
        #                figsize=(20, 10))  # (1 + scale * self.ncols, scale * self.nrows), )

        step_seq = [0]  # starting for point 0
        crawler_metric_seq = dict([(c, dict([(m, [0]) for m in metrics])) for c in crawlers])
        pbar = tqdm(total=self.budget, desc='Running iterations')  # drawing crawling progress bar

        remap_iter_to_step = remap_iter(400)  # dynamic step size
        i = 0  # current step
        batch = 1  # current batch
        while i < self.budget:
            assert i in remap_iter_to_step
            batch = min(self.budget - i, remap_iter_to_step[i])
            i += batch
            step_seq.append(i)

            # if i % 50000 == 0:  # making backup every 50k
            #     self.save_history(crawler_metric_seq, step_seq)
            #     self.save_pics()
            #     logging.info('\n{} : Iteration {} made backup of pics and history'.format(datetime.datetime.now(), i))

            for crawler in crawlers:
                crawler.crawl_budget(batch)
                for metric in metrics:
                    # Calculate metric for crawler
                    value = metric(crawler)
                    crawler_metric_seq[crawler][metric].append(value)

                    # Draw graph with networkx
                    if draw_networkx and (i % self.batches_per_pic == 0):
                        networkx_graph = crawler.orig_graph.snap_to_networkx

                        # coloring nodes
                        s = [n.GetId() for n in crawler.orig_graph.snap.Nodes()]
                        s.sort()
                        gen_node_color = ['gray'] * (max(s) + 1)
                        # for node in crawler.observed_set:
                        for node in crawler.nodes_set:
                            gen_node_color[node] = 'y'
                        for node in crawler.crawled_set:
                            gen_node_color[node] = 'cyan'
                        for node in crawler.seed_sequence_[:-batch]:
                            gen_node_color[node] = 'red'

                        plt.title(crawler.name + " " + str(len(crawler.crawled_set)))
                        nx.draw(networkx_graph, pos=self.layout_pos,
                                with_labels=(len(gen_node_color) < 1000),  # if little, we draw
                                node_size=100, node_list=networkx_graph.nodes,
                                node_color=[gen_node_color[node] for node in networkx_graph.nodes])
                        file_path = os.path.join(PICS_DIR, self.graph.name)
                        plt.savefig(file_path + '/{}_{}.png'.format(crawler.name, str(i).zfill(5)))
            pbar.update(batch)

        pbar.close()  # closing progress bar
        self._save_history(crawler_metric_seq, step_seq)  # saving ending history
        # self._save_pics()
        logging.info("Finished running at %s" % (datetime.datetime.now()))
        # if draw_networkx:
        #     plt.show()


def test_crawler_runner():
    # g = GraphCollections.get('socfb-Bingham82')
    g = GraphCollections.get('digg-friends')

    p = 0.01
    # initial_seed = g.random_nodes(1)
    crawler_defs = [
        # (MaximumObservedDegreeCrawler, {'batch': 1}),
        # (MaximumObservedDegreeCrawler, {'batch': 10}),
        # (MaximumObservedDegreeCrawler, {'batch': 100}),
        # (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 10})}),
        # (ThreeStageCrawler, {'s': 699, 'n': 1398, 'p': p}),
    ]
    metrics = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]
    cr = CrawlerHistoryRunner(g, crawler_defs, metrics)
    # cr.run()
    cr.run_parallel(4)

    # # Run merger
    # crm = CrawlerRunsMerger([g.name], crawler_defs, [metrics[0].name], n_instances=10)
    # crm.draw_by_crawler(x_lims=(0, 1500), x_normalize=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # # graph_name = 'digg-friends'
    # # graph_name = 'douban'
    # # graph_name = 'ego-gplus'
    # # graph_name = 'slashdot-threads;'
    # graph_name = 'facebook-wosn-links'
    # # graph_name = 'petster-hamster'
    # graph = GraphCollections.get(graph_name, giant_only=True)
    #
    # initial_seed = graph.random_node()
    # crawlers = [
    #     MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed[0]),
    #     MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed[0]),
    #     MaximumObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed[0]),
    # ]
    # logging.info([c.name for c in crawlers])
    # metrics = []
    # target_set = {}  # {i.name: set() for i in statistics}
    # for target_statistics in [s for s in Stat if 'DISTR' in s.name]:
    #     target_set = set(get_top_centrality_nodes(graph, target_statistics, count=int(0.1 * graph[Stat.NODES])))
    #     # creating metrics and giving callable function to it (here - total fraction of nodes)
    #     # metrics.append(Metric(r'observed' + target_statistics.name, lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]))
    #     metrics.append(Metric(r'crawled_' + target_statistics.name,  # TODO rename crawled to observed
    #                           lambda crawler, t: len(t.intersection(crawler.crawled_set)) / len(t), t=target_set
    #                           ))
    #
    # ci = CrawlerRunner(graph, crawlers, metrics, budget=0,
    #                    # step=ceil(10 ** (len(str(graph.nodes())) - 3)),
    #                    # if 5*10^5 then step = 10**2,if 10^7 => step=10^4
    #                    # batches_per_pic=10,
    #                    # draw_mod='traversal', layout_pos=layout_pos,
    #                    )  # if you want gifs, draw_mod='traversal'. else: 'metric'
    # ci.run()

    test_crawler_runner()
