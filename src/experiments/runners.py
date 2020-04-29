import glob
import os

import imageio
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from centralities import get_top_centrality_nodes
from crawlers.basic import MaximumObservedDegreeCrawler, \
    PreferentialObservedDegreeCrawler, BreadthFirstSearchCrawler, RandomCrawler, Crawler
from graph_io import GraphCollections
from graph_io import MyGraph
from statistics import Stat
from utils import PICS_DIR  # PICS_DIR = '/home/jzargo/PycharmProjects/crawling/crawling/pics/'


def make_gif(crawler_name, duration=1):
    images = []
    # crawler_name = crawler_name.replace('[', '\\[').replace(']', '\\]')
    filenames = glob.glob(PICS_DIR + "{}*_*.png".format(crawler_name))  # [:3] #if is too long
    filenames.sort()
    print("adding")
    print(filenames)
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    file_path = PICS_DIR + "result/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    name = file_path + "{}.gif".format(crawler_name)
    imageio.mimsave(name, images, duration=0.2 * duration, loop=2)
    print("made gif " + name)


class Metric:
    def __init__(self, name, callback):
        self.name = name
        self._callback = callback

    def __call__(self, crawler: Crawler):
        return self._callback(crawler)


class AnimatedCrawlerRunner:
    def __init__(self, graph: MyGraph, crawlers, metrics, budget=-1, step=1,
                 draw_mod='metric', batches_per_pic=1, layout_pos=None):
        """
        :param graph:
        :param crawlers: list of crawlers to run
        :param metrics: list of metrics to compute at each step. Metric should be callable function
         crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :param batches_per_pic: save picture every batches_per_pic batches. more -> faster
        :return:
        """
        self.graph = graph
        g = self.graph.snap
        for crawler in crawlers:
            assert crawler.orig_graph == graph
        self.crawlers = crawlers
        self.metrics = metrics
        self.budget = min(budget, g.GetNodes() - 1) if budget > 0 else g.GetNodes()
        assert step < self.budget
        self.step = step
        self.batches_per_pic = batches_per_pic
        self.draw_mod = draw_mod  # 'metric'   # 'traversal' / 'metric'
        self.nrows = 1
        self.ncols = 1
        self.layout_pos = layout_pos
        scale = 5
        # if len(self.crawlers) > 1:
        #     self.nrows = 2
        # self.ncols = ceil(len(self.crawlers) / self.nrows)

        if self.draw_mod == 'metric':
            fig = plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
                self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
                             figsize=(1 + scale * self.ncols, scale * self.nrows))
        else:  # if it is traversal
            plt.figure(figsize=(14, 6))
            for crawler in crawlers:
                if os.path.exists(PICS_DIR):
                    for file in glob.glob(PICS_DIR + crawler.name + "*.png"):
                        os.remove(file)
                else:
                    os.makedirs(PICS_DIR)
            if self.layout_pos is None:
                print('need to draw layout pos')
                self.layout_pos = nx.spring_layout(graph.snap_to_networkx, iterations=40)
            else:
                print('using current layout')

    def run(self):
        linestyles = ['-', '--', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']  # for different methods

        step_seq = []
        crawler_metric_seq = dict([(c, dict([(m, []) for m in self.metrics])) for c in self.crawlers])

        i = 0
        logging.critical('Crawling with budget {} and step {}'.format(self.budget, self.step))
        pbar = tqdm(total=self.budget)  # drawing crawling progress bar
        while i < self.budget:
            batch = min(self.step, self.budget - i)
            i += batch
            step_seq.append(i)
            pbar.update(batch)
            plt.cla()
            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(budget=batch)

                for m, metric in enumerate(self.metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))  # calculate metric for crawler

                    if (self.draw_mod == 'metric') and (i % self.batches_per_pic == 0):
                        plt.plot(step_seq, metric_seq, marker='.',
                                 linestyle=linestyles[m % len(linestyles)],
                                 color=colors[c % len(colors)],
                                 label=r'%s, %s' % (crawler.name, metric.name))
                    elif (self.draw_mod == 'traversal') and (i % self.batches_per_pic == 0):
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
                        plt.savefig(PICS_DIR + '{}_{}.png'.format(crawler.name, str(i).zfill(3)))
                        plt.cla()
                        # plt.show()

            if self.draw_mod == 'metric':
                plt.legend()
                plt.ylim((0, 1))
                plt.xlabel('iteration, n')
                plt.ylabel('metric value')
                plt.grid()
                plt.tight_layout()
                plt.pause(0.0001)

        # after all drawing gif
        if self.draw_mod == 'metric':
            plt.show()
        else:  # if traversal
            for crawler in self.crawlers:
                make_gif(crawler_name=crawler.name, duration=2)
                print('compiled +')
        pbar.close()  # closing progress bar


def test_runner(graph, layout_pos=None):
    crawlers = [
        # MultiCrawler(graph, crawlers=[
        #     DepthFirstSearchCrawler(graph, batch=1, initial_seed=0),
        #     DepthFirstSearchCrawler(graph, batch=1, initial_seed=10),
        # ]),
        # DepthFirstSearchCrawler(graph, initial_seed=45),
        # ForestFireCrawler(graph, initial_seed=1),
        # # RandomWalkCrawler(graph, initial_seed=1),
        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=45),
        PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=45),
        BreadthFirstSearchCrawler(graph, initial_seed=45),
        RandomCrawler(graph, initial_seed=45),
    ]
    print(crawlers[0].name)
    # change target set to calculate another metric
    target_set = set(get_top_centrality_nodes(graph, 'degree', count=int(0.1 * graph[Stat.NODES])))
    metrics = [  # creating metrics and giving callable function to it (here - total fraction of nodes)
        Metric(r'$|V_o|/|V|$', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        Metric(r'$|V_o \cap V^*|/|V^*|$',
               lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=100, step=1,
                               draw_mod='traversal', layout_pos=layout_pos, batches_per_pic=1
                               )  # if you want gifs, draw_mod='traversal'. else: 'metrics'
    ci.run()


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    name = 'dolphins'
    g = GraphCollections.get(name)
    # from crawlers.multiseed import test_carpet_graph, MultiCrawler
    # name = 'carpet_graph'
    # g, layout_pos = test_carpet_graph(10, 10)  # GraphCollections.get(name)
    logging.critical("running graph ".format(name))
    test_runner(g,
                # layout_pos
                )
