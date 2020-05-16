import datetime
import glob
import json
import os
from math import ceil, log
import logging
import imageio
import networkx as nx
# import snap
from matplotlib import pyplot as plt
from tqdm import tqdm

from crawlers.advanced import CrawlerWithAnswer
from crawlers.basic import DepthFirstSearchCrawler, RandomWalkCrawler, MaximumObservedDegreeCrawler, \
    PreferentialObservedDegreeCrawler, BreadthFirstSearchCrawler, RandomCrawler, Crawler, SnowBallSampling
from crawlers.multiseed import MultiCrawler
# ForestFireCrawler
from graph_io import GraphCollections
from graph_io import MyGraph
from statistics import Stat
from statistics import get_top_centrality_nodes
from utils import PICS_DIR, RESULT_DIR, REMAP_ITER  # PICS_DIR = '/home/jzargo/PycharmProjects/crawling/crawling/pics/'

REMAP_ITER_TO_STEP = REMAP_ITER(300)  # dynamic step size


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


class Metric:
    def __init__(self, name, callback, **kwargs):
        self.name = name
        self._callback = callback
        self.kwargs = kwargs
        # for arg, value in kwargs.items():
        #     setattr(self, arg, value)

    def __call__(self, crawler: Crawler):
        return self._callback(crawler, **self.kwargs)


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

        plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
            self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
                   figsize=(1 + scale * self.ncols, scale * self.nrows))

    def run(self):
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
            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(budget=batch)
                if isinstance(crawler, CrawlerWithAnswer):
                    crawler._compute_answer()

                for m, metric in enumerate(self.metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))
                    plt.plot(step_seq, metric_seq, marker='.',
                             linestyle=linestyles[m % len(linestyles)],
                             color=colors[c % len(colors)],
                             label=r'%s, %s' % (crawler.name, metric.name))

            plt.legend()
            plt.ylim((0, 1))
            plt.xlabel('iteration, n')
            plt.ylabel('metric value')
            plt.grid()
            plt.tight_layout()
            plt.pause(0.001)

        file_path = os.path.join(RESULT_DIR, self.graph.name, 'crawling_plot')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for metric in self.metrics:
            file_name = os.path.join(file_path, metric.name + ':' +
                                     ','.join([crawler.name for crawler in self.crawlers]) + 'animated.png')
            logging.info('Saved pic ' + file_name)
            plt.savefig(file_name)
        plt.show()


class CrawlerRunner:  # take budget=int(graph.snap.GetNodes() / 10)
    def __init__(self, graph: MyGraph, crawlers, metrics, budget=-1, step=1,
                 draw_mod=None, batches_per_pic=1, layout_pos=None):
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
        self.budget = min(budget, g.GetNodes()) if budget > 0 else g.GetNodes()
        assert step < self.budget
        self.step = step

        self.batches_per_pic = batches_per_pic
        self.draw_mod = draw_mod  # 'metric'   # 'traversal' / 'metric' / 'None
        print('Crawler runner with budget={},step={},iters={}. Draw {}'.format(
            self.budget, step, ceil(self.budget / step), draw_mod))
        self.nrows = 1
        self.ncols = 1
        self.layout_pos = layout_pos
        scale = 5
        # if len(self.crawlers) > 1:
        #     self.nrows = 2
        # self.ncols = ceil(len(self.crawlers) / self.nrows)

        if self.draw_mod == 'metric':
            plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
                self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
                       figsize=(20, 10))  # (1 + scale * self.ncols, scale * self.nrows), )

        elif self.draw_mod == 'traversal':  # if it is traversal
            for crawler in crawlers:
                file_path = os.path.join(PICS_DIR, graph.name)
                if os.path.exists(file_path):
                    for file in glob.glob(file_path + crawler.name + "*.png"):
                        os.remove(file)
                else:
                    os.makedirs(file_path)
            if self.layout_pos is None:
                logging.info('need to draw layout pos')
                self.layout_pos = nx.spring_layout(graph.snap_to_networkx, iterations=40)
            else:
                logging.info('using current layout')

    def save_history(self, crawler_metric_seq, step_seq, step=1):
        # print('crawler_metric_seq', crawler_metric_seq)
        logging.info('Making backup')
        for crawler_name in tqdm(crawler_metric_seq.keys()):
            crawler_seq = crawler_metric_seq[crawler_name]
            for metric_object in crawler_seq.keys():
                if step is None:
                    metric_list = dict((it * 1000, res) for it, res in enumerate(crawler_seq[metric_object]))
                else:
                    # print('seq', crawler_seq[metric_object], '\n')

                    metric_list = dict((step_seq[it], res) for it, res in enumerate(crawler_seq[metric_object]))
                # like this: file_path = ../results/Graph_name/step=10,budget=10/MOD/crawled_centrality[NUM].json
                file_path = os.path.join(RESULT_DIR, self.graph.name,
                                         'budget={}'.format(self.budget), crawler_name.name)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name = os.path.join(file_path, metric_object.name + '.json')

                if os.path.isfile(file_name):  # if name exists, adding number to it
                    expand = 1
                    while True:
                        expand += 1
                        new_file_name = str(file_name.split(".json")[0]) + str(expand) + ".json"
                        if os.path.isfile(new_file_name):
                            continue
                        else:
                            file_name = new_file_name
                            break

                with open(file_name, 'w') as f:
                    json.dump(metric_list, f)
                # print('saving', crawler_name.name, metric_object.name, metric_list, file_name)

    def save_pics(self):  # TODO need to change it somehow
        if self.draw_mod == 'metric':
            plt.legend()
            for metric in self.metrics:
                plt.title(metric.name)
                file_path = os.path.join(RESULT_DIR, self.graph.name, 'crawling_plot')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                plt.savefig(os.path.join(file_path, metric.name + ':' +
                                         ','.join([crawler.name for crawler in self.crawlers]) + '.png'))
        elif self.draw_mod == 'traversal':  # if traversal
            for crawler in self.crawlers:
                make_gif(crawler_name=crawler.name, duration=2)
                logging.info('compiled +')

    def run(self):
        linestyles = ['-', '--', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']  # for different methods

        step_seq = [0]  # starting for point 0
        crawler_metric_seq = dict([(c, dict([(m, [0]) for m in self.metrics])) for c in self.crawlers])

        i, batch = 0, 1
        # logging.info('Crawling method {} with budget {} and step {}'.format(self.self.budget, self.step))
        pbar = tqdm(total=self.budget)  # drawing crawling progress bar
        # print('self metrics', [(metric._callback, metric.name + '\n') for metric in self.metrics])

        while i < self.budget:
            # batch =self.budget - i)
            if i in REMAP_ITER_TO_STEP:
                batch = min(self.budget - i, REMAP_ITER_TO_STEP[i])
            # else: batch =
            i += batch  # TODO make exponential steps from 1 to 1000 depending on current iteration. (exp(log))
            step_seq.append(i)
            pbar.update(batch)

            # print('i={}/{},batch={}'.format(i, self.budget, batch))
            if i % 50000 == 0:  # making backup every 50k
                self.save_history(crawler_metric_seq, step_seq)
                self.save_pics()
                logging.info('\n{} : Iteration {} made backup of pics and history'.format(datetime.datetime.now(), i))

            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(budget=batch)
                for m, metric in enumerate(self.metrics):
                    crawler_metric_seq[crawler][metric].append(metric(crawler))  # calculate metric for crawler
                    # print('adding', c, crawler.name, m, metric.name, crawler_metric_seq[crawler][metric])
                    if (self.draw_mod == 'metric') and (i % self.batches_per_pic == 0):
                        continue
                        # plt.plot(step_seq, metric_seq, marker='.',  # TODO remove extra labels
                        #          linestyle=linestyles[m % len(linestyles)],
                        #          color=colors[c % len(colors)],
                        #          label=r'%s, %s' % (crawler.name, metric.name))
                        # # else:
                        # #     plt.plot(step_seq, metric_seq, marker='.',
                        # #              linestyle=linestyles[m % len(linestyles)],
                        # #              color=colors[c % len(colors)])
                        # plt.xlabel('iteration, n')
                        # plt.ylabel('metric value')


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
                        file_path = os.path.join(PICS_DIR, self.graph.name)
                        plt.savefig(file_path + '/{}_{}.png'.format(crawler.name, str(i).zfill(5)))
        #                plt.cla()
        #                      plt.show()

        pbar.close()  # closing progress bar
        self.save_history(crawler_metric_seq, step_seq)  # saving ending history
        self.save_pics()
        # plt.show()
        logging.info('{} : finished running'.format(datetime.datetime.now()))


def test_runner(graph, animated=False, statistics: list = None, layout_pos=None):
    import random
    initial_seed = random.sample([n.GetId() for n in graph.snap.Nodes()], 1)[0]
    crawlers = [
        # RandomWalkCrawler(graph, initial_seed=initial_seed),
        # RandomCrawler(graph, initial_seed=initial_seed),

        # ForestFireCrawler(graph, initial_seed=initial_seed), # FIXME fix and rewrite
        DepthFirstSearchCrawler(graph, initial_seed=initial_seed),
        # SnowBallSampling(graph, p=0.1, initial_seed=initial_seed),
        # SnowBallSampling(graph, p=0.25, initial_seed=initial_seed),
        # SnowBallSampling(graph, p=0.5, initial_seed=initial_seed),
        # SnowBallSampling(graph, p=0.75, initial_seed=initial_seed),
        # SnowBallSampling(graph, p=0.9, initial_seed=initial_seed),
        BreadthFirstSearchCrawler(graph, initial_seed=initial_seed),  # is like take SBS with p=1

        MaximumObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed),
        # MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed),
        # MaximumObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed),
        # MaximumObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed),
        # MaximumObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed),
        #
        # PreferentialObservedDegreeCrawler(graph, batch=1, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=100, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=1000, initial_seed=initial_seed),
        # PreferentialObservedDegreeCrawler(graph, batch=10000, initial_seed=initial_seed),

        # MultiCrawler(graph, crawlers=[
        #      PreferentialObservedDegreeCrawler(graph, batch=10) for i in range(2)]),

        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, skl_mode=True, batch=100) for i in range(2)]),
        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, skl_mode=True, batch=100) for i in range(5)]),
        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, skl_mode=True, batch=100) for i in range(10)]),
        # MultiCrawler(graph, crawlers=[
        #     MaximumObservedDegreeCrawler(graph, skl_mode=True, batch=100) for i in range(100)]),
    ]
    logging.info([c.name for c in crawlers])
    metrics = []
    target_set = {}  # {i.name: set() for i in statistics}
    for target_statistics in statistics:
        target_set = set(get_top_centrality_nodes(graph, target_statistics, count=int(0.1 * graph[Stat.NODES])))
        # creating metrics and giving callable function to it (here - total fraction of nodes)
        # metrics.append(Metric(r'observed' + target_statistics.name, lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]))
        metrics.append(Metric(r'crawled_' + target_statistics.name,  # TODO rename crawled to observed
                              lambda crawler, t: len(t.intersection(crawler.crawled_set)) / len(t), t=target_set
                              ))

        # print(metrics[-1], target_set)
    if animated == True:
        ci = AnimatedCrawlerRunner(graph,
                                   crawlers,
                                   metrics,
                                   budget=10000, step=10)
    else:
        ci = CrawlerRunner(graph, crawlers, metrics, budget=0,
                           # step=ceil(10 ** (len(str(graph.snap.GetNodes())) - 3)),
                           # if 5*10^5 then step = 10**2,if 10^7 => step=10^4
                           # batches_per_pic=10,
                           # draw_mod='traversal', layout_pos=layout_pos,
                           )  # if you want gifs, draw_mod='traversal'. else: 'metric'
    ci.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # graph_name = 'digg-friends'   # with 261489 nodes and 1536577 edges
    # graph_name = 'douban' #           with 154908 nodes and 327162 edges
    # graph_name = 'ego-gplus' # http://konect.uni-koblenz.de/networks/ego-gplus
    # graph_name = 'slashdot-threads' # N=51083, V=116573.  use step=100,
    # graph_name = 'facebook-wosn-links' #  with 63392 nodes and 816831 edges
    graph_name = 'petster-hamster'  # with 2000 nodes and 16098 edges
    g = GraphCollections.get(graph_name, giant_only=True)

    # graph_name = 'mipt'  #  with 14313 nodes and 488852 edges
    # g = GraphCollections.get(graph_name, 'other', giant_only=True)

    # g._snap_graph = snap.GetMxWcc(g.snap)  # Taking only giant component
    print('Graph {} with {} nodes and {} edges'.format(graph_name, g.snap.GetNodes(), g.snap.GetEdges()))
    # from crawlers.multiseed import test_carpet_graph, MultiCrawler
    # x,y = 7,7
    # graph_name = 'carpet_graph_'+str(x)+'_'+str(y)
    # g, layout_pos = test_carpet_graph(x,y)  # GraphCollections.get(name)
    iterations = 10
    for exp in range(iterations):
        logging.info('Running iteration {}/'.format(exp, iterations))
        test_runner(g,
                    animated=False,
                    statistics=[s for s in Stat if 'DISTR' in s.name],
                    # layout_pos=layout_pos
                    )

    from experiments.merger import merge

    merge(graph_name, show=True)

# 'degree', 'betweenness', 'eccentricity', 'k-coreness',  'pagerank', 'clustering'
