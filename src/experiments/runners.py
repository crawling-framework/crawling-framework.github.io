import datetime
import glob
import json
import os

import imageio
import networkx as nx
import snap
from matplotlib import pyplot as plt
from tqdm import tqdm

from centralities import get_top_centrality_nodes
from crawlers.basic import MaximumObservedDegreeCrawler, DepthFirstSearchCrawler, RandomWalkCrawler, \
    PreferentialObservedDegreeCrawler, BreadthFirstSearchCrawler, RandomCrawler, Crawler, ForestFireCrawler
from graph_io import GraphCollections
from graph_io import MyGraph
from statistics import Stat
from utils import PICS_DIR, RESULT_DIR  # PICS_DIR = '/home/jzargo/PycharmProjects/crawling/crawling/pics/'


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
    def __init__(self, name, callback):
        self.name = name
        self._callback = callback

    def __call__(self, crawler: Crawler):
        return self._callback(crawler)


class AnimatedCrawlerRunner:
    def __init__(self, graph: MyGraph, crawlers, metrics, budget=-1, step=1, target_metric='degree',
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
        self.target_metric = target_metric  # for naming graph
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
            plt.figure("Graph %s:  N=%d, E=%d, d_max=%d" % (
                self.graph.name, graph[Stat.NODES], graph[Stat.EDGES], graph[Stat.MAX_DEGREE]),
                       figsize=(20, 10))  # (1 + scale * self.ncols, scale * self.nrows), )

        else:  # if it is traversal

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

    def save_history(self, crawler_metric_seq):
        for crawler_name in crawler_metric_seq.keys():
            crawler_seq = crawler_metric_seq[crawler_name]
            for metric_object in crawler_seq.keys():
                metric_list = crawler_seq[metric_object]
                file_path = os.path.join(RESULT_DIR, self.graph.name, 'budget={}'.format(len(crawler_metric_seq)),
                                         crawler_name.name, )
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name = os.path.join(file_path, metric_object.name + '.json')  # TODO test
                if os.path.isfile(file_name):
                    expand = 1
                    while True:
                        expand += 1
                        new_file_name = file_name.split(".json")[0] + str(expand) + ".json"
                        if os.path.isfile(new_file_name):
                            continue
                        else:
                            file_name = new_file_name
                            break

                with open(file_name, 'w') as f:
                    json.dump(metric_list, f)

    def save_pics(self):
        if self.draw_mod == 'metric':
            plt.legend()
            plt.title(self.target_metric)
            file_path = os.path.join(RESULT_DIR, self.graph.name, 'crawling_plot')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(os.path.join(file_path, self.target_metric + ':' +
                                     ','.join([crawler.name for crawler in self.crawlers]) + '.png'))
        else:  # if traversal
            for crawler in self.crawlers:
                make_gif(crawler_name=crawler.name, duration=2)
                logging.info('compiled +')

    def run(self):
        linestyles = ['-', '--', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']  # for different methods

        step_seq = [0]  # starting for point 0
        crawler_metric_seq = dict([(c, dict([(m, [0]) for m in self.metrics])) for c in self.crawlers])

        i = 0
        logging.info('Crawling with budget {} and step {}'.format(self.budget, self.step))
        pbar = tqdm(total=self.budget)  # drawing crawling progress bar
        while i < self.budget:
            batch = min(self.step, self.budget - i)
            i += batch
            step_seq.append(i)
            pbar.update(batch)

            if i % 1000 == 0:  # making backup every 10k
                self.save_history(crawler_metric_seq)
                self.save_pics()
                logging.info('\n{} : Iteration {} made backup of pics and history'.format(datetime.datetime.now(), i))

            for c, crawler in enumerate(self.crawlers):
                crawler.crawl_budget(budget=batch)
                for m, metric in enumerate(self.metrics):
                    metric_seq = crawler_metric_seq[crawler][metric]
                    metric_seq.append(metric(crawler))  # calculate metric for crawler
                    if (self.draw_mod == 'metric') and (i % self.batches_per_pic == 0):
                        plt.plot(step_seq, metric_seq, marker='.',  # TODO remove extra labels
                                 linestyle=linestyles[m % len(linestyles)],
                                 color=colors[c % len(colors)],
                                 label=r'%s, %s' % (crawler.name, metric.name))
                        # else:
                        #     plt.plot(step_seq, metric_seq, marker='.',
                        #              linestyle=linestyles[m % len(linestyles)],
                        #              color=colors[c % len(colors)])
                        plt.xlabel('iteration, n')
                        plt.ylabel('metric value')


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
                                #  with_labels=(len(gen_node_color) < 1000),  # if little, we draw
                                node_size=100, node_list=networkx_graph.nodes,
                                node_color=[gen_node_color[node] for node in networkx_graph.nodes])
                        file_path = os.path.join(PICS_DIR, self.graph.name)
                        plt.savefig(file_path + '/{}_{}.png'.format(crawler.name, str(i).zfill(5)))
                        plt.cla()
                        # plt.show()

        pbar.close()  # closing progress bar
        self.save_history(crawler_metric_seq)  # saving ending history
        self.save_pics()
        # plt.show()
        logging.info('{} : finished running'.format(datetime.datetime.now()))


def test_runner(graph, target_metric=None, layout_pos=None):
    import random
    initial_seed = random.sample([n.GetId() for n in graph.snap.Nodes()], 1)[0]
    crawlers = [
        # MultiCrawler(graph, crawlers=[
        #     DepthFirstSearchCrawler(graph, batch=1, initial_seed=0),
        #     DepthFirstSearchCrawler(graph, batch=1, initial_seed=10),
        # ]),
        DepthFirstSearchCrawler(graph, initial_seed=initial_seed),
        ForestFireCrawler(graph, initial_seed=initial_seed),
        RandomWalkCrawler(graph, initial_seed=initial_seed),
        MaximumObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed),
        PreferentialObservedDegreeCrawler(graph, batch=10, initial_seed=initial_seed),
        BreadthFirstSearchCrawler(graph, initial_seed=initial_seed),
        RandomCrawler(graph, initial_seed=initial_seed),
    ]
    logging.info(crawlers[0].name)
    # change target set to calculate another metric
    target_set = set(get_top_centrality_nodes(graph, target_metric, count=int(0.1 * graph[Stat.NODES])))
    metrics = [  # creating metrics and giving callable function to it (here - total fraction of nodes)
        #  Metric(r'observed_degree', lambda crawler: len(crawler.nodes_set) / graph[Stat.NODES]),
        Metric(r'crawled_' + target_metric,
               lambda crawler: len(target_set.intersection(crawler.nodes_set)) / len(target_set)),
    ]

    ci = AnimatedCrawlerRunner(graph, crawlers, metrics, budget=int(graph.snap.GetNodes() / 10), step=50,
                               batches_per_pic=10,
                               draw_mod='metric', layout_pos=layout_pos, target_metric=target_metric,
                               )  # if you want gifs, draw_mod='traversal'. else: 'metric'
    ci.run()


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    # name = 'petster-hamster'
    # name = 'slashdot-threads'  # 'dolphins' #  #''
    name = 'slashdot-threads'  # 'petster-hamster' # 'advogato'
    g = GraphCollections.get(name)
    g._snap_graph = snap.GetMxWcc(g.snap)  # Taking only giant component
    # all about sexgraph
    # name = 'sexgraph'
    # snap_g = TUNGraph.New()
    # pos = dict()
    # nx_graph = nx.read_edgelist('/home/jzargo/PycharmProjects/crawling/crawling/data/Graphs/edgelist_sexgraph.json')
    # for i in nx_graph.nodes():
    #     snap_g.AddNodeUnchecked(int(i))
    # for i, j in nx_graph.edges():
    #     snap_g.AddEdgeUnchecked(int(i), int(j))
    # g = MyGraph.new_snap(snap_g, name='test', directed=False)
    # nx.draw_spring(g.snap_to_networkx)
    # plt.plot()

    # from crawlers.multiseed import test_carpet_graph, MultiCrawler
    # name = 'carpet_graph'
    # g, layout_pos = test_carpet_graph(10, 10)  # GraphCollections.get(name)
    logging.info("running graph ".format(name))
    from utils import CENTRALITIES

    for exp in range(4):
        for metric in CENTRALITIES:
            print('running', metric)
            test_runner(g,
                        target_metric=metric
                        # layout_pos
                        )
# target_metric = 'degree', 'betweenness', 'eccentricity', 'k-coreness',  'pagerank', 'clustering'
