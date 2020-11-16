import glob
import logging
import os
import shutil

import imageio
import matplotlib.pyplot as plt
import networkx as nx
from base.cgraph import MyGraph
from crawlers.cadvanced import CrawlerWithAnswer
from crawlers.cbasic import definition_to_filename
from tqdm import tqdm

from graph_stats import Stat, get_top_centrality_nodes
from running.metrics_and_runner import CrawlerRunner, TopCentralityMetric
from utils import PICS_DIR


def create_gif(directory, duration=1):
    """ Create gif from all .png files in directory with fixed duration between images.
    """
    images = []
    filenames = glob.glob(os.path.join(directory, "*.png"))
    filenames.sort()
    for filename in tqdm(filenames, desc="Making gif"):
        images.append(imageio.imread(filename))
    file_path = os.path.join(directory, "gif.gif")
    imageio.mimsave(file_path, images, duration=0.2 * duration, loop=2)


class CrawlerVisualRunner(CrawlerRunner):
    """
    Visualize the crawling process step by step on a small graph.
    Graph nodes and edges are plotted in colors corresponding to their state: crawled, observed,
    target, etc.
    Saves .png pictures to the disk.
    """
    def __init__(self, graph: MyGraph, crawler_def, metric_def, budget: int = 100, step: int = 1,
                 steps_per_pic=1, layout_pos=None):
        """
        :param graph: graph to run
        :param crawler_def: crawler definitions to run. Crawler definition will be initialized when
         run() is called
        :param metric_def: metric definition to compute at each step. Metric should be callable
         function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default 100
        :param step: compute metrics each `step` steps, by default 1
        :param steps_per_pic: plot every this number of steps, by default 1
        :param layout_pos: specific layout for networkx drawer
        :return:
        """
        assert step > 0
        if graph[Stat.NODES] > 1000:
            logging.warning("Graphs with > 1000 nodes are too big to be plotted.")
        super().__init__(graph, crawler_defs=[crawler_def], metric_defs=[metric_def], budget=budget, step=step)

        self.steps_per_pic = steps_per_pic
        self.layout_pos = layout_pos

    def _save_dir(self):
        return os.path.join(PICS_DIR, 'visual', self.graph.name)

    def run(self, draw_orig=True, target_set=set(), labels=False, bold_edges=False, make_gif=False):
        """
        Run crawler and plot graph with nodes colored. All plots are saved in a series of png files.
        Coloring depends on node type: unobserved (gray), observed (yellow), current (red), crawled (blue).
        Target nodes are labeled by bigger orange circles.

        :param draw_orig: whether to draw original graph
        :param target_set: these nodes will be highlighted (bigger)
        :param labels: whether to draw node labels (ids)
        :param bold_edges: if True draw observed edges in bold, unobserved edges dotted
        :param make_gif: whether to create gif from a png series.
        :return:
        """
        crawlers, metrics, batch_generator = self._init_runner()
        crawler = crawlers[0]
        metric = metrics[0]
        nx_orig_graph = self.graph.snap_to_networkx()

        # Define layout via networkx
        if self.layout_pos is None:
            logging.info("Computing networkx layout...")
            # self.layout_pos = nx.spring_layout(nx_orig_graph, iterations=40)
            self.layout_pos = nx.kamada_kawai_layout(nx_orig_graph)
            logging.info("done.")

        # Drawing parameters
        other_color = 'lightgray'
        observed_color = 'yellow'
        crawled_color = 'blue'
        current_color = 'red'
        target_color = 'orange'
        node_size = 250 if labels else 50
        target_node_size = node_size + 150
        bold_width = 1  # width of observed edges
        last_crawled = set()

        # Handle directory to save pics to
        save_dir = self._save_dir()
        for c in crawlers:
            d = os.path.join(save_dir, definition_to_filename(c.definition))
            if not os.path.exists(d):
                os.makedirs(d)
            else:
                shutil.rmtree(d)
                os.makedirs(d)

        plt.figure("Graph %s" % self.graph.name, figsize=(16, 12))
        pbar = tqdm(total=self.budget, desc='Running iterations')  # drawing crawling progress bar

        step = 0
        for batch in batch_generator:
            step += batch

            # crawler.crawl_budget(batch)
            last_crawled.clear()
            for i in range(batch):
                seed = crawler.next_seed()
                crawler.crawl(seed)
                # Memorize crawled seeds to draw them later
                last_crawled.add(seed)
            if isinstance(crawler, CrawlerWithAnswer):
                # Recompute actual answer
                _ = crawler.answer

            # Calculate metric for crawler
            value = metric(crawler)

            # Draw graph via networkx
            if step % self.steps_per_pic == 0:
                plt.cla()

                # Draw target set nodes of bigger size
                if len(target_set) > 0:
                    layout_pos = {n: self.layout_pos[n] for n in target_set}
                    nx.draw(nx_orig_graph, pos=layout_pos, nodelist=list(target_set), edgelist=[],
                            node_size=[target_node_size] * len(target_set), node_color=target_color)

                # Draw orig graph with node coloring
                if draw_orig:
                    node_colors = []  # nodes colorings
                    for node in nx_orig_graph.nodes:
                        if node in last_crawled:
                            node_colors.append(current_color)
                        elif node in crawler.observed_set:
                            node_colors.append(observed_color)
                        elif node in crawler.crawled_set:
                            node_colors.append(crawled_color)
                        elif node in target_set:
                            node_colors.append(target_color)
                        else:
                            node_colors.append(other_color)

                    nx.draw(nx_orig_graph, pos=self.layout_pos,
                            with_labels=labels,
                            node_size=node_size, node_list=nx_orig_graph.nodes,
                            node_color=node_colors, edge_color='black',
                            style='dotted' if bold_edges else 'solid',
                            alpha=0.4 if bold_edges else 1)

                # Draw observed graph edges in bold
                if bold_edges:
                    nodes_set = crawler.nodes_set
                    node_colors = []  # nodes colorings
                    for node in nodes_set:
                        if node in last_crawled:
                            node_colors.append(current_color)
                        elif node in crawler.crawled_set:
                            node_colors.append(crawled_color)
                        else:
                            node_colors.append(observed_color)

                    nx.draw(nx_orig_graph, pos=self.layout_pos,
                            with_labels=labels,
                            node_size=node_size, node_color=node_colors,
                            nodelist=list(nodes_set),
                            edgelist=list(crawler._observed_graph.iter_edges()), width=bold_width)

                plt.title("Graph: %s. nodes: %s, edges: %s\n"
                          "Crawler: %s. crawled: %s, observed: %s\n"
                          "Metric: %s. Value: %.4f" % (
                    self.graph.name, self.graph.nodes(), self.graph.edges(),
                    crawler.name, len(crawler.crawled_set), len(crawler.observed_set),
                    metric.name, value))
                plt.tight_layout()
                plt.pause(0.005)

                # fig.set_size_inches(25, 22, forward=False)
                plt.savefig(os.path.join(save_dir, definition_to_filename(crawler.definition), "%05.f" % step))

            pbar.update(batch)

        pbar.close()  # closing progress bar
        if make_gif:
            for c in crawlers:
                create_gif(os.path.join(save_dir, definition_to_filename(c.definition)), duration=2)
        plt.show()


def test_visual_runner():
    from crawlers.cbasic import MaximumObservedDegreeCrawler
    from graph_io import GraphCollections
    # g = GraphCollections.get('dolphins')
    g = GraphCollections.get('PDZBase')
    # g = GraphCollections.get('Infectious')
    # g = GraphCollections.get('soc-wiki-Vote')
    # g = GraphCollections.get('Jazz musicians')
    # print(g[Stat.PLM_MODULARITY])

    p = 0.1

    crawler_def = (MaximumObservedDegreeCrawler, {'initial_seed': 1})
    # crawler_def = (MaximumObservedCommunityDegreeCrawler, {'initial_seed': 1})
    # crawler_def = (ThreeStageMODCrawler, {'s': 10, 'n': g.nodes(), 'p': p, 'b': 1})

    metric_def = (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'})
    # metric_def = (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'answer'})
    target_set = get_top_centrality_nodes(g, Stat.DEGREE_DISTR, count=int(p*g.nodes()))

    cvr = CrawlerVisualRunner(g, crawler_def, metric_def, budget=g.nodes(), step=1)
    cvr.run(draw_orig=True, bold_edges=True, labels=False, make_gif=False, target_set=target_set)


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    test_visual_runner()
