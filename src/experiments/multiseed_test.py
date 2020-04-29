import glob
import json
import logging
import os

import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from centralities import get_top_centrality_nodes
from crawlers.advanced import AvrachenkovCrawler
# TODO need to finish crawlers and replace into crawlers.basic
from crawlers.basic import PreferentialObservedDegreeCrawler, DepthFirstSearchCrawler, RandomWalkCrawler, \
    RandomCrawler, BreadthFirstSearchCrawler, ForestFireCrawler, MaximumObservedDegreeCrawler
from experiments.runners import make_gif
from graph_io import MyGraph, GraphCollections
from utils import CENTRALITIES


def Crawler_Runner(Graph: MyGraph, crawler_name: str, total_budget=1, step=1,  # n1=1,
                   top_set=True, jsons=False, gif=False, layout_pos=None, ending_sets=False, **kwargs):
    """   # TODO other arguments like top_k=False,

    The core function that takes crawler and does everything with it (crawling budget, export results....)
    :param crawler_name: example of Crawler class (preferably MultiSeedCrawler class )
    :param total_budget:  how many crawling operations we could make
    :param n1:    number of random crawled nodes at the beggining. n1<2 if dont need to.
    :param top_set: dictionary of needed sets of nodes to intersect in history (ex {degree: {1,2,3,},kcore: {1,3,6,}}...)
    :param jsons: if need to export final dumps of crawled and observed sets after ending
    :param pngs:  if need to export pngs  in
    :param gif:   if need to make gif from traversal history every #gif times (ex gif=2 - plot every two iterations
    :param ending_sets: if need to make json files of crawled and observed sets at the end of crawling
    :return:
    """

    if layout_pos is None:
        layout_pos = dict()
    if crawler_name in ('MOD', 'POD'):
        top_k = 5  # TODO do something with it, need to recount every top_k iterations
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph, top_k)  # top_k=kwargs['top_k'])
    else:
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph)


    plt.figure(figsize=(14, 6))
    pngs_path = "../data/gif_files/"

    if gif:  # TODO need to replace directories in utils.py after all and make normal directions
        graph_path = '../data/crawler_history/'
        crawler_history_path = "../data/crawler_history/"

        gif_export_path = '../data/graph_traversal.gif'
        gif_counter = 0
        # for drawing
        # file preparing

        if os.path.exists(pngs_path):
            for file in glob.glob(pngs_path + crawler_name + "*.png"):
                os.remove(file)
        else:
            os.makedirs(pngs_path)

        # with open(crawler_history_path + 'sequence.json', 'r') as f:
        #    sequence = json.load(f)

    # centrality_dict = dict()
    o_count, c_count = {centrality_name: [] for centrality_name in CENTRALITIES}, \
                       {centrality_name: [] for centrality_name in CENTRALITIES}  # history of crawling
    top_dict = dict()
    top_centrality_set_count = int(0.1 * Graph.snap.GetNodes())
    if top_set:  # TODO need to make plots for every centrality (utils.py CENTRALITIES)
        for centrality_name in CENTRALITIES:
            Graph.get_node_property_dict(centrality_name)
            # centrality_dict[centrality_name] = Graph.get_node_property_dict(centrality_name)
            top_dict[centrality_name] = set(get_top_centrality_nodes(Graph, centrality_name,
                                                                     top_centrality_set_count))

    #  history_plots = {centr: [] for centr in centralities}  # list of numbers of intersections for every step
    # plot of crawled history will be just plotting this graph plt.plot( history_plots.)
    # TODO + all node set. it was 'nodes' in our paper

    # if n1:  # crawl first n1 seeds
    #     print('RUNNER: crawl_multi_seed{}'.format(n1))
    #     crawler.crawl_multi_seed(n1=n1)

    for iterator in tqdm(range(0, total_budget, step)):
        # print('RUNNER: iteration:{}/{}'.format(iterator,total_budget))
        crawler.crawl_budget(step)  # file=True)  # TODO return crawling export in files

        if top_set:
            for centrality_name in CENTRALITIES:
                c_count[centrality_name].extend([(iterator, len(
                    top_dict[centrality_name].intersection(crawler.crawled_set)))])  # *step) # if need to add several
                o_count[centrality_name].extend(
                    [(iterator, len(top_dict[centrality_name].intersection(crawler.observed_set)) +
                      c_count[centrality_name][-1][1])])  # *step)

        if gif:  # and (iterator % gif == 0):
            # TODO some strange things coulf happen, because need to give @pos back
            last_seed = crawler.seed_sequence_[-1]
            networkx_graph = crawler.orig_graph.snap_to_networkx

            # coloring nodes
            s = [n.GetId() for n in crawler.orig_graph.snap.Nodes()]
            s.sort()
            gen_node_color = ['gray'] * (max(s) + 1)
            for node in crawler.observed_set:
                gen_node_color[node] = 'y'
            for node in crawler.crawled_set:
                gen_node_color[node] = 'cyan'
            gen_node_color[last_seed] = 'red'
            with_labels = True
            if len(gen_node_color) > 1000:
                with_labels = False

            plt.title(crawler_name + " " + str(iterator) + '  ' + "current node:" + str(last_seed))
            nx.draw(networkx_graph, pos=layout_pos, with_labels=with_labels,
                    node_size=75, node_list=networkx_graph.nodes,
                    node_color=[gen_node_color[node] for node in networkx_graph.nodes]
                    # if node in networkx_graph.nodes()]
                    )
            # plt.xlim(min(axis_x_coords) - 1, max(axis_x_coords) + 1)
            # plt.ylim(min(axis_y_coords) - 1, max(axis_y_coords) + 1)

            plt.savefig(pngs_path + '/gif{}{}.png'.format(crawler_name, str(iterator).zfill(3)))
            plt.pause(0.001)
            plt.cla()

        # finishes when see all nodes. it means even if they are only obsered, we know most part of degrees
        if len(crawler.observed_set) + len(crawler.crawled_set) == crawler.orig_graph.snap.GetNodes():
            print("Finished coz crawled+observed all")
            break

    # drawing_graph.draw_new_png(crawler.observed_graph, crawler.observed_set, crawler.crawled_set,
    #                          iterator, last_seed=crawler.seed_sequence_[-1], pngs_path=pngs_path,
    #                           crawler_name = crawler_name, labels = False)
    # if top_set: # TODO every (successful) crawling iteration need to calculate intersection and write in history_plots

    # # TODO uncomment this or do something with empty node numbers (ex. nodes starts from 1, or does not exist)
    # nx_graph = crawler.observed_graph.networkx_graph  # snap_to_nx_graph(snap_graph)
    # # node_list = list(nx_graph.nodes())
    # print("----", crawler.observed_set, crawler.crawled_set)
    # for node in range(max(nx_graph.nodes())):  # filling empty indexes in graph
    #     if node not in nx_graph.nodes():
    #         nx_graph.add_node(1)
    #         print('i added node', node)
    if top_set:
        file_path = "../data/crawler_history/"  # TODO do something with export
        if os.path.exists(file_path):
            for file in glob.glob("../data/crawler_history/*{}*.json".format(crawler_name)):
                os.remove(file)
        else:
            os.makedirs(file_path)

        for centrality_name in CENTRALITIES:
            with open(
                    "../data/crawler_history/crawled_history_{}_{}.json".format(crawler.crawler_name, centrality_name),
                    'w') as top_set_file:
                json.dump([(x, y / top_centrality_set_count) for x, y in c_count[centrality_name]], top_set_file)
        for centrality_name in CENTRALITIES:
            with open(
                    "../data/crawler_history/observed_history_{}_{}.json".format(crawler.crawler_name, centrality_name),
                    'w') as top_set_file:
                json.dump([(x, y / top_centrality_set_count) for x, y in c_count[centrality_name]], top_set_file)

    if gif:
        make_gif(crawler_name=crawler_name, pngs_path=pngs_path)  # , duration=step)

    print(crawler_name + ": after first: crawled {}: {},".format(len(crawler.crawled_set), crawler.crawled_set),
          " observed {}: {}".format(len(crawler.observed_set), crawler.observed_set))

    # dumps final versions of crawled_set and observed_set after finishing crawling all budget.
    if ending_sets:
        with open("../data/crawler_history/crawled{}{}.json".format(crawler.crawler_name,
                                                                    str(len(crawler.seed_sequence_)).zfill(6)),
                  'a') as cr_file:
            json.dump(list(crawler.crawled_set), cr_file)
        with open("../data/crawler_history/observed{}{}.json".format(crawler.crawler_name,
                                                                     str(len(crawler.seed_sequence_)).zfill(6)),
                  'a') as ob_file:
            json.dump(list(crawler.observed_set), ob_file)

    return crawler


# using simple dictionary to work with crawlers. aAlso .keys of it are names of crawlers for files
CRAWLERS_DICTIONARY = {'POD': PreferentialObservedDegreeCrawler,
                       'MOD': MaximumObservedDegreeCrawler,
                       'DFS': DepthFirstSearchCrawler,
                       'BFS': BreadthFirstSearchCrawler,
                       'RWC': RandomWalkCrawler,
                       'RC_': RandomCrawler,
                       'FFC': ForestFireCrawler,
                       'AVR': AvrachenkovCrawler,  # dont use, it iscompletely different
                       }


def test_crawlers(Graph: MyGraph, total_budget=100, crawlers=None, layout_pos=None, gif=False, step=1):
    if crawlers is None:
        crawlers = ['DFS']
    file_path = "../data/crawler_history/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = "../data/gif_files/"  # TODO do something with export
    if os.path.exists(file_path):
        for file in glob.glob("../data/gif_files/*.png"):
            os.remove(file)
    else:
        os.makedirs(file_path)

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    print("N=%s E=%s" % (Graph.snap.GetNodes(), Graph.snap.GetEdges()))

    n1 = 1
    total_budget = min(total_budget - n1, Graph.snap.GetNodes())

    # pos = None  # position layout for drawing similar graphs (with nodes on same positions). updates at the end

    # crawlers = [(name, crawlers_dictionary[name]) for name in crawlers_dictionary]
    result_crawlers = []
    for crawler_name in crawlers:
        print("Running {} with budget={}, n1={}".format(crawler_name, total_budget, n1))
        result_crawlers.append(Crawler_Runner(Graph, crawler_name, total_budget=total_budget,
                                              gif=gif, step=step,  # n1=n1,
                                              layout_pos=layout_pos
                                              ))
        print("Seed sequence due crawling:", result_crawlers[-1].seed_sequence_)
    # with open("./data/crawler_history/sequence.json", 'w') as f:
    #     json.dump(crawler.seed_sequence_, f)
    return result_crawlers


def add_nodes_for_networkx(Graph: MyGraph):
    graph_nodes = [n.GetId() for n in Graph.snap.Nodes()]
    for node in range(max(graph_nodes) + 1):
        if node not in graph_nodes:
            Graph.snap.AddNode(node)
            Graph.snap.AddEdge(node, 0)
    print('nodes', [n.GetId() for n in Graph.snap.Nodes()])


if __name__ == '__main__':
    layout_pos = None
    Graph = GraphCollections.get('petster-hamster')  # petster-hamster')  #   #
    add_nodes_for_networkx(Graph)  # TODO little костыль for normal graph drawing
    # layout_pos = nx.spring_layout(Graph.snap_to_networkx, iterations=100)
    ####Graph = MyGraph.new_snap(g.snap, name='test', directed=False)
    # g, layout_pos = test_carpet_graph(10, 8)
    # Graph = MyGraph.new_snap(g.snap, name='test', directed=False)
    print(Graph.snap.GetNodes())
    if layout_pos is None:
        print('need to draw layout pos', layout_pos)
        layout_pos = nx.spring_layout(Graph.snap_to_networkx, iterations=40)

    test_crawlers(Graph, 11000, ['RWC'],
                  gif=True, step=400, layout_pos=layout_pos, )
# crawlers = ['DFS']  # , 'BFS', 'RWC', 'RC_', 'FFC', ]  # 'MOD', 'POD',
