import glob
import json
import logging
import os

from tqdm import tqdm

from crawlers.advanced import AvrachenkovCrawler
# from crawlers.basic import  # TODO need to finish crawlers and replace into crawlers.basic
from crawlers.multiseed import PreferentialObservedDegreeCrawler, MaximumObservedDegreeCrawler, \
    DepthFirstSearchCrawler, BreadthFirstSearchCrawler, RandomWalkCrawler, RandomCrawler, \
    ForestFireCrawler, test_carpet_graph
from experiments import drawing_graph
from graph_io import MyGraph


def Crawler_Runner(Graph: MyGraph, crawler_name: str, total_budget=1, n1=1,
                   top_set=False, jsons=False, gif=0, layout_pos=None, ending_sets=False, **kwargs):
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
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph, top_k=kwargs['top_k'])
    else:
        crawler = CRAWLERS_DICTIONARY[crawler_name](Graph)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(14, 6))
    pngs_path = "../data/gif_files/"

    if gif:  # TODO need to replace directories in utils.py after all and make normal directions
        graph_path = '../data/crawler_history/'
        crawler_history_path = "../data/crawler_history/"

        gif_export_path = '../data/graph_traversal.gif'
        gif_counter = 0
        # for drawing

        # print(layout_pos)
        # print([i for i in layout_pos])
        # fig, ax = plt.subplots()
        # print("lp",[layout_pos[v][0] for v in layout_pos])
        # axis_x_coords = [layout_pos[v][0] for v in layout_pos]
        # axis_y_coords = [layout_pos[v][1] for v in layout_pos]

        # file preparing
        if os.path.exists(pngs_path):
            for file in glob.glob(pngs_path + crawler_name + "*.png"):
                os.remove(file)
        else:
            os.makedirs(pngs_path)

        # with open(crawler_history_path + 'sequence.json', 'r') as f:
        #    sequence = json.load(f)

    if top_set:  # TODO need to make plots for every centrality (utils.py CENTRALITIES)
        top_set = dict()
        centralities = top_set.keys()
        history_plots = {centr: [] for centr in centralities}  # list of numbers of intersections for every step
        # plot of crawled history will be just plotting this graph plt.plot( history_plots.)
        # TODO + all node set. it was 'nodes' in our paper

    if n1:  # crawl first n1 seeds
        print('RUNNER: crawl_multi_seed{}'.format(n1))
        crawler.crawl_multi_seed(n1=n1)

    for iterator in tqdm(range(total_budget)):
        # print('RUNNER: iteration:{}/{}'.format(iterator,total_budget))
        crawler.crawl_budget(1)  # file=True)  # TODO return crawling export in files

        if gif and (iterator % gif == 0):
            from networkx import draw
            # TODO some strange things coulf happen, because need to give @pos back
            last_seed = crawler.seed_sequence_[-1]
            networkx_graph = crawler.orig_graph.snap_to_networkx

            # coloring nodes
            s = [n.GetId() for n in crawler.orig_graph.snap.Nodes()]
            s.sort()
            gen_node_color = ['gray'] * (max(s) + 1)
            for node in crawler.observed_set:
                #    print(node, gen_node_color)
                gen_node_color[node] = 'y'
            for node in crawler.crawled_set:
                gen_node_color[node] = 'cyan'
            gen_node_color[last_seed] = 'red'

            plt.title(crawler_name + " " + str(iterator) + '  ' + "current node:" + str(last_seed))
            if layout_pos is None:
                layout_pos = Graph.snap.snap_to_networkx.spring_layout(Graph.snap_to_networkx, iterations=100)
            draw(networkx_graph, pos=layout_pos, with_labels=True, node_size=150,
                 node_color=[gen_node_color[node] for node in networkx_graph.nodes]
                 # if node in networkx_graph.nodes()]
                 )
            # plt.xlim(min(axis_x_coords) - 1, max(axis_x_coords) + 1)
            # plt.ylim(min(axis_y_coords) - 1, max(axis_y_coords) + 1)

            plt.savefig(pngs_path + '/gif{}{}.png'.format(crawler_name, str(iterator).zfill(3)))
            plt.clf()

        # finishes when see all nodes. it means even if they are only obsered, we know most part of degrees
        if len(crawler.observed_set) + len(crawler.crawled_set) == crawler.orig_graph.snap.GetNodes():
            print("Finished coz crawled+observed all")
            break

    # drawing_graph.draw_new_png(crawler.observed_graph, crawler.observed_set, crawler.crawled_set,
    #                          iterator, last_seed=crawler.seed_sequence_[-1], pngs_path=pngs_path,
    #                           crawler_name = crawler_name, labels = False)
    # if top_set: # TODO every (successful) crawling iteration need to calculate intersection and write in history_plots

    # TODO uncomment this or do something with empty node numbers (ex. nodes starts from 1, or does not exist)
    # nx_graph = crawler.observed_graph.networkx_graph  # snap_to_nx_graph(snap_graph)
    # # node_list = list(nx_graph.nodes())
    # print("----", crawler.observed_set, crawler.crawled_set)
    # for node in range(max(nx_graph.nodes())):  # filling empty indexes in graph
    #     if node not in nx_graph.nodes():
    #         nx_graph.add_node(1)
    #         print('i added node', node)

    if gif:
        drawing_graph.make_gif(crawler_name=crawler_name, pngs_path=pngs_path)

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


def test_crawlers():
    file_path = "../data/crawler_history/"
    if os.path.exists(file_path):
        for file in glob.glob("../data/crawler_history/*.json"):
            os.remove(file)
    else:
        os.makedirs(file_path)

    file_path = "../data/gif_files/"
    if os.path.exists(file_path):
        for file in glob.glob("../data/gif_files/*.png"):
            os.remove(file)
    else:
        os.makedirs(file_path)

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    #### g = GraphCollections.get('dolphins').snap  # test_graph()  #
    #### layout_pos = nx.spring_layout(Graph.snap_to_networkx, iterations=100)
    ####Graph = MyGraph.new_snap(g.snap, name='test', directed=False)
    g, layout_pos = test_carpet_graph(9, 7)
    Graph = MyGraph.new_snap(g.snap, name='test', directed=False)

    # Graph.snap.AddNode(0)
    # Graph.snap.AddEdge(0, 1)

    print("N=%s E=%s" % (Graph.snap.GetNodes(), Graph.snap.GetEdges()))

    # min(100,Graph.snap.GetNodes())
    n1 = 1
    total_budget = min(100 - n1, Graph.snap.GetNodes())
    top_k = 5
    k = 4
    # pos = None  # position layout for drawing similar graphs (with nodes on same positions). updates at the end

    # crawlers = [(name, crawlers_dictionary[name]) for name in crawlers_dictionary]
    crawlers = ['DFS']  # , 'BFS', 'RWC', 'RC_', 'FFC', ]  # 'MOD', 'POD',

    for crawler_name in crawlers:
        print("Running {} with budget={}, n1={}".format(crawler_name, total_budget, n1))
        crawler = Crawler_Runner(Graph, crawler_name, total_budget=total_budget, n1=n1,
                                 gif=1, layout_pos=layout_pos)
        print("Seed sequence due crawling:", crawler.seed_sequence_)
    # with open("./data/crawler_history/sequence.json", 'w') as f:
    #     json.dump(crawler.seed_sequence_, f)


if __name__ == '__main__':
    test_crawlers()
