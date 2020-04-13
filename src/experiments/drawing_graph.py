import glob

import imageio
from tqdm import tqdm


# def draw_new_png(graph: MyGraph, observed_set, crawled_set, iterator, last_seed,
#                  pngs_path, crawler_name='', labels=False):
#     # snap_graph = snap.LoadEdgeList(snap.PUNGraph, graph_path + 'orig_graph.txt', 0, 1, '\t')
#
#     gen_node_color = ['gray'] * graph.networkx_graph.number_of_nodes()
#
#     # print("nodes in nx", nx_graph.nodes())
#
#     # with open(crawler_history_path + "observed{}{}.json".format(crawler_name, str(iterator).zfill(6)), 'r') as f:
#         # observed = json.load(f)
#     for node in observed_set:
#         print("draw-", len(gen_node_color), gen_node_color, node)
#         gen_node_color[node] = 'y'
#
#     # with open(crawler_history_path + "crawled{}{}.json".format(crawler_name, str(iterator).zfill(6)), 'r') as f:
#     #    crawled = json.load(f)
#     for node in crawled_set:
#         gen_node_color[node] = 'cyan'
#
#     gen_node_color[last_seed] = 'red'
#     plt.title(str(iterator) + '/' + "cur:" + str(last_seed) + " craw:" + str(crawled_set))
#     nx.draw(graph.networkx_graph, pos=graph.graph_layout_pos, with_labels=labels, node_size=80,
#             node_color=[gen_node_color[node] for node in graph.networkx_graph.nodes])
#     plt.savefig(pngs_path + '/gif{}{}.png'.format(crawler_name, str(iterator).zfill(3)))
#     plt.clf()


def make_gif(crawler_name, pngs_path):
    images = []
    duration = 5
    filenames = glob.glob(pngs_path + "gif{}*.png".format(crawler_name))
    filenames.sort()
    print("adding")
    print(filenames)
    for filename in tqdm(filenames):
        for i in range(duration):
            images.append(imageio.imread(filename))
    print("compiling")
    imageio.mimsave(pngs_path + "result_{}.gif".format(crawler_name), images)
    print("done")

    # return pos

# graph_path = './data/crawler_history/'
# crawler_history_path = "./data/crawler_history/sequence.json"
# pngs_path = "./data/gif_files/"
# gif_export_path = './data/graph_traversal.gif'


# def draw_crawling_curve(crawler_history_path, )
