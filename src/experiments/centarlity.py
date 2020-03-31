from centralities import get_top_centrality_nodes
from graph_io import GraphCollections

if __name__ == '__main__':
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # logging.getLogger().setLevel(logging.INFO)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    name = 'ego-gplus'
    # name = 'petster-hamster'
    graph = GraphCollections.get(name)
    degs = get_top_centrality_nodes(graph, 'degree', 10)
    print(degs)
    btws = get_top_centrality_nodes(graph, 'betweenness', 10)
    print(btws)

    # plt.show()
