import logging

import matplotlib.pyplot as plt

from crawlers.basic import MaximumObservedDegreeCrawler
from graph_io import GraphCollections


def test_mod(graph):
    crawler = MaximumObservedDegreeCrawler(graph, batch=10)
    # crawler = BreadthFirstSearchCrawler(graph)
    # crawler = RandomCrawler(graph)

    iterations = []
    os = []
    ns = []

    step = 1
    for i in range(1, 100, step):
        iterations.append(i)
        crawler.crawl_budget(step)
        os.append(len(crawler.observed_set))
        ns.append(len(crawler.nodes_set))

        plt.cla()
        plt.plot(iterations, os, marker='o', color='g', label='observed_set')
        plt.plot(iterations, ns, marker='o', color='b', label='nodes_set')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.pause(0.005)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # name = 'libimseti'
    # name = 'petster-friendships-cat'
    # name = 'soc-pokec-relationships'
    # name = 'digg-friends'
    # name = 'loc-brightkite_edges'
    # name = 'ego-gplus'
    name = 'petster-hamster'
    g = GraphCollections.get(name)

    # test_mod(g)
