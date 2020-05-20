import os
from time import time

from crawlers.basic import Crawler
from cyth.cbasic import CCrawler
from graph_io import GraphCollections
from utils import GRAPHS_DIR

from cyth.cgraph import CGraph


def test_neighbours():
    # name = 'dolphins'
    # name = 'petster-hamster'
    name = 'ego-gplus'
    cgraph = GraphCollections.cget(name)
    size = cgraph.nodes()
    n = 100

    t = time()
    for _ in range(n):
        for i in cgraph.iter_nodes():
            a = i
        # for i in range(size):
        #     a = [x for x in cgraph.neighbors(i + 1)]
    print("CGraph.neghbors %.3f ms" % ((time()-t)*1000))

    g = GraphCollections.get(name)
    s = g.snap

    t = time()
    for _ in range(n):
        for i in s.Nodes():
            a = i
        # for i in range(size):
        #     a = g.neighbors(i+1)
    print("snap.neighbors %.3f ms" % ((time()-t)*1000))


def test_crawling():
    # name = 'dolphins'
    # name = 'petster-hamster'
    name = 'ego-gplus'
    cgraph = GraphCollections.cget(name)
    ccrawler = CCrawler(cgraph)
    nodes = list(cgraph.iter_nodes())
    n = 10

    t = time()
    for _ in range(n):
        ccrawler = CCrawler(cgraph)
        # for i in cgraph.iter_nodes():
        for i in nodes:
            ccrawler.crawl(i)
    print("CGraph.crawl %.3f ms" % ((time()-t)*1000))

    g = GraphCollections.get(name)
    s = g.snap
    crawler = Crawler(g)

    t = time()
    for _ in range(n):
        crawler = Crawler(g)
        # for i in s.Nodes():
        for i in nodes:
            crawler.crawl(i)
    print("snap.crawl %.3f ms" % ((time()-t)*1000))


if __name__ == '__main__':
    # test_neighbours()
    test_crawling()
