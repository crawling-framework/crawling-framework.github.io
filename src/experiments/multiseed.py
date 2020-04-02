import logging
from operator import itemgetter

import snap
import numpy as np
import matplotlib.pyplot as plt

from crawlers import Crawler, AvrachenkovCrawler
from centralities import get_top_centrality_nodes
from graph_io import MyGraph, GraphCollections


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

# name = 'libimseti'
# name = 'petster-friendships-cat'
# name = 'soc-pokec-relationships'
# name = 'digg-friends'
# name = 'loc-brightkite_edges'
# name = 'ego-gplus'

name = 'petster-hamster'
g = GraphCollections.get(name)



#plt.show()