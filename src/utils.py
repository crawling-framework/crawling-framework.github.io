import os

rel_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep

GRAPHS_DIR = os.path.join(rel_dir, 'data')  # root directory to store all graph data
TMP_GRAPHS_DIR = os.path.join(GRAPHS_DIR, 'tmp')  # root directory to store temporal graphs
PICS_DIR = os.path.join(rel_dir, 'pics')  # directory to store pictures
RESULT_DIR = os.path.join(rel_dir, 'results')  # storing results
# os.path.join(PICS_DIR, )

COLLECTIONS = ['konect', 'networkrepository']

CENTRALITIES = ['degree', 'betweenness', 'eccentricity', 'k-coreness', 'pagerank', 'clustering']  # 'closeness',]
