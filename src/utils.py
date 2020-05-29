import os

rel_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep

GRAPHS_DIR = os.path.join(rel_dir, 'data')  # root directory to store all graph data
TMP_GRAPHS_DIR = os.path.join(GRAPHS_DIR, 'tmp')  # root directory to store temporal graphs
PICS_DIR = os.path.join(rel_dir, 'pics')  # directory to store pictures
RESULT_DIR = os.path.join(rel_dir, 'results')  # directory to store pictures

COLLECTIONS = ['other', 'konect', 'netrepo']

USE_CYTHON_CRAWLERS = False  # python/cython mode switcher

if USE_CYTHON_CRAWLERS:
    # Should go before any cython imports. By calling here it is run once
    from cyth.setup import build_cython
    build_cython(rel_dir)
