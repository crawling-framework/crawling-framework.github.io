import os

rel_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep

CONFIG_PATH = os.path.join(rel_dir, 'config')  # config file with flags anf paths

GRAPHS_DIR = os.path.join(rel_dir, 'data')  # root directory to store all graph data
TMP_GRAPHS_DIR = os.path.join(GRAPHS_DIR, 'tmp')  # root directory to store temporal graphs
PICS_DIR = os.path.join(rel_dir, 'pics')  # directory to store pictures
RESULT_DIR = os.path.join(rel_dir, 'results')  # directory to store pictures

COLLECTIONS = ['other', 'konect', 'netrepo']

SNAP_DIR = None; LIGRA_DIR = None; LFR_DIR = None; USE_NETWORKIT = None; USE_LIGRA = None; VK_ID = None  # defined in config
config = exec(open(CONFIG_PATH, 'r').read())  # updates the above VARIABLES from config file


# Should go before any cython imports. By calling here it is run once
from cyth.setup import build_cython
build_cython(rel_dir, SNAP_DIR)
