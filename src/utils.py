from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()  # directory of source root

CONFIG_PATH = root_dir / 'config'  # config file with flags anf paths

GRAPHS_DIR = root_dir / 'data'  # root directory to store all graph data
PICS_DIR = root_dir / 'pics'  # directory to store pictures
RESULT_DIR = root_dir / 'results'  # directory to store pictures
DATASET_DIR = root_dir / 'datasets'  # directory to store generated datasets
STATISTIC_DIR = root_dir / 'statistic'

LFR_DIR = root_dir / 'soft' / 'lfr' / 'benchmark'  # path to LFR binaries
SNAP_DIR = root_dir / 'soft' / 'snap'  # path to LFR binaries

# Should go before any cython imports. By calling here it is run once
from setup import build_cython
build_cython(root_dir, SNAP_DIR)
