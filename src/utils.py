import os

rel_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep

SNAP_DIR = "/home/misha/Snap-5.0/"  # directory with snap built

GRAPHS_DIR = os.path.join(rel_dir, 'data')  # root directory to store all graph data
TMP_GRAPHS_DIR = os.path.join(GRAPHS_DIR, 'tmp')  # root directory to store temporal graphs
PICS_DIR = os.path.join(rel_dir, 'pics')  # directory to store pictures
RESULT_DIR = os.path.join(rel_dir, 'results')  # directory to store pictures

COLLECTIONS = ['other', 'konect', 'networkrepository']

USE_CYTHON_CRAWLERS = True


# Remaping steps depending on used budget - on first iters step=1, on last it grows ~x^2
def REMAP_ITER(total=300):
    step_budget = 0
    REMAP_ITER_TO_STEP = {}
    for i in range(total):  # for budget less than 100 mln nodes
        remap = int(max(1, step_budget / 20))
        REMAP_ITER_TO_STEP[step_budget] = remap
        step_budget += remap
    return REMAP_ITER_TO_STEP
