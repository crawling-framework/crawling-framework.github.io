from cyth.build_cython import build_cython  # in order to compile all needed cython files
from utils import rel_dir

build_cython(rel_dir)

from cyth.cbasic import cbasic_test
from cyth.node_deg_set import test_ndset
from cyth.test_cython import test_class

from cyth.cgraph import cgraph_test

if __name__ == '__main__':

    # test_class()
    # test_ndset()

    cbasic_test()
    # cgraph_test()
