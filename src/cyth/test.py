from cyth.build_cython import build_cython  # in order to compile all needed cython files
from utils import rel_dir

build_cython(rel_dir)

from base.cmultiseed import test_multiseed

if __name__ == '__main__':

    # test_class()
    # test_ndset()

    # cgraph_test()
    # cbasic_test()
    test_multiseed()
