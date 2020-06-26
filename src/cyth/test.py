from utils import USE_CYTHON_CRAWLERS  # to compile cython

from base.cgraph import cgraph_test
from base.cbasic import cbasic_test
from base.cadvanced import test_cadvanced
from cyth.cstatistics import test_cstats

# from base.cmultiseed import test_multiseed

if __name__ == '__main__':

    # test_class()
    # test_ndset()

    # cgraph_test()
    # cbasic_test()
    # test_multiseed()
    # test_cstats()
    test_cadvanced()
