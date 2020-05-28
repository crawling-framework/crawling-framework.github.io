from utils import rel_dir, USE_CYTHON_CRAWLERS
from cyth.setup import build_cython

build_cython(rel_dir)  # Should go before any cython imports

from base.cgraph import cgraph_test
from base.cbasic import cbasic_test
from cyth.cstatistics import test_cstats
from utils import rel_dir

build_cython(rel_dir)

from base.cmultiseed import test_multiseed

if __name__ == '__main__':

    # test_class()
    # test_ndset()

    cgraph_test()
    # cbasic_test()
    # test_multiseed()
    # test_cstats()
