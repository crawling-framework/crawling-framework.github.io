import logging
import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
# from Cython.Build import cythonize

SNAP_DIR = "/home/misha/Snap-5.0/"  # directory with snap built


def build_cython(rel_dir):
    logging.info("Building cyth...")

    src_dir = os.path.join(rel_dir, 'src')
    os.chdir(src_dir)
    command = "python3 '%s' build_ext --inplace" % os.path.join(src_dir, os.path.realpath(__file__))
    exit_code = os.system(command)

    if exit_code != 0:
        raise RuntimeError(" *** Building Cython files failed (exit code %s) ***" % exit_code)
    logging.info(" *** Built Cython files successfully *** \n\n\n")


if __name__ == '__main__':
    # Compiling Cython modules
    ext_modules = [
        # Extension("base.test_cython",
        #           ["base/test_cython.pyx"],
        #           language='c++',
        #           extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        #           extra_link_args=['-fopenmp']
        #           ),
        Extension("base.node_deg_set",
                  ["base/node_deg_set.pyx"],
                  language='c++',
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp']
                  ),
        Extension("base.cbasic",
                  ["base/cbasic.pyx"],
                  language='c++',
                  # extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp'],  # '-lrt'
                  extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
                  ),
        Extension("base.cmultiseed",
                  ["base/cmultiseed.pyx"],
                  language='c++',
                  # extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp'],  # '-lrt'
                  extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
                  ),
        Extension("base.cgraph",
                  ["base/cgraph.pyx"],
                  language='c++',
                  extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
                  extra_link_args=['-fopenmp', '-lrt'],  # '-lrt'
                  extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
                  ),
        Extension("cyth.cstatistics",
                  ["cyth/cstatistics.pyx"],
                  language='c++',
                  extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
                  extra_link_args=['-fopenmp', '-lrt'],  # '-lrt'
                  extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
                  ),
    ]

    setup(
        name="crawlers_cython",
        packages=["base"],
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        # ext_modules=cythonize("cyth/test_cython.pyx"),
    )
