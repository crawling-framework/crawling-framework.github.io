import logging
import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
# from Cython.Build import cythonize


def build_cython(rel_dir, snap_dir):
    logging.info("Building cyth...")

    src_dir = os.path.join(rel_dir, 'src')
    os.chdir(src_dir)
    command = "python3 '%s' build_ext --inplace -S'%s'" % (os.path.join(src_dir, os.path.realpath(__file__)), snap_dir)
    exit_code = os.system(command)

    if exit_code != 0:
        raise RuntimeError(" *** Building Cython files failed (exit code %s) ***" % exit_code)
    logging.info(" *** Built Cython files successfully *** \n\n\n")


if __name__ == '__main__':
    import sys
    for arg in sys.argv:
        if arg.startswith('-S'):
            snap_dir = arg[2:]
            sys.argv.remove(arg)

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
        Extension("base.cgraph",
                  ["base/cgraph.pyx"],
                  language='c++',
                  extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
                  extra_link_args=['-fopenmp', '-lrt'],  # '-lrt'
                  extra_objects=[os.path.join(snap_dir, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core")],
                  ),
        Extension("base.cbasic",
                  ["base/cbasic.pyx"],
                  language='c++',
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp'],
                  extra_objects=[os.path.join(snap_dir, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core")],
                  ),
        Extension("base.cmultiseed",
                  ["base/cmultiseed.pyx"],
                  language='c++',
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp'],
                  extra_objects=[os.path.join(snap_dir, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core")],
                  ),
        Extension("base.cadvanced",
                  ["base/cadvanced.pyx"],
                  language='c++',
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp'],
                  extra_objects=[os.path.join(snap_dir, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core")],
                  ),
        Extension("cyth.cstatistics",
                  ["cyth/cstatistics.pyx"],
                  language='c++',
                  extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
                  extra_link_args=['-fopenmp', '-lrt'],
                  extra_objects=[os.path.join(snap_dir, "snap-core/Snap.o")],
                  include_dirs=[os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core")],
                  ),
    ]

    setup(
        name="crawlers_cython",
        packages=["base"],
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        # ext_modules=cythonize("cyth/test_cython.pyx"),
    )
