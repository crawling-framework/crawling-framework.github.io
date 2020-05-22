import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
# from Cython.Build import cythonize
from utils import SNAP_DIR

# Compiling Cython modules
ext_modules = [
    Extension("cyth.test_cython",
              ["cyth/test_cython.pyx"],
              language='c++',
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']
              ),
    Extension("cyth.node_deg_set",
              ["cyth/node_deg_set.pyx"],
              language='c++',
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']
              ),
    Extension("cyth.cbasic",
              ["cyth/cbasic.pyx"],
              language='c++',
              # extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'],  # '-lrt'
              extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
              include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
              ),
    Extension("cyth.cgraph",
              ["cyth/cgraph.pyx"],
              language='c++',
              extra_compile_args=["-std=c++98", "-Wall", "-DNDEBUG", "-O3", "-fopenmp", "-ffast-math", "-march=native"],
              extra_link_args=['-fopenmp', '-lrt'],  # '-lrt'
              extra_objects=[os.path.join(SNAP_DIR, "snap-core/Snap.o")],
              include_dirs=[os.path.join(SNAP_DIR, "snap-core/"), os.path.join(SNAP_DIR, "glib-core")],
              ),
]

setup(
    name="crawlers_cython",
    packages=["cyth"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    # ext_modules=cythonize("cyth/test_cython.pyx"),
)
