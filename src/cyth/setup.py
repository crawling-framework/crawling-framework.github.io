from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
# from Cython.Build import cythonize

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
]

setup(
    name="crawlers_cython",
    packages=["cyth"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    # ext_modules=cythonize("cyth/test_cython.pyx"),
)
