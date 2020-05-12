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
    # Extension("cython_test.representing.constr",
    #           ["cython_test/representing/constr.pyx"],
    #           language='c++',
    #           extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    #           extra_link_args=['-fopenmp']
    #           ),
    # Extension("cython_test.test_cyth",
    #           ["cython_test/test_cyth.pyx"],
    #           language='c++',
    #           extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    #           extra_link_args=['-fopenmp']
    #           ),
    # Extension("cython_test.cembedding.model",
    #           ["cython_test/cembedding/model.pyx"],
    #           language='c++',
    #           extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    #           extra_link_args=['-fopenmp']
    #           ),
    # Extension("cython_test.cembedding.blm_mod",
    #           ["cython_test/cembedding/blm_mod.pyx"],
    #           language='c++',
    #           ),
]

setup(
    name="crawlers_cython",
    packages=["cyth"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    # ext_modules=cythonize("cyth/test_cython.pyx"),
)
