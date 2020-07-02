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
    snap_dir = None
    for arg in sys.argv:
        if arg.startswith('-S'):
            snap_dir = arg[2:]
            sys.argv.remove(arg)
    if snap_dir is None:
        raise ValueError("snap directory path is not specified. Use '-S' flag with this script")

    # Defining GCC flags depending on operating system
    import platform
    plt = platform.system()
    if plt == 'Linux':
        extra_compile_flags = ["-O3", "-ffast-math", "-march=native"]
        snap_extra_compile_flags = ["-std=c++98", "-Wall", "-O3", "-DNDEBUG", "-fopenmp"]
        snap_extra_link_args = ["-fopenmp", "-lrt"]

    elif plt == 'Darwin':
        snap_extra_compile_flags = ["-std=c++98", "-Wall", "-Wno-unknown-pragmas", "-DNDEBUG", "-O3"]
        extra_compile_flags = ["-std=c++11", "-O3", "-ffast-math", "-march=native"]
        import subprocess
        result = subprocess.Popen('g++ -v 2>&1 | grep clang | cut -d " " -f 2', shell=True,
                                  stdout=subprocess.PIPE).stdout.read().decode('utf-8')
        snap_extra_compile_flags.append("-fopenmp" if result == 'LLVM' else "-DNOMP")
        snap_extra_link_args = []

    # elif plt == 'Cygwin':  # TODO test on Windows
    #     snap_extra_compile_flags = ["-Wall", "-D__STDC_LIMIT_MACROS", "-DNDEBUG", "-O3"]
    #     snap_extra_link_args = []
    else:
        raise OSError("Sorry, your OS (%s) is not supported" % plt)

    extra_objects = [os.path.join(snap_dir, "snap-core/Snap.o")]
    include_dirs = [os.path.join(snap_dir, "snap-core/"), os.path.join(snap_dir, "glib-core"), 'base']  # append here all project directories with C/C++ code

    # Compiling Cython modules
    ext_modules = [
        Extension("base.node_deg_set",
                  ["base/node_deg_set.pyx"],
                  language='c++',
                  extra_compile_args=extra_compile_flags,
                  ),
        Extension("base.cgraph",
                  ["base/cgraph.pyx"],
                  language='c++',
                  extra_compile_args=snap_extra_compile_flags,
                  extra_link_args=snap_extra_link_args,
                  extra_objects=extra_objects,
                  include_dirs=include_dirs,
                  ),
        Extension("crawlers.cbasic",
                  ["crawlers/cbasic.pyx"],
                  language='c++',
                  extra_compile_args=extra_compile_flags,
                  extra_link_args=snap_extra_link_args,
                  extra_objects=extra_objects,
                  include_dirs=include_dirs,
                  ),
        Extension("crawlers.cadvanced",
                  ["crawlers/cadvanced.pyx"],
                  language='c++',
                  extra_compile_args=extra_compile_flags,
                  extra_link_args=snap_extra_link_args,
                  extra_objects=extra_objects,
                  include_dirs=include_dirs,
                  ),
        Extension("cyth.cstatistics",
                  ["cyth/cstatistics.pyx"],
                  language='c++',
                  extra_compile_args=extra_compile_flags + ["-lrt"],
                  extra_link_args=snap_extra_link_args,
                  extra_objects=extra_objects,
                  include_dirs=include_dirs,
                  ),
        Extension("models.cmodels",
                  ["models/cmodels.pyx"],
                  language='c++',
                  extra_compile_args=extra_compile_flags,
                  extra_link_args=snap_extra_link_args,
                  extra_objects=extra_objects,
                  include_dirs=include_dirs,
                  ),
    ]

    setup(
        name="crawlers_cython",
        packages=["base"],
        cmdclass={"build_ext": build_ext},
        ext_modules=ext_modules,
        # ext_modules=cythonize("cyth/test_cython.pyx"),
    )
