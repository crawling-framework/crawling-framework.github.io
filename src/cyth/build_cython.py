import logging
import os


def build_cython(rel_dir, setup_file="cyth/setup.py"):
    logging.info("Building cyth...")

    src_dir = os.path.join(rel_dir, 'src')
    os.chdir(src_dir)
    command = "python3 '%s' build_ext --inplace" % os.path.join(src_dir, setup_file)
    exit_code = os.system(command)

    if exit_code != 0:
        raise RuntimeError(" *** Building Cython files failed (exit code %s) ***" % exit_code)
    logging.info(" *** Built Cython files successfully *** \n\n\n")


# build_cython()
