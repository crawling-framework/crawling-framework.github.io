import os

rel_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep
GRAPHS_DIR = os.path.join(rel_dir, "data")  # root directory to store all graph data
PICS_DIR = os.path.join(rel_dir, 'pics')  # directory to store pictures
