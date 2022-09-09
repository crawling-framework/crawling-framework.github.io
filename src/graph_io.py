import logging
import os
import os.path
import re
import shutil
import urllib.error
import urllib.request
from time import time

import patoolib
from base.cgraph import MyGraph

from utils import GRAPHS_DIR

FORMAT = 'ij'
TMP_GRAPHS_DIR = GRAPHS_DIR / 'tmp'
netrepo_metadata_path = GRAPHS_DIR / 'netrepo' / 'metadata'

# Graphs used in current session. Need this to avoid loading the same object several times.
current_graphs = {}  # full_name -> MyGraph


def parse_netrepo_page():
    """ Parse networkrepository page and create name resolution dict: name -> url
    """
    from bs4 import BeautifulSoup

    logging.info("Parsing networkrepository metadata...")
    name_ref_dict = {}
    url = 'http://networkrepository.com/networks.php'
    try:
        html = urllib.request.urlopen(url).read()
    except urllib.error.URLError as e:
        logging.error("Unfortunately, web-cite %s is unavailable. Try again later. Perhaphs the URL could change" % url)
        return

    rows = BeautifulSoup(html, "lxml").table.find_all('tr')
    for row in rows[1:]:
        name = row.contents[0].contents[0].text.strip()
        ref = row.contents[-1].contents[2]['href']
        name_ref_dict[name] = ref

    if not os.path.exists(os.path.dirname(netrepo_metadata_path)):
        os.makedirs(os.path.dirname(netrepo_metadata_path))
    with open(netrepo_metadata_path, 'w') as f:
        f.write(str(name_ref_dict))
    logging.info("networkrepository metadata saved to %s" % netrepo_metadata_path)


# Should be run before downloading any graph
if not os.path.exists(netrepo_metadata_path): parse_netrepo_page()


# netrepo_name_ref_dict = eval(open(netrepo_metadata_path, 'r').read())


def reformat_graph_file(path, out_path, ignore_lines_starting_with='#%',
                        remove_original=False, self_loops=False, renumerate=False):
    """

    :param path:
    :param out_path:
    :param ignore_lines_starting_with: lines starting with these symbols will be ignored
    :param remove_original: if True, original file will be removed
    :param self_loops: if True, self loops will be removed
    :param renumerate: if True, nodes are re-numerated from 0 to N-1
    :return:
    """
    in_format = None
    out_format = FORMAT
    renums = {}
    separators = ' |\t|,'

    assert out_path != path
    with open(out_path, 'w') as out_file:
        for line in open(path, 'r'):
            if line[0] in ignore_lines_starting_with:  # Filter comments
                continue
            line = line.rstrip('\n')
            assert line[0].isdigit(), "expected alpha-numeric line: '%s'" % line
            if not in_format:
                # Define format
                items = re.split(separators, line)
                in_format = 'ijwt'[:len(items)]
                if len(out_format) > len(in_format):
                    raise Exception("Could not reformat from '%s' to '%s'" % (in_format, out_format))
                logging.info("Reformatting %s->%s for '%s' ..." % (in_format, out_format, path))

            items = re.split(separators, line)
            i, j = items[0], items[1]
            if not self_loops and i == j:
                continue

            if renumerate:
                if i not in renums:
                    renums[i] = len(renums)
                if j not in renums:
                    renums[j] = len(renums)
                items[0] = str(renums[i])
                items[1] = str(renums[j])

            res_line = '%s %s\n' % (items[0], items[1])
            out_file.write(res_line)

    if remove_original:
        os.remove(path)
    logging.info("Reformatting finished '%s'." % out_path)


class GraphCollections:
    """
    Manager of graph data.
    By calling method `get(graph_full_name)`, it loads graph from file if any or downloads a graph
    from online graph collection.
    `graph_full_name` is string or tuple ([collection], [subcollection], ... , name) containing at
    least one element, the last one is treated as graph name.
    Corresponding graph file is stored at `collection/subcollection/../name.format` file.

    `networkrepository <http://networkrepository.com/>`_ collection is available.

    Example:
    >>> graph = GraphCollections.get('konect', 'dolphins')

    """
    networkrepository_url_pattern = 'http://nrvis.com/download/data/%s/%s.zip'

    @staticmethod
    def get(*full_name, directed=False, giant_only=True, self_loops=False, not_load=False) -> MyGraph:
        """
        Read graph from storage or download it from the specified collection. In order to apply
        giant_only and self_loops, you need to remove the file manually. #TODO maybe make a rewrite?

        :param full_name: string or sequence [collection], [subcollection], ... , name containing at
         least one element, the last one is treated as graph name. In case of konect collection, graph
         name could be any of e.g. 'CL' or 'Actor collaborations' or 'actor-collaborations'.
        :param directed: undirected by default
        :param giant_only: giant component instead of full graph. Component extraction is applied
         only once when the graph is downloaded.
        :param self_loops: self loops are removed by default. Applied only once when the graph is
         downloaded.
        :param not_load: if True do not load the graph (useful for stats exploring). Note: any graph
         modification will lead to segfault
        :return: MyGraph object
        """
        if isinstance(full_name, str):
            full_name = (full_name,)  # root data directory

        # Check if graph was already loaded
        if full_name in current_graphs:
            return current_graphs[full_name]

        path = GraphCollections._full_name_to_path(*full_name)

        if not os.path.exists(path):
            # Download graph if absent
            temp_path = path + '_tmp'

            if len(full_name) == 2 and full_name[0] == 'netrepo':
                GraphCollections._download_netrepo(temp_path, netrepo_name_ref_dict[full_name[-1]])

            else:
                raise FileNotFoundError("File '%s' not found. Check graph name, collection or file existence." % path)

            reformat_graph_file(temp_path, path, remove_original=True, self_loops=self_loops)

            if giant_only:
                # Replace graph by its giant component
                logging.info("Extracting giant component ...")
                MyGraph(path, full_name, directed, format=FORMAT).giant_component(inplace=True)
                logging.info("done.")

        my_graph = MyGraph(path, full_name, directed, format=FORMAT, not_load=not_load)
        current_graphs[full_name] = my_graph
        return my_graph

    @staticmethod
    def _full_name_to_path(*full_name) -> str:
        """ Convert MyGraph full_name into path it should be stored at"""
        format = FORMAT
        return os.path.join(GRAPHS_DIR, *full_name[:-1], "%s.%s" % (full_name[-1], format))

    @staticmethod
    def get_by_path(path: str, not_load=False, store=True) -> MyGraph:
        """ Create and load graph from specified file path.
        If the path is <GRAPHS_DIR>/a/b/name.ij the full_name will be ('a', 'b', 'name')
        """
        from utils import GRAPHS_DIR
        assert str(GRAPHS_DIR) in path, "Please, put your graph file to %s" % GRAPHS_DIR
        parts = path.split(os.path.sep)[len(GRAPHS_DIR.parts):]
        last_dot = parts[-1].rfind('.')
        last = parts[-1][:last_dot] if last_dot > 0 else parts[-1]
        full_name = tuple(parts[:-1]) + (last,)

        my_graph = MyGraph(path, full_name, not_load=not_load)
        if store:
            current_graphs[full_name] = my_graph
        return my_graph

    @staticmethod
    def register_new_graph(*full_name) -> MyGraph:
        """ Create a new MyGraph object, define its path corresponding to the specified full_name.

        NOTE: by default the graph is not loaded, call load() if want to use this object.

        :param full_name: string or sequence [collection], [subcollection], ... , name containing
         at least one element, the last one is treated as graph name.
        :return: new MyGraph
        """
        if len(full_name) == 0:  # tmp graph, not gonna save
            path = str(TMP_GRAPHS_DIR / f"{str(time())}.{FORMAT}")
            return GraphCollections.get_by_path(path, not_load=True, store=False)

        path = GraphCollections._full_name_to_path(*full_name)
        if os.path.exists(path):
            raise IOError("Path corresponding to specified graph full_name %s is not free: look at ")
        return MyGraph(path=path, full_name=full_name, not_load=True)

    @staticmethod
    def _download_netrepo(graph_path, url):
        """
        Downloads graph data from http://networkrepository.com/networks.php

        :param graph_path: full path to edge list file
        :param url: URL of graph data
        :return:
        """
        # 'http://nrvis.com/download/data/eco/eco-florida.zip'
        # Convert "url.tar.*" -> "filename.tar"
        if (url.rsplit('/', 1)[1]).rsplit('.', 2)[1] == "tar":
            archive_name = (url.rsplit('/', 1)[1]).rsplit('.', 1)[0]
        else:
            archive_name = (url.rsplit('/', 1)[1])

        graph_dir = os.path.dirname(graph_path)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        # Download archive and extract graph file
        logging.info("Downloading graph archive from %s..." % url)
        filename = os.path.join(graph_dir, archive_name)
        urllib.request.urlretrieve(url, filename=filename)
        logging.info("done.")
        archive_dir_name = archive_name.split('.', 1)[0]
        patoolib.extract_archive(filename, outdir=os.path.join(graph_dir, archive_dir_name))

        out_file_name = os.path.join(graph_dir, os.path.basename(graph_path))

        # Rename extracted graph file
        while True:
            # todo are there else formats besides '.edges', '.mtx' ?
            try:
                netrepo_file_name = os.path.join(graph_dir, archive_dir_name, archive_dir_name + ".edges")
                os.rename(netrepo_file_name, out_file_name)
                break
            except IOError: pass
            try:
                netrepo_file_name = os.path.join(graph_dir, archive_dir_name, archive_dir_name + ".mtx")
                # Remove first two lines (solution from https://stackoverflow.com/a/2329972/8900030)
                fro = open(netrepo_file_name, "rb")
                fro.readline()
                fro.readline()
                frw = open(netrepo_file_name, "r+b")
                chars = fro.readline()
                while chars:
                    frw.write(chars)
                    chars = fro.readline()
                fro.close()
                frw.truncate()
                frw.close()
                os.rename(netrepo_file_name, out_file_name)
                break
            except IOError: pass
            break

        os.remove(os.path.join(graph_dir, archive_name))
        shutil.rmtree(os.path.join(graph_dir, archive_dir_name))


def test_netrepo():
    for name in ['soc-wiki-Vote', 'socfb-Bingham82']:
        g = GraphCollections.get('netrepo', name, directed=False, giant_only=True, self_loops=False)
        g = GraphCollections.get('netrepo', name, directed=False, giant_only=True, self_loops=False)
        g = GraphCollections.get('netrepo', name, directed=False, giant_only=True, self_loops=False)
        print("N=%s E=%s" % (g.nodes(), g.edges()))


class temp_dir(object):
    """
    Creates a temporary directory to store some files, which will be removed by exit.
    Current working directory is also changed to this directory.
    """
    def __init__(self):
        self.dir_name = os.path.join(TMP_GRAPHS_DIR, str(time()))
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

    def __enter__(self):
        os.chdir(self.dir_name)
        return self.dir_name

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.dir_name)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # parse_netrepo_page()
    test_netrepo()
