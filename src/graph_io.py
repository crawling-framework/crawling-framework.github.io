import logging
import os.path
import re
import shutil
import urllib.error
import urllib.request
from time import time

import patoolib
from base.cgraph import MyGraph

from utils import GRAPHS_DIR, TMP_GRAPHS_DIR

konect_metadata_path = os.path.join(GRAPHS_DIR, 'konect', 'metadata')
netrepo_metadata_path = os.path.join(GRAPHS_DIR, 'netrepo', 'metadata')


def parse_konect_page():
    """
    Parse konect page and create name resolution dict. E.g. 'CL' -> 'actor-collaborations'.
    Note it has many non-unique codes.
    """
    from bs4 import BeautifulSoup
    logging.info("Parsing Konect metadata...")
    name_ref_dict = {}
    # url = 'http://konect.uni-koblenz.de/networks/'  # The old one, but could be useful
    url = 'http://konect.cc/networks/'
    try:
        html = urllib.request.urlopen(url).read()
    except urllib.error.URLError as e:
        logging.error("Unfortunately, web-cite %s is unavailable. Try again later. Perhaphs the URL could change" % url)
        return

    rows = BeautifulSoup(html, "lxml").table.find_all('tr')
    for row in rows[1:]:
        cols = row.find_all('td')
        code = cols[0].contents[0].contents[0]
        name = cols[1].contents[0].contents[0]
        ref = cols[1].contents[0]['href']
        ref = ref.replace('/', '')
        if code in name_ref_dict:
            logging.warning("Konect repeating code %s" % code)
            pass
        name_ref_dict[code] = ref
        name_ref_dict[name] = ref
        name_ref_dict[ref] = ref

    if not os.path.exists(os.path.dirname(konect_metadata_path)):
        os.makedirs(os.path.dirname(konect_metadata_path))
    with open(konect_metadata_path, 'w') as f:
        f.write(str(name_ref_dict))
    logging.info("Konect metadata saved to %s" % konect_metadata_path)


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
if not os.path.exists(konect_metadata_path): parse_konect_page()
if not os.path.exists(netrepo_metadata_path): parse_netrepo_page()


konect_name_ref_dict = eval(open(konect_metadata_path, 'r').read())
netrepo_name_ref_dict = eval(open(netrepo_metadata_path, 'r').read())


def reformat_graph_file(path, out_path, ignore_lines_starting_with='#%',
                        remove_original=False, self_loops=False, renumerate=False):
    """

    :param path:
    :param out_path:
    :param out_format: 'ij'
    :param ignore_lines_starting_with: lines starting with these symbols will be ignored
    :param remove_original: if True, original file will be removed
    :param self_loops: if True, self loops will be removed
    :param renumerate: if True, nodes are re-numerated from 0 to N-1
    :return:
    """
    in_format = None
    out_format = 'ij'
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
    By calling method `get(name, collection)`, it loads graph from file if any or downloads a graph from online
    collection.
    `Konect <http://konect.uni-koblenz.de>`_ and `networkrepository <http://networkrepository.com/>`_ collections are
    available.

    Example:
    >>> graph = GraphCollections.get(name='dolphins', collection='konect')

    """
    # konect_url_pattern = 'http://konect.uni-koblenz.de/downloads/tsv/%s.tar.bz2'  # The old one
    konect_url_pattern = 'http://konect.cc/files/download.tsv.%s.tar.bz2'
    networkrepository_url_pattern = 'http://nrvis.com/download/data/%s/%s.zip'

    @staticmethod
    def get(name, collection=None, directed=False, giant_only=True, self_loops=False, not_load=False):
        """
        Read graph from storage or download it from the specified collection. In order to apply
        giant_only and self_loops, you need to remove the file manually. #TODO maybe make a rewrite?

        :param name: any of e.g. 'CL' or 'Actor collaborations' or 'actor-collaborations'
        :param collection: 'konect', 'netrepo', 'other', or any other subfolder in data/. If not
         specified, it searches by name in 'konect', then 'neterepo', then 'other'.
        :param directed: undirected by default
        :param giant_only: giant component instead of full graph. Component extraction is applied
         only once when the graph is downloaded.
        :param self_loops: self loops are removed by default. Applied only once when the graph is
         downloaded.
        :param not_load: if True do not load the graph (useful for stats exploring). Note: any graph
         modification will lead to segfault
        :return: MyGraph object
        """
        format = 'ij'
        if collection is None:
            # Resolve name: search in konect then neterpo, if no set collection to other
            if name in konect_name_ref_dict:
                collection = 'konect'
                name = konect_name_ref_dict[name]
            else:
                if name in netrepo_name_ref_dict:
                    collection = 'netrepo'
                else:
                    collection = 'other'

        path = os.path.join(GRAPHS_DIR, collection, "%s.%s" % (name, format))

        # Download graph if absent
        if not os.path.exists(path):
            temp_path = os.path.join(GRAPHS_DIR, collection, '%s.tmp' % name)

            if collection == 'konect':
                GraphCollections._download_konect(
                    temp_path, GraphCollections.konect_url_pattern % name)

            elif collection == 'netrepo':
                GraphCollections._download_netrepo(temp_path, netrepo_name_ref_dict[name])

            else:
                raise FileNotFoundError("File '%s' not found. Check graph name, collection or file existence." % path)

            reformat_graph_file(temp_path, path, remove_original=True, self_loops=self_loops)

            if giant_only:
                # Replace graph by its giant component
                logging.info("Extracting giant component ...")
                assert format == 'ij'
                MyGraph(path, name, directed, format=format).giant_component()
                logging.info("done.")

        return MyGraph(path, name, directed, format=format, not_load=not_load)

    @staticmethod
    def _download_konect(graph_path, url_konect):
        """
        Downloads graph data from Konect http://konect.uni-koblenz.de

        :param graph_path: full path to edge list file
        :param url_konect: URL of graph data
        :return:
        """
        # Convert "url_konect.tar.*" -> "filename.tar"
        if (url_konect.rsplit('/', 1)[1]).rsplit('.', 2)[1] == "tar":
            archive_name = (url_konect.rsplit('/', 1)[1]).rsplit('.', 1)[0]
        else:
            archive_name = (url_konect.rsplit('/', 1)[1])

        graph_dir = os.path.dirname(graph_path)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        # Download archive and extract graph file
        logging.info("Downloading graph archive from Konect...")
        filename = os.path.join(graph_dir, archive_name)
        urllib.request.urlretrieve(url_konect, filename=filename)
        logging.info("done.")
        patoolib.extract_archive(filename, outdir=graph_dir)

        # Rename extracted graph file
        # archive_dir_name = archive_name.split('.', 1)[0]  # For the old cite
        archive_dir_name = archive_name.split('.')[2]
        out_file_name = os.path.join(graph_dir, os.path.basename(graph_path))

        # multigraphs' filenames end with '-uniq'
        # signed networks' filenames are '.matrix todo what else?
        for ending in [archive_dir_name, archive_dir_name + '-uniq', 'matrix']:
            try:
                konect_file_name = os.path.join(graph_dir, archive_dir_name, "out." + ending)
                os.rename(konect_file_name, out_file_name)
                break
            except IOError:
                pass

        os.remove(os.path.join(graph_dir, archive_name))
        shutil.rmtree(os.path.join(graph_dir, archive_dir_name))

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


def test_konect():
    for name in ['petster-hamster', 'digg-friends']:
        g = GraphCollections.get(name, directed=False, giant_only=True, self_loops=False)
        print("N=%s E=%s" % (g.nodes(), g.edges()))
        # print("neigbours of %d: %s" % (2, graph.neighbors(2)))


def test_netrepo():
    for name in ['soc-wiki-Vote', 'socfb-Bingham82']:
        g = GraphCollections.get(name, 'netrepo', directed=False, giant_only=True, self_loops=False)
        print("N=%s E=%s" % (g.nodes(), g.edges()))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # parse_konect_page()
    # parse_netrepo_page()
    test_konect()
    test_netrepo()
