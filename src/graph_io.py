import logging
import os.path
import shutil
import urllib.request
import re
import patoolib
import snap

from base.cgraph import MyGraph
from utils import GRAPHS_DIR, COLLECTIONS


konect_metadata_path = os.path.join(GRAPHS_DIR, 'konect', 'metadata')
netrepo_metadata_path = os.path.join(GRAPHS_DIR, 'netrepo', 'metadata')


def parse_konect_page():
    """ Parse konect page and create name resolution dict. E.g. 'CL' -> 'actor-collaborations'.
    Note several non-unique codes: DB, HY, OF, PL, WT.
    """
    from bs4 import BeautifulSoup
    import lxml
    logging.info("Parsing Konect metadata...")
    name_ref_dict = {}
    url = 'http://konect.uni-koblenz.de/networks/'
    html = urllib.request.urlopen(url).read()

    rows = BeautifulSoup(html, "lxml").table.find_all('tr')
    for row in rows[1:]:
        cols = row.find_all('td')
        code = cols[0].contents[0].contents[0]
        name = cols[1].contents[0].contents[0]
        ref = cols[1].contents[0]['href']
        if code in name_ref_dict:
            logging.warning("Konect repeating code %s" % code)
            pass  # FIXME codes are not unique, some repeat!
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
    html = urllib.request.urlopen(url).read()

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


# Should be run before
if not os.path.exists(konect_metadata_path): parse_konect_page()
if not os.path.exists(netrepo_metadata_path): parse_netrepo_page()


netrepo_name_ref_dict = eval(open(netrepo_metadata_path, 'r').read())
konect_name_ref_dict = eval(open(konect_metadata_path, 'r').read())


konect_names = [
    'petster-hamster',          # N=2000,    E=16098,    d_avg=16.10
    'ego-gplus',                # N=23613,   E=39182,    d_avg=3.32
    'slashdot-threads',         # N=51083,   E=116573,   d_avg=4.56
    'facebook-wosn-links',      # N=63392,   E=816831,   d_avg=25.77
    'petster-friendships-cat',  # N=148826,  E=5447464,  d_avg=73.21
    'petster-friendships-dog',  # N=426485,  E=8543321,  d_avg=40.06
    'douban',                   # N=154908,  E=327162,   d_avg=4.22
    'digg-friends',             # N=261489,  E=1536577,  d_avg=11.75
    'munmun_twitter_social',    # N=465017,  E=833540,   d_avg=3.58
    'com-youtube',              # N=1134890, E=2987624,  d_avg=5.27
    'flixster',                 # N=2523386, E=7918801,  d_avg=6.28
    'youtube-u-growth',         # N=3216075, E=9369874,  d_avg=5.83
    'soc-pokec-relationships',  # N=1632803, E=22301964, d_avg=27.32
]


netrepo_names = [
    # Graphs used in https://dl.acm.org/doi/pdf/10.1145/3201064.3201066
    # Guidelines for Online Network Crawling: A Study of DataCollection Approaches and Network Properties

    # 'soc-wiki-Vote',  # N=889, E=2914, d_avg=6.56
    'socfb-Bingham82',  # N=10001, E=362892, d_avg=72.57
    'soc-brightkite',  # N=56739, E=212945, d_avg=7.51

    # Collaboration
    'ca-citeseer',  # N=227320, E=814134, d_avg=7.16
    'ca-dblp-2010',  # N=226413, E=716460, d_avg=6.33
    'ca-dblp-2012',  # N=317080, E=1049866, d_avg=6.62
    'ca-MathSciNet',  # N=332689, E=820644, d_avg=4.93

    # Recommendation
    'rec-amazon',  # N=91813, E=125704, d_avg=2.74
    'rec-github',  # N=121331, E=439642, d_avg=7.25

    # FB
    # 'socfb-OR',  # N=63392, E=816886, d_avg=25.77
    'socfb-Penn94',  # N=41536, E=1362220, d_avg=65.59
    'socfb-wosn-friends',  # N=63392, E=816886, d_avg=25.77

    # Tech
    'tech-p2p-gnutella',  # N=62561, E=147878, d_avg=4.73
    'tech-RL-caida',  # N=190914, E=607610, d_avg=6.37

    # Web
    'web-arabic-2005',  # N=163598, E=1747269, d_avg=21.36
    'web-italycnr-2000',  # N=325557, E=2738969, d_avg=16.83
    'web-sk-2005',  # N=121422, E=334419, d_avg=5.51
    'web-uk-2005',  # N=129632, E=11744049, d_avg=181.19

    # OSNs
    'soc-slashdot',  # N=70068, E=358647, d_avg=10.24
    'soc-themarker',  # ? N=69317, E=1644794, d_avg=47.46
    'soc-BlogCatalog',  # N=88784, E=2093195, d_avg=47.15

    # Scientific
    'sc-pkustk13',  # N=94893, E=3260967, d_avg=68.73
    'sc-pwtk',  # N=217883, E=5653217, d_avg=51.89
    'sc-shipsec1',  # N=139995, E=1705212, d_avg=24.36
    'sc-shipsec5',  # N=178573, E=2197367, d_avg=24.61
]


def reformat_graph_file(path, out_path, out_format='ij', ignore_lines_starting_with='#%',
                        remove_original=False, self_loops=False, renumerate=False):
    """

    :param path:
    :param out_path:
    :param out_format: 'ij', 'ijw', 'ijwt'
    :param ignore_lines_starting_with:
    :param remove_original: original file is not removed by default
    :param self_loops: self loops are removed by default.
    :param renumerate: nodes are not re-numerated from 0 to N-1 by default.
    :return:
    """
    in_format = None
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

            # TODO format depending on each symbol of 'ijwt'
            res_line = ' '.join(items[:len(out_format)]) + '\n'
            out_file.write(res_line)

    if remove_original:
        os.remove(path)
    logging.info("Reformatting finished '%s'." % out_path)


class GraphCollections:
    konect_url_pattern = 'http://konect.uni-koblenz.de/downloads/tsv/%s.tar.bz2'
    networkrepository_url_pattern = 'http://nrvis.com/download/data/%s/%s.zip'

    @staticmethod
    def get(name, collection=None, directed=False, format='ij', giant_only=False, self_loops=False, not_load=False):
        """
        Read graph from storage or download it from the specified collection. In order to apply
        giant_only and self_loops, you need to remove the file manually.

        :param name: any of e.g. 'CL' or 'Actor collaborations' or 'actor-collaborations'
        :param collection: 'other', 'konect', 'netrepo'. By default it searches by name in 'konect',
         then 'neterepo', then 'other'
        :param directed: undirected by default
        :param format: output will be in this format, 'ij' by default
        :param giant_only: giant component instead of full graph. Component extraction is applied
         only once when the graph is downloaded.
        :param self_loops: self loops are removed by default. Applied only once when the graph is
         downloaded.
        :param not_load: if True do not load the graph (useful for stats exploring). Note: any graph
         modification will lead to segfault
        :return: MyGraph with snap graph
        """
        # assert collection in COLLECTIONS

        if collection is None:
            # Resolve name: find in konect then neterpo, if no set collection to other
            try:
                name = konect_name_ref_dict[name]
                collection = 'konect'
            except KeyError:
                try:
                    netrepo_url = netrepo_name_ref_dict[name]
                    collection = 'netrepo'
                except KeyError:
                    collection = 'other'

        path = os.path.join(GRAPHS_DIR, collection, "%s.%s" % (name, format))

        # TODO let collection be not specified, try Konect then Netrepo, etc
        if not os.path.exists(path):
            temp_path = os.path.join(GRAPHS_DIR, collection, '%s.tmp' % name)

            if collection == 'konect':
                GraphCollections._download_konect(
                    temp_path, GraphCollections.konect_url_pattern % name)

            elif collection == 'netrepo':
                # name_ref_dict = eval(open(netrepo_metadata_path, 'r').read())
                # url = name_ref_dict[name]
                GraphCollections._download_netrepo(temp_path, netrepo_url)

            elif collection == 'other':
                raise FileNotFoundError("File '%s' not found. Check graph name or file existence." % path)

            reformat_graph_file(temp_path, path, out_format=format, remove_original=True, self_loops=self_loops)

            if giant_only:  # replace graph by its giant component
                logging.info("Extracting giant component ...")
                assert format == 'ij'
                s = snap.LoadEdgeList(snap.PNGraph, path, 0, 1)  #
                s = snap.GetMxWcc(s)
                # snap.SaveEdgeList(s, path, "")
                with open(path, 'w') as f:
                    for e in s.Edges():
                        f.write("%s %s\n" % (e.GetSrcNId(), e.GetDstNId()))
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
        archive_dir_name = archive_name.split('.', 1)[0]
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


def test_konect():
    # name = 'soc-pokec-relationships'
    # name = 'petster-hamster'
    # name = 'github'
    # name = 'twitter'
    # name = 'ego-gplus'
    # name = 'libimseti'
    name = 'Advogato'  # 'AD' 'advogato' # 'Advogato'
    # name = 'facebook-wosn-links'
    # name = 'soc-Epinions1'
    # name = 'douban'
    # name = 'slashdot-threads'
    # name = 'digg-friends'
    # name = 'petster-friendships-cat'  # snap load is long, possibly due to unordered ids
    g = GraphCollections.get(name, directed=False, giant_only=True, self_loops=False)
    # g = GraphCollections.get('eco-florida', collection='networkrepository').snap
    print("N=%s E=%s" % (g.nodes(), g.edges()))
    # print("neigbours of %d: %s" % (2, graph.neighbors(2)))


def test_netrepo():
    # name = 'cit-DBLP'
    # name = 'cit-HepPh'
    # name = 'road-chesapeake'
    # name = 'fb-pages-tvshow'
    # name = 'socfb-Amherst41'
    # name = 'socfb-nips-ego'
    # name = 'ca-CSphd'
    name = 'ia-crime-moreno'
    g = GraphCollections.get(name, 'netrepo', directed=False, giant_only=True, self_loops=False)
    # g = GraphCollections.get('eco-florida', collection='networkrepository').snap
    print("N=%s E=%s" % (g.nodes(), g.edges()))
    # print("neigbours of %d: %s" % (2, graph.neighbors(2)))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # test_konect()
    test_netrepo()
    # test_graph_manipulations()
    # parse_konect_page()
    # parse_netrepo_page()
