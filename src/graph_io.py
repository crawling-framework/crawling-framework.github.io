from utils import rel_dir, USE_CYTHON_CRAWLERS
from cyth.build_cython import build_cython; build_cython(rel_dir)  # Should go before any cython imports

import logging
import os.path
import shutil
import urllib.request

import patoolib
import snap

from utils import GRAPHS_DIR, COLLECTIONS


if USE_CYTHON_CRAWLERS:
    from base.cgraph import CGraph as MyGraph
else:
    from base.graph import MyGraph


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

    assert out_path != path
    with open(out_path, 'w') as out_file:
        for line in open(path, 'r'):
            if line[0] in ignore_lines_starting_with:  # Filter comments
                continue
            line = line.rstrip('\n')
            assert line[0].isdigit(), "expected alpha-numeric line: '%s'" % line
            if not in_format:
                # Define format
                items = line.split()
                in_format = 'ijwt'[:len(items)]
                if len(out_format) > len(in_format):
                    raise Exception("Could not reformat from '%s' to '%s'" % (in_format, out_format))
                logging.info("Reformatting %s->%s for '%s' ..." % (in_format, out_format, path))

            items = line.split()
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
    def get(name, collection='konect', directed=False, format='ij', giant_only=False, self_loops=False):
        """
        Read graph from storage or download it from the specified collection. In order to apply
        giant_only and self_loops, you need to remove the file manually.

        :param name:
        :param collection: 'other', 'konect', 'networkrepository'.
        :param directed: undirected by default
        :param format: output will be in this format, 'ij' by default
        :param giant_only: giant component instead of full graph. Component extraction is applied
         only once when the graph is downloaded.
        :param self_loops: self loops are removed by default. Applied only once when the graph is
         downloaded.
        :return: MyGraph with snap graph
        """
        assert collection in COLLECTIONS
        # category = ''
        path = os.path.join(GRAPHS_DIR, collection, "%s.%s" % (name, format))

        # TODO let collection be not specified, try Konect then Netrepo, etc
        if not os.path.exists(path):
            temp_path = os.path.join(GRAPHS_DIR, collection, '%s.tmp' % name)

            if collection == 'konect':
                GraphCollections._download_konect(
                    temp_path, GraphCollections.konect_url_pattern % name)

            elif collection == 'networkrepository':
                raise NotImplementedError()
                category = name.split('-')[0]
                GraphCollections._download_networkrepository(
                    temp_path, GraphCollections.networkrepository_url_pattern % (category, name))

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

        return MyGraph(path, name, directed, format=format)

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
    def _download_networkrepository(graph_path, url):
        """
        Downloads graph data from http://networkrepository.com/networks.php

        :param graph_path: full path to edge list file
        :param url: URL of graph data
        :return:
        """
        # 'http://nrvis.com/download/data/eco/eco-florida.zip'
        # Convert "url_konect.tar.*" -> "filename.tar"
        if (url.rsplit('/', 1)[1]).rsplit('.', 2)[1] == "tar":
            archive_name = (url.rsplit('/', 1)[1]).rsplit('.', 1)[0]
        else:
            archive_name = (url.rsplit('/', 1)[1])

        graph_dir = os.path.dirname(graph_path)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        # Download archive and extract graph file
        logging.info("Downloading graph archive from networkrepository...")
        filename = os.path.join(graph_dir, archive_name)
        urllib.request.urlretrieve(url, filename=filename)
        logging.info("done.")
        archive_dir_name = archive_name.split('.', 1)[0]
        patoolib.extract_archive(filename, outdir=os.path.join(graph_dir, archive_dir_name))

        # Rename extracted graph file
        # FIXME files are in various formats '.edges', '.mtx' etc
        netwrepo_file_name = os.path.join(graph_dir, archive_dir_name, archive_dir_name + ".edges")
        out_file_name = os.path.join(graph_dir, os.path.basename(graph_path))
        os.rename(netwrepo_file_name, out_file_name)

        os.remove(os.path.join(graph_dir, archive_name))
        shutil.rmtree(os.path.join(graph_dir, archive_dir_name))


def test_io():
    # name = 'soc-pokec-relationships'
    # name = 'petster-hamster'
    # name = 'github'
    # name = 'twitter'
    name = 'ego-gplus'
    # name = 'libimseti'
    # name = 'advogato'
    # name = 'facebook-wosn-links'
    # name = 'soc-Epinions1'
    # name = 'douban'
    # name = 'slashdot-threads'
    # name = 'digg-friends'
    # name = 'petster-friendships-cat'  # snap load is long, possibly due to unordered ids
    graph = GraphCollections.get(name, directed=False, giant_only=True, self_loops=False)
    g = graph.snap
    # g = GraphCollections.get('eco-florida', collection='networkrepository').snap
    print("N=%s E=%s" % (g.GetNodes(), g.GetEdges()))
    # print("neigbours of %d: %s" % (2, graph.neighbors(2)))


def test_graph_manipulations():
    path = '/home/misha/workspace/crawling/data/mipt.ij'

    # # # nodes renumerating
    # reformat_graph_file(path, path+'_', renumerate=True)

    # giant extraction
    print("Giant component extraction")
    graph = MyGraph(path + '_', 'mipt', False)
    s = graph.snap
    s = snap.GetMxWcc(s)
    graph._snap_graph = s
    graph.save(path)


def test_graph():
    g = snap.TUNGraph.New()
    g.AddNode(1)
    g.AddNode(2)
    g.AddNode(3)
    g.AddNode(4)
    g.AddNode(5)
    g.AddEdge(1, 1)
    g.AddEdge(1, 1)
    g.AddEdge(1, 2)
    g.AddEdge(1, 2)
    g.AddEdge(2, 1)
    g.AddEdge(2, 3)
    g.AddEdge(4, 2)
    g.AddEdge(4, 3)
    g.AddEdge(5, 4)
    print("N=%s E=%s" % (g.GetNodes(), g.GetEdges()))
    for e in g.Edges():
        print(e)  # Exception

    graph = MyGraph.new_snap(g)
    g.AddEdge(4, 1)
    print(graph['EDGES'])  # Exception


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    # test_io()
    test_graph()
    # test_graph_manipulations()