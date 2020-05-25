import logging
import os

from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, address as addr
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from utils import TMP_GRAPHS_DIR

cimport cgraph  # pxd import DON'T DELETE

logger = logging.getLogger(__name__)

cdef TRnd t_random
t_random.Randomize()

cdef inline fingerprint(const TUNGraph* snap_graph):  # FIXME duplicate
    """ Graph fingerprint to make sure briefly if it has changed.

    :param snap_graph:
    :return: (|V|, |E|)
    """
    return deref(snap_graph).GetNodes(), deref(snap_graph).GetEdges()


cdef class CGraph:
    def __init__(self, path: str=None, name: str='noname', directed: bool=False, weighted: bool=False, str format='ij'):
        """

        :param path: load from path. If None, create empty graph
        :param name: name. 'noname' by default
        :param directed: ignored: undirected only
        :param weighted: ignored: unweighted only
        :param format: ignored: 'ij' only
        """
        assert directed == False
        assert weighted == False
        cdef TUNGraph g
        if path is None:
            from datetime import datetime
            path = os.path.join(TMP_GRAPHS_DIR, "%s_%s" % (name, datetime.now()))
            self._path = str_to_chars(path)
            self._snap_graph_ptr = <PUNGraph> new TUNGraph()
            self._snap_graph = deref(self._snap_graph_ptr)
            # NOTE: If we define a pointer as address of object, segfault occurs
        else:
            self._path = str_to_chars(path)
            self._snap_graph_ptr = LoadEdgeList[PUNGraph](TStr(self._path), 0, 1)
            self._snap_graph = deref(self._snap_graph_ptr)
            # self.load()

        self._name = str_to_chars(name)
        self._directed = directed
        self._weighted = weighted
        # self._format = format  # unused

        self._fingerprint = fingerprint(addr(self._snap_graph))
        self._stats_dict = {}

    def __dealloc__(self):
        # self._snap_graph_ptr.Clr()
        pass

    cdef CGraph load(self):
        self._snap_graph = deref(LoadEdgeList[PUNGraph](TStr(self._path), 0, 1))

    @property
    def path(self):
        return bytes.decode(self._path)

    @property
    def name(self):
        return bytes.decode(self._name)

    @property
    def directed(self):
        return self._directed

    @property
    def weighted(self):
        return self._weighted

    cdef PUNGraph snap_graph_ptr(self):
        return self._snap_graph_ptr

    cpdef int nodes(self):
        """ Number of nodes """
        return self._snap_graph.GetNodes()

    cpdef int edges(self):
        """ Number of edges """
        return self._snap_graph.GetEdges()

    cpdef bint add_node(self, int node):
        return self._snap_graph.AddNode(node)

    cpdef bint add_edge(self, int i, int j):
        return self._snap_graph.AddEdge(i, j)

    cpdef bint has_node(self, int node):
        return self._snap_graph.IsNode(node)

    cpdef bint has_edge(self, int i, int j):
        return self._snap_graph.IsEdge(i, j)

    cpdef int deg(self, int node):
        """ Get node degree. """
        return self._snap_graph.GetNI(node).GetDeg()

    cpdef int max_deg(self):
        """ Get maximal node degree. """
        cdef int res = -1, d
        cdef TUNGraph.TNodeI ni = self._snap_graph.BegNI()
        cdef int count = self._snap_graph.GetNodes()
        for i in range(count):
            d = ni.GetDeg()
            if d > res:
                res = d
            pinc(ni)
        return res

    def neighbors(self, int node):
        """ Generator of neighbors of the given node in this graph.

        :param node: 
        :return: 
        """
        cdef TUNGraph.TNodeI n_iter = self._snap_graph.GetNI(node)
        # cdef TUNGraph.TNode dat = n.NodeHI.GetDat()  TODO could be optimized if have access to private NodeHI
        for i in range(0, n_iter.GetDeg()):
            # yield dat.GetNbrNId(i)
            yield n_iter.GetNbrNId(i)

    def iter_nodes(self):
        cdef TUNGraph.TNodeI ni = self._snap_graph.BegNI()
        cdef int count = self._snap_graph.GetNodes()
        for i in range(count):
            yield ni.GetId()
            pinc(ni)

    def iter_edges(self):
        cdef TUNGraph.TEdgeI ei = self._snap_graph.BegEI()
        cdef int count = self._snap_graph.GetEdges()
        for i in range(count):
            yield ei.GetSrcNId(), ei.GetDstNId()
            pinc(ei)

    cpdef int random_node(self):
        """ Return a random node. """
        return self._snap_graph.GetRndNId(t_random)

    cpdef int random_neighbor(self, int node):
        """ Return a random neighbor of the given node in this graph.
        """
        cdef TUNGraph.TNodeI n_iter = self._snap_graph.GetNI(node)
        cdef int r = t_random.GetUniDevInt(n_iter.GetDeg())
        return n_iter.GetNbrNId(r)

    # @classmethod TODO
    # def new_snap(cls, snap_graph=None, name='tmp', directed=False, weighted=False, format='ij'):
    #     """
    #     Create a new instance of MyGraph with a given snap graph.
    #     :param snap_graph: initial snap graph, or empty if None
    #     :param name: name will be appended with current timestamp
    #     :param directed: will be ignored if snap_graph is specified
    #     :param weighted:
    #     :param format:
    #     :return: MyGraph
    #     """
    #     from datetime import datetime
    #     path = os.path.join(TMP_GRAPHS_DIR, "%s_%s" % (name, datetime.now()))
    #
    #     if snap_graph:
    #         if isinstance(snap_graph, snap.PNGraph):
    #             directed = True
    #         elif isinstance(snap_graph, snap.PUNGraph):
    #             directed = False
    #         else:
    #             raise TypeError("Unknown snap graph type: %s" % type(snap_graph))
    #     else:
    #         snap_graph = snap.TNGraph.New() if directed else snap.TUNGraph.New()
    #
    #     graph = MyGraph(path=path, name=name, directed=directed, weighted=weighted, format=format)
    #     graph._snap_graph = snap_graph
    #     graph._fingerprint = fingerprint(snap_graph)
    #     return graph
    #
    def _check_consistency(self):
        """ Raise exception if graph has changed. """
        f = fingerprint(&self._snap_graph)
        if fingerprint(&self._snap_graph) != self._fingerprint:
            raise Exception("snap graph has changed from the one saved in %s" % self._path)

    def __getitem__(self, stat):
        """ Get graph statistics. Index by str or Stat. Works only if snap graph is immutable. """
        self._check_consistency()
        if isinstance(stat, str):
            from statistics import Stat
            stat = Stat[stat]

        if stat in self._stats_dict:
            value = self._stats_dict[stat]
        else:
            # Try to load from file or compute
            stat_path = os.path.join(
                os.path.dirname(self.path), os.path.basename(self.path) + '_stats', stat.short)
            if not os.path.exists(stat_path):
                # Compute and save stats
                logger.info("Could not find stats '%s' at '%s'. Will be computed." %
                             (stat, stat_path))
                from cyth.cstatistics import stat_computer
                # value = stat.computer(self)
                value = stat_computer[stat](self)

                # Save stats to file
                if not os.path.exists(os.path.dirname(stat_path)):
                    os.makedirs(os.path.dirname(stat_path))
                with open(stat_path, 'w') as f:
                    f.write(str(value))
            else:
                # Read stats from file - value or dict
                value = eval(open(stat_path, 'r').read())

        return value

    def save(self, new_path=None):
        """ Write current edge list of snap graph into file. """
        raise NotImplementedError()
        # s = self._snap_graph TODO
        # assert s
        # if new_path is None:
        #     new_path = self.path
        # if new_path == self.path:
        #     logging.warning("Graph file '%s' will be overwritten." % self.path)
        # # snap.SaveEdgeList writes commented section, we don't want it
        # with open(new_path, 'w') as f:
        #     for e in s.Edges():
        #         f.write("%s %s\n" % (e.GetSrcNId(), e.GetDstNId()))

    @property  # Denis:  could be useful to handle nx version of graph
    def snap_to_networkx(self):
        raise NotImplementedError()
        # nx_graph = nx.Graph() TODO
        # for NI in self.nodes():
        #     nx_graph.add_node(NI.GetId())
        #     for Id in NI.GetOutEdges():
        #         nx_graph.add_edge(NI.GetId(), Id)
        #
        # return nx_graph


def cgraph_test():
    # print("cgraph")

    # cdef TUNGraph g
    # g = TUNGraph()
    # g.AddNode(0)
    # g.AddNode(1)
    # g.AddEdge(0, 1)
    # N = g.GetNodes()
    # E = g.GetEdges()
    # print(N, E)

    # cdef char* a = 'asd'
    # cdef TStr s = TStr(a)
    # # cdef PUNGraph p
    # cdef TPt[TUNGraph] p
    # p = LoadEdgeList[TPt[TUNGraph]](TStr('/home/misha/workspace/crawling/data/konect/dolphins.ij'), 0, 1)
    #
    # cdef TUNGraph t
    # # t = *p
    # t = deref(p)
    #
    # print(t.GetNodes())

    # cdef char* name = 'douban'
    # cdef char* path = '/home/misha/workspace/crawling/data/konect/dolphins.ij'
    empty = CGraph()
    graph = CGraph(path='/home/misha/workspace/crawling/data/konect/dolphins.ij', name='d')
    cdef TUNGraph g = deref(LoadEdgeList[PUNGraph](TStr('/home/misha/workspace/crawling/data/konect/dolphins.ij'), 0, 1))
    # graph = CGraph.CLoad(path)
    # graph = CGraph.Empty('')
    # print(graph.add_node(10))
    # print("path=%s" % 'abc' + graph.path)
    # print("N=%s" % graph.nodes())
    # print("E=%s" % graph.edges())

    print("Nodes:")
    for n in graph.iter_nodes():
        print(n)
    # cdef TUNGraph.TNodeI ni = g.BegNI()
    # cdef int count = g.GetNodes()
    # for i in range(count):
    #     print(ni.GetId())
    #     pinc(ni)

    n = 1
    print("Neighs of %s:" % n)
    for n in graph.neighbors(n):
        print(n)

    print("Rand neighs")
    for _ in range(5):
        # print(graph.random_node())
        print(graph.random_neighbor(1))
    print("End")
