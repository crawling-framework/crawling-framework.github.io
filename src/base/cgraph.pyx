import logging
import os

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, address as addr
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from utils import TMP_GRAPHS_DIR

cimport cgraph  # pxd import DON'T DELETE

logger = logging.getLogger(__name__)

cdef inline fingerprint(const PUNGraph snap_graph_ptr):  # FIXME duplicate
    """ Graph fingerprint to make sure briefly if it has changed.

    :param snap_graph:
    :return: (|V|, |E|)
    """
    # if snap_graph_ptr is NULL:
    #     return 0, 0
    return deref(snap_graph_ptr).GetNodes(), deref(snap_graph_ptr).GetEdges()


from time import time
cdef TRnd t_random = TRnd(int(time()*1e7 % 1e9), 0)

cpdef void seed_random(int seed):
    t_random.PutSeed(seed)

cdef class CGraph:
    def __init__(self, path: str=None, name: str='noname', directed: bool=False, weighted: bool=False, str format='ij', not_load: bool=False):
        """

        :param path: load from path. If None, create empty graph
        :param name: name. 'noname' by default
        :param directed: ignored: undirected only
        :param weighted: ignored: unweighted only
        :param format: ignored: 'ij' only
        :param not_load: if True do not load the graph (useful for stats exploring). Note: any graph
         modification will lead to segfault
        """
        assert directed == False
        assert weighted == False
        self._name = str_to_chars(name)
        self._directed = directed
        self._weighted = weighted
        # self._format = format  # unused

        cdef TUNGraph g
        if path is None:
            from datetime import datetime
            path = os.path.join(TMP_GRAPHS_DIR, "%s_%s" % (name, datetime.now()))
            self._path = str_to_chars(path)
            self._snap_graph_ptr = PUNGraph.New()
            self._fingerprint = fingerprint(self._snap_graph_ptr)
            # NOTE: If we define a pointer as address of object, segfault occurs ?
        else:
            self._path = str_to_chars(path)
            if not not_load:
                self.load()

        self._stats_dict = {}

    def __dealloc__(self):
        # print("dealloc", self.name)
        pass

    cpdef void load(self):
        logger.debug("Loading graph '%s' from '%s'..." % (self.name, self.path))
        self._snap_graph_ptr = LoadEdgeList[PUNGraph](TStr(self._path), 0, 1)
        self._fingerprint = fingerprint(self._snap_graph_ptr)
        logger.debug("done.")

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
        # self._snap_graph_ptr = TPt[TUNGraph](&self._snap_graph)
        return self._snap_graph_ptr
        # return TPt[TUNGraph](&self._snap_graph)

    cpdef int nodes(self):
        """ Number of nodes """
        return deref(self._snap_graph_ptr).GetNodes()

    cpdef int edges(self):
        """ Number of edges """
        return deref(self._snap_graph_ptr).GetEdges()

    cpdef bint add_node(self, int node):
        return deref(self._snap_graph_ptr).AddNode(node)

    cpdef bint add_edge(self, int i, int j):
        return deref(self._snap_graph_ptr).AddEdge(i, j)

    cpdef bint has_node(self, int node):
        return deref(self._snap_graph_ptr).IsNode(node)

    cpdef bint has_edge(self, int i, int j):
        return deref(self._snap_graph_ptr).IsEdge(i, j)

    cpdef int deg(self, int node):
        """ Get node degree. """
        return deref(self._snap_graph_ptr).GetNI(node).GetDeg()

    cpdef double clustering(self, int node):
        """ Get clustering of a node. """
        return GetANodeClustCf[PUNGraph](self._snap_graph_ptr, node)

    cpdef int max_deg(self):
        """ Get maximal node degree. """
        cdef int res = -1, d, i
        cdef TUNGraph.TNodeI ni = deref(self._snap_graph_ptr).BegNI()
        cdef int count = deref(self._snap_graph_ptr).GetNodes()
        for i in range(count):
            d = ni.GetDeg()
            if d > res:
                res = d
            pinc(ni)
        return res

    def neighbors(self, int node):
        """ Generator of neighbors of the given node in this graph.
        """
        cdef TUNGraph.TNodeI n_iter = deref(self._snap_graph_ptr).GetNI(node)
        # cdef TUNGraph.TNode dat = n.NodeHI.GetDat()  TODO could be optimized if have access to private NodeHI
        for i in range(0, n_iter.GetDeg()):
            # yield dat.GetNbrNId(i)
            yield n_iter.GetNbrNId(i)

    def iter_nodes(self):
        cdef TUNGraph.TNodeI ni = deref(self._snap_graph_ptr).BegNI()
        cdef int count = deref(self._snap_graph_ptr).GetNodes()
        for i in range(count):
            yield ni.GetId()
            pinc(ni)

    def iter_edges(self):
        cdef TUNGraph.TEdgeI ei = deref(self._snap_graph_ptr).BegEI()
        cdef int count = deref(self._snap_graph_ptr).GetEdges()
        for i in range(count):
            yield ei.GetSrcNId(), ei.GetDstNId()
            pinc(ei)

    cpdef int random_node(self):
        """ Return a random node. O(1) """
        return deref(self._snap_graph_ptr).GetRndNId(t_random)

    cpdef vector[int] random_nodes(self, int count=1):
        """ Return a vector of random nodes without repetition. O(N) """
        cdef int size = deref(self._snap_graph_ptr).GetNodes(), i, n
        assert count <= size
        cdef TInt* it
        cdef vector[int] res
        cdef TIntV NIdV
        deref(self._snap_graph_ptr).GetNIdV(NIdV)
        NIdV.Shuffle(t_random)
        it = NIdV.BegI()
        for i in range(count):
            res.push_back(deref(it)())
            inc(it)
        return res

    cpdef int random_neighbor(self, int node):
        """ Return a random neighbor of the given node in this graph.
        """
        cdef TUNGraph.TNodeI n_iter = deref(self._snap_graph_ptr).GetNI(node)
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
        if not self._snap_graph_ptr.Empty():
            f = fingerprint(self._snap_graph_ptr)
            if fingerprint(self._snap_graph_ptr) != self._fingerprint:
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
                logger.info("Could not find stats '%s' at '%s'. Will be computed." % (stat, stat_path))
                if self._snap_graph_ptr.Empty():
                    self.load()
                from cyth.cstatistics import stat_computer
                # value = stat.computer(self)
                value = stat_computer[stat](self)
                self._stats_dict[stat] = value

                # Save stats to file
                if not os.path.exists(os.path.dirname(stat_path)):
                    os.makedirs(os.path.dirname(stat_path))
                with open(stat_path, 'w') as f:
                    f.write(str(value))
            else:
                # Read stats from file - value or dict
                value = eval(open(stat_path, 'r').read())

        return value

    def __setitem__(self, stat, value):
        self._check_consistency()
        if isinstance(stat, str):
            from statistics import Stat
            stat = Stat[stat]
        self._stats_dict[stat] = value

        # Save stats to file
        stat_path = os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + '_stats', stat.short)
        if not os.path.exists(stat_path):
            if not os.path.exists(os.path.dirname(stat_path)):
                os.makedirs(os.path.dirname(stat_path))
        with open(stat_path, 'w') as f:
            f.write(str(value))

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

    def networkit(self, node_map: dict = None):
        """ Get networkit graph, create node ids mapping (neworkit_node_id -> snap_node_id) if
        node_map is specified. Some neworkit_node_id -> None (those ids not present in snap graph).
        """
        import networkit as nk
        tab_or_space = '\t' in open(self.path).readline()
        reader = nk.Format.EdgeListTabZero if tab_or_space else nk.Format.EdgeListSpaceZero
        networkit_graph = nk.readGraph(self.path, reader, directed=self.directed)

        # Create node mapping if needed
        if node_map is not None:
            node_map.clear()
            nodes = sorted(self.iter_nodes())
            n_max = nodes[-1]
            assert networkit_graph.numberOfNodes() == n_max+1
            n = 0
            for s in range(n_max+1):
                if nodes[n] > s:
                    node_map[s] = None
                else:
                    node_map[s] = nodes[n]
                    n += 1

        return networkit_graph

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
    import numpy as np

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

    graph = CGraph(path='/home/misha/workspace/crawling/data/konect/petster-hamster.ij', name='d')
    cdef TUNGraph g = deref(LoadEdgeList[PUNGraph](TStr('/home/misha/workspace/crawling/data/konect/dolphins.ij'), 0, 1))
    empty = CGraph(name='empty')

    # graph = CGraph.CLoad(path)
    # graph = CGraph.Empty('')
    # print(empty.add_node(10))
    # print("path=%s" % 'abc' + graph.path)
    # print("N=%s" % graph.nodes())
    # print("E=%s" % graph.edges())

    # print("Nodes:")
    # for n in graph.iter_nodes():
    #     print(n)
    # cdef TUNGraph.TNodeI ni = g.BegNI()
    # cdef int count = g.GetNodes()
    # for i in range(count):
    #     print(ni.GetId())
    #     pinc(ni)

    # n = 1
    # print("Neighs of %s:" % n)
    # for n in graph.neighbors(n):
    #     print(n)
    #
    # print("Rand neighs")

    # cdef TRnd t_random = TRnd(int(time()*1e6 % 1e9), 0)
    # t_random.PutSeed(3)
    for _ in range(5):
        print(t_random.GetUniDevInt(10))
        # print(graph.random_node())
    #     print(graph.random_neighbor(1))

    # print("Rand nodes")
    # cdef vector[int] rnodes
    # cdef int n
    #
    # t = time()
    # for _ in range(100):
    #     n = graph.random_node()
    #     # rnodes = graph.random_nodes(1000)
    #     # np.random.choice(range(250000), 1000, replace=False)
    # # for n in rnodes:
    # #     print(n)
    # print(time()-t)

    # cdef PUNGraph ptr = PUNGraph.New()
    # cdef TUNGraph g = deref(ptr)
    # cdef TUNGraph g2 = g
    # cdef TUNGraph g = TUNGraph()
    # cdef PUNGraph ptr2 = TPt[TUNGraph](&g)

    # print(ptr == TPt[TUNGraph](&g))
    # print(g.GetNodes())
    # print(deref(ptr).GetNodes())
    # print(deref(ptr2).GetNodes())

    # g.AddNode(1)
    # print(g.GetNodes())
    # print(g2.GetNodes())
    # print(deref(ptr).GetNodes())
    # print(deref(ptr2).GetNodes())

    # deref(ptr).AddNode(1)
    # print(g.GetNodes())
    # print(deref(ptr).GetNodes())
    # print(deref(ptr2).GetNodes())

    # cdef int i, n = 10000000
    # t = time()
    # for i in range(n):
    #     deref(ptr).AddNode(i)
    # print(time()-t)
    #
    # t = time()
    # for i in range(n):
    #     g.AddNode(i)
    # print(time()-t)

    empty.add_node(1)
    cdef double cc = GetANodeClustCf[PUNGraph](empty.snap_graph_ptr(), 1)
    # print("cc", cc)
    print("End")
