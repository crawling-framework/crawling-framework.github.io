import logging
import os
import shutil
import json

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc, address as addr

from utils import TMP_GRAPHS_DIR

cimport cgraph  # pxd import DON'T DELETE

logger = logging.getLogger(__name__)

cdef inline fingerprint(const PUNGraph snap_graph_ptr):
    """ Graph fingerprint to make sure briefly if it has changed.

    :param snap_graph:
    :return: (|V|, |E|)
    """
    return deref(snap_graph_ptr).GetNodes(), deref(snap_graph_ptr).GetEdges()


from time import time
cdef TRnd t_random = TRnd(int(time()*1e7 % 1e9), 0)

cpdef void seed_random(int seed):
    t_random.PutSeed(seed)

cpdef int get_UniDevInt(int max):
    return t_random.GetUniDevInt(max)

cdef class MyGraph:
    """

    TODO add docs
    """
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
            path = os.path.join(TMP_GRAPHS_DIR, "%s_%s.ij" % (name, datetime.now()))
            self._path = str_to_chars(path)
            self._snap_graph_ptr = PUNGraph.New()
            self._fingerprint = fingerprint(self._snap_graph_ptr)
            # NOTE: If we define a pointer as address of object, segfault occurs ?
        else:
            self._path = str_to_chars(path)
            if not not_load:
                self.load()

        self._stats_dict = {}

    # def __dealloc__(self):
    #     # print("dealloc", self.name)
    #     pass

    cpdef bint is_loaded(self):
        return not self._snap_graph_ptr.Empty()

    cpdef void load(self):
        logging.info("Loading graph '%s' from '%s'..." % (self.name, self.path))
        self._snap_graph_ptr = LoadEdgeList[PUNGraph](TStr(self._path), 0, 1)
        self._fingerprint = fingerprint(self._snap_graph_ptr)
        logging.info("done.")

    cpdef void save(self):
        """ Write current edge list of snap graph into file. """
        # snap.SaveEdgeList writes commented section, we don't want it
        cdef int i, j
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        try:
            self._check_consistency()
        except Exception:
            # Fingerprint changed - we remove graph file and all stats
            # os.remove(self.path)
            self._stats_dict.clear()
            if os.path.exists(self._stat_dir()):
                shutil.rmtree(self._stat_dir())

        with open(self.path, 'w') as f:
            for i, j in self.iter_edges():
                f.write("%s %s\n" % (i, j))

    cpdef MyGraph copy(self):
        """ Create and return a copy of this graph """
        # TODO create snap graph(N, E) could be faster
        cdef MyGraph g = MyGraph(name=self.name + '_copy', directed=self._directed, weighted=self._weighted)
        cdef int n, i, j
        for n in self.iter_nodes():
            g.add_node(n)
        for i, j in self.iter_edges():
            g.add_edge(i, j)
        return g

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

    cdef new_snap(self, PUNGraph snap_graph_ptr, name=None):
        """
        Replace self snap graph with a given snap graph. NOTE: all computed and saved stats will be removed.
        File with graph will be overwritten if exists.
        
        :param snap_graph_ptr: initial snap graph
        :param name: name will be appended with current timestamp
        :return: MyGraph with updated graph
        """
        if name is not None:
            self._name = str_to_chars(name)
        self._snap_graph_ptr = snap_graph_ptr
        self._fingerprint = fingerprint(snap_graph_ptr)
        # Remove all stats
        self._stats_dict.clear()
        if os.path.exists(self._stat_dir()):
            shutil.rmtree(self._stat_dir())
        # Save graph if path exists
        if os.path.exists(self.path):
            self.save()
        return self

    cpdef giant_component(self):
        """
        Return a new graph containing the giant component of this graph.
        Note: all computed and saved stats will be removed.
        """
        cdef PUNGraph p = GetMxWcc[PUNGraph](self._snap_graph_ptr)
        return self.new_snap(p)

    def _check_consistency(self):
        """ Raise exception if graph has changed. """
        if not self._snap_graph_ptr.Empty():
            if fingerprint(self._snap_graph_ptr) != self._fingerprint:
                logger.warning("snap graph has changed from the one saved in %s" % self._path)

    def _stat_dir(self):
        return os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + '_stats')

    def __getitem__(self, stat):
        """ Get graph statistics. Index by str or Stat. Works only if snap graph is immutable. """
        self._check_consistency()
        if isinstance(stat, str):
            from graph_stats import Stat
            stat = Stat[stat]

        if stat in self._stats_dict:
            value = self._stats_dict[stat]
        else:
            # Try to load from file or compute
            stat_path = os.path.join(self._stat_dir(), stat.short)
            if not os.path.exists(stat_path):
                # Compute and save stats
                logger.info("Could not find stats '%s' at '%s'. Will be computed." % (stat, stat_path))
                if self._snap_graph_ptr.Empty():
                    self.load()
                from cyth.cstatistics import stat_computer
                # value = stat.computer(self)
                value = stat_computer[stat](self)

                # Save stats to file
                if not os.path.exists(os.path.dirname(stat_path)):
                    os.makedirs(os.path.dirname(stat_path))
                with open(stat_path, 'w') as f:
                    f.write(json.dumps(value))
            else:
                # Read stats from file - value or dict
                # TODO: json.loads() could be memory-consuming for large graphs
                value = json.loads(open(stat_path, 'r').read(),
                                   object_hook=lambda x: {int(k): v for k, v in x.items()} if isinstance(x, dict) else x)

            self._stats_dict[stat] = value

        return value

    def __setitem__(self, stat, value):
        self._check_consistency()
        if isinstance(stat, str):
            from graph_stats import Stat
            stat = Stat[stat]
        self._stats_dict[stat] = value

        # Save stats to file
        stat_path = os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + '_stats', stat.short)
        if not os.path.exists(stat_path):
            if not os.path.exists(os.path.dirname(stat_path)):
                os.makedirs(os.path.dirname(stat_path))
        with open(stat_path, 'w') as f:
            f.write(json.dumps(value))

    def networkit(self, node_map: dict = None):
        """ Get networkit graph, create node ids mapping (neworkit_node_id -> snap_node_id) if
        node_map is specified. Some neworkit_node_id -> None (those ids not present in snap graph).
        """
        import networkit as nk
        if not os.path.exists(self.path):
            self.save()
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

    def snap_to_networkx(self):
        import networkx as nx
        nx_graph = nx.Graph()  # undirected
        cdef int n, k
        for n in self.iter_nodes():
            nx_graph.add_node(n)
            for k in self.neighbors(n):
                nx_graph.add_edge(n, k)

        return nx_graph


def cgraph_test():
    # print("MyGraph")
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

    graph = MyGraph(path='/home/misha/workspace/crawling/data/konect/petster-hamster.ij', name='d')
    cdef TUNGraph g = deref(LoadEdgeList[PUNGraph](TStr('/home/misha/workspace/crawling/data/konect/dolphins.ij'), 0, 1))
    empty = MyGraph(name='empty')

    # graph = MyGraph.CLoad(path)
    # graph = MyGraph.Empty('')
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
