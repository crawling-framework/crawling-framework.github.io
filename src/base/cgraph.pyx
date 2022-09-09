import logging
import shutil
import json
from os import listdir
from pathlib import Path

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, postincrement as pinc

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


class StatDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, dct):
        if isinstance(dct, dict):
            try:
                dct = {int(k): v for k, v in dct.items()}
            except:
                pass
        return dct


cdef class MyGraph:
    """
    Graph object representing nodes and undirected unweighted edges.
    Uses `SNAP <https://github.com/snap-stanford/snap>`_ library for graph operations.
    Graph is stored in edgelist format.

    Fast operations via snap:

    * IO operations
    * nodes/edges addition/iteration
    * maximal degree, clustering coefficient, extraction of giant component
    * getting random nodes, neighbours

    Other methods:

    * graph statistics are available by index access. Once computed it is saved in file.
    * converting to `networkit <https://networkit.github.io/>`_ (for some statistics computing) and
      `networkx <https://networkx.github.io/>`_ (for drawing) graphs

    NOTES:

    * When graph is modified, all computed statistics are removed. If want to keep statistics in
      file and modify graph, make a copy of it.

    """
    def __init__(self, path: str, full_name: tuple= ('noname',) or str,
                 directed: bool=False, weighted: bool=False, str format='ij',
                 not_load: bool=False):
        """

        :param path: where the graph contents is stored
        :param full_name: string or tuple ([collection], [subcollection], ... , name) containing at
         least one element, the last one is treated as graph name. By default, name='noname' which
         means graph will be stored in `tmp/noname_timestamp`
        :param directed: ignored: undirected only
        :param weighted: ignored: unweighted only
        :param format: ignored: 'ij' only
        :param not_load: if True do not load the graph (useful for stats exploring). Note: any graph
         modification will lead to segfault
        """
        assert directed == False
        assert weighted == False
        assert isinstance(path, str)
        if isinstance(full_name, str):
            full_name = (full_name,)
        self._full_name = full_name
        self._directed = directed
        self._weighted = weighted
        # self._format = format  # unused
        self._stats_dict = {}
        self._attr_dict = {}  # attr_name -> {node -> value}

        cdef TUNGraph g
        # if path is None:  # is never the case in our framework?
        #     self._snap_graph_ptr = PUNGraph.New()
        #     self._fingerprint = fingerprint(self._snap_graph_ptr)
        #     # NOTE: If we define a pointer as address of object, segfault occurs ?
        self._path = path
        if not_load:
            self._snap_graph_ptr = PUNGraph.New()
            self._fingerprint = fingerprint(self._snap_graph_ptr)
        else:
            self.load()
        self._read_attributes_names()

    cpdef bint is_loaded(self):
        return self.nodes() + self.edges() > 0

    cpdef void load(self):
        logging.info("Loading graph '%s' from '%s'..." % (self.name, self._path))
        self._snap_graph_ptr = LoadEdgeList[PUNGraph](TStr(str_to_chars(self._path)), 0, 1)
        self._fingerprint = fingerprint(self._snap_graph_ptr)
        logging.info("done.")

    cpdef _read_attributes_names(self):
        """ Read attributes as file names, not load contents. """
        attr_path = self._attr_dir()
        if attr_path.exists():
            for attr in listdir(attr_path):
                path = attr_path / attr
                assert path.is_file()
                self._attr_dict[attr] = None

    cpdef _load_attribute(self, attr: str):
        """ Load {node -> value} dict from file for a given attribute. """
        assert attr in self._attr_dict
        attr_path = self._attr_dir()
        logging.info(f"Loading attribute dict for '{attr}'...")
        value = json.loads(open(attr_path / attr, 'r').read())
        self._attr_dict[attr] = {int(id): val for id, val in value.items()}
        logging.info("done.")

    cpdef void save(self):
        """ Write current edge list of snap graph into file. """
        # snap.SaveEdgeList writes commented section, we don't want it
        cdef int i, j
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._check_consistency()
        except Exception:
            # Fingerprint changed - we remove graph file and all stats
            # os.remove(self.path)
            self._stats_dict.clear()
            if self._stat_dir().exists():
                shutil.rmtree(self._stat_dir())

        with self.path.open('w') as f:
            for i, j in self.iter_edges():
                f.write("%s %s\n" % (i, j))

    cpdef MyGraph copy(self, name=None):
        """ Create and return a copy of this graph. Attributes are ignored. """
        # TODO create snap graph(N, E) could be faster
        new_name = ()
        if name:
            new_name = list(self._full_name)
            new_name[-1] = name
            # new_name[-1] += '_copy_%s' % str(time())
        from graph_io import GraphCollections
        cdef MyGraph g = GraphCollections.register_new_graph(*new_name)
        # cdef MyGraph g = MyGraph(full_name=tuple(new_name), directed=self._directed, weighted=self._weighted)
        cdef int n, i, j
        for n in self.iter_nodes():
            g.add_node(n)
        for i, j in self.iter_edges():
            g.add_edge(i, j)
        return g

    @property
    def path(self) -> Path:
        return Path(self._path)

    @property
    def name(self) -> str:
        """ Get graph name. """
        return str(self._full_name[-1])

    @property
    def full_name(self) -> tuple:
        """ Get graph full name prepended with all collections it belongs. """
        return tuple(self._full_name)

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
        Replace self snap graph with a given snap graph.
        NOTE: all computed and saved stats will be removed.
        File with graph will be overwritten if exists.
        
        :param snap_graph_ptr: initial snap graph
        :param name: name will be appended with current timestamp
        :return: MyGraph with updated graph
        """
        if name is not None:
            assert isinstance(name, [str, tuple])
            self._full_name = name
        self._snap_graph_ptr = snap_graph_ptr
        self._fingerprint = fingerprint(snap_graph_ptr)
        # Remove all stats
        self._stats_dict.clear()
        if self._stat_dir().exists():
            shutil.rmtree(self._stat_dir())
        # Save graph if path exists
        if self.path.exists():
            self.save()
        return self

    cpdef giant_component(self, inplace=False):
        """
        Return a new graph containing the giant component of this graph.

        :param inplace: if True, this graph is overwritten, all computed and saved stats will be removed.
        :return: graph
        """
        if inplace:
            return self.new_snap(GetMxWcc[PUNGraph](self._snap_graph_ptr))
        else:
            return self.copy().giant_component(inplace=True)

    def _check_consistency(self):
        """ Raise exception if graph has changed. """
        if not self._snap_graph_ptr.Empty():
            if fingerprint(self._snap_graph_ptr) != self._fingerprint:
                logger.warning("snap graph has changed from the one saved in %s" % self._path)

    def _stat_dir(self):
        path = Path(self.path.parent, self.path.name + '_stats')
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _attr_dir(self, create=False):
        path = Path(self.path.parent, self.path.name + '_attrs')
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # def __getitem__(self, stat):
    #     """ Get graph statistics. Index by str or Stat. Works only if snap graph is immutable. """
    #     self._check_consistency()
    #     if isinstance(stat, str):
    #         from graph_stats import Stat
    #         stat = Stat[stat]
    #
    #     if stat in self._stats_dict:
    #         value = self._stats_dict[stat]
    #     else:
    #         # Try to load from file or compute
    #         stat_path = self._stat_dir() / stat.short
    #         if not stat_path.exists():
    #             # Compute and save stats
    #             logger.info("Could not find stats '%s' at '%s'. Will be computed." % (stat, stat_path))
    #             if self._snap_graph_ptr.Empty():
    #                 self.load()
    #             from base.cstatistics import stat_computer
    #             # value = stat.computer(self)
    #             value = stat_computer[stat](self)
    #
    #             # Save stats to file
    #             with stat_path.open('w') as f:
    #                 json.dump(value, f)
    #         else:
    #             # Read stats from file - value or dict
    #             # TODO: json.loads() could be memory-consuming for large graphs
    #             value = json.loads(open(stat_path, 'r').read(), cls=StatDecoder)
    #                                # object_hook=lambda x: {int(k): v for k, v in x.items()} if isinstance(x, dict) else x)
    #
    #         self._stats_dict[stat] = value
    #
    #     return value
    #
    # def __setitem__(self, stat, value):
    #     self._check_consistency()
    #     if isinstance(stat, str):
    #         from graph_stats import Stat
    #         stat = Stat[stat]
    #     self._stats_dict[stat] = value
    #
    #     # Save stats to file
    #     stat_path = self._stat_dir() / stat.short
    #     with stat_path.open('w') as f:
    #         json.dump(value, f)

    def attributes(self):
        """ Return a set of available attributes names. """
        return set(self._attr_dict.keys())

    def get_attribute(self, node: int, attr: str, *keys):
        """ Get node attribute value.

        :param node: node id
        :param attr: attribute name
        :param keys: inner attribute keys, e.g. for VK graphs attr='personal', *keys='alcohol',
        :return: attribute value or None if attribute is not defined for this node
        """
        if attr not in self._attr_dict:
            raise AttributeError("Graph has no attribute '%s', possible attributes are: %s" %
                                 (attr, list(self._attr_dict.keys())))
        attr_dict = self._attr_dict[attr]
        if attr_dict is None:
            self._load_attribute(attr)
            attr_dict = self._attr_dict[attr]
        if node not in attr_dict:
            return None
            # raise KeyError("Node %s has no attribute '%s'" % (id, attr))
        res = attr_dict[node]
        try:
            # Get sub-attribute
            for key in keys:
                res = res[key]
        except (KeyError, IndexError, TypeError):
            return None
        return res

    def set_attributes(self, attr: str, attr_values: dict):
        """ Set nodes attribute values. Save to file.

        :param attr: attribute name
        :param attr_values: dict {node id -> value}
        """
        self._attr_dict[attr] = attr_values

        # Save attributes to file
        attr_path = self._attr_dir(create=True) / attr
        with attr_path.open('w') as f:
            json.dump(attr_values, f, indent=2, ensure_ascii=False)

    def to_dgl(self, node_map=dict()):
        """
        Get dgl graph representation, create node ids mapping (dgl_node_id -> snap_node_id) and write it
        to node_map variable.
        :param node_map: this variable will store {dgl_node_id -> snap_node_id} mapping
        :return: undirected (all edges are reciprocal) dgl graph
        TODO add attributes?
        """
        import numpy as np
        from dgl import DGLGraph

        reverse_map = {}  # snap_node_id -> dgl_node_id
        node_map.clear()
        for i, n in enumerate(self.iter_nodes()):
            node_map[i] = n
            reverse_map[n] = i

        src = np.ndarray(2 * self.edges(), dtype=int)
        dst = np.ndarray(2 * self.edges(), dtype=int)
        for ix, (i, j) in enumerate(self.iter_edges()):
            src[2*ix] = reverse_map[i]
            dst[2*ix] = reverse_map[j]
            src[2*ix+1] = reverse_map[j]
            dst[2*ix+1] = reverse_map[i]
        return DGLGraph((src, dst))

    def to_networkx(self, with_attributes=False):
        """
        Get networkx graph representation.
        :param with_attributes: add all attributes to networkx graph.
         Absent attributes are stored as Nones.
        :return: undirected networkx graph
        """
        import networkx as nx
        nx_graph = nx.Graph()  # undirected
        cdef int n, k
        for n in self.iter_nodes():
            nx_graph.add_node(n)
            for k in self.neighbors(n):
                nx_graph.add_edge(n, k)

        if with_attributes:
            for n in self.iter_nodes():
                attr_dict = {}
                for attr in self._attr_dict.keys():
                    val = self.get_attribute(n, attr)
                    # if val is not None:
                    attr_dict[attr] = val
                nx_graph.add_node(n, **attr_dict)

        return nx_graph
