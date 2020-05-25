import logging
import os.path

import networkx as nx
import numpy as np
import snap

from utils import TMP_GRAPHS_DIR


def fingerprint(snap_graph):
    """ Graph fingerprint to make sure briefly if it has changed.

    :param snap_graph:
    :return: (|V|, |E|)
    """
    return snap_graph.GetNodes(), snap_graph.GetEdges()


class MyGraph(object):
    def __init__(self, path=None, name='noname', directed=False, weighted=False, format='ij'):
        self._snap_graph = None
        self.path = path
        self.name = name
        self.directed = directed
        self.weighted = weighted
        self.format = format
        self._fingerprint = None
        self._stats_dict = {}

        if path is None:
            from datetime import datetime
            self.path = os.path.join(TMP_GRAPHS_DIR, "%s_%s" % (name, datetime.now()))
            self._snap_graph = snap.TNGraph.New() if directed else snap.TUNGraph.New()
        else:
            self.snap()

    # @classmethod
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

    def _check_consistency(self):
        """ Raise exception if graph has changed. """
        f = fingerprint(self.snap)
        if fingerprint(self.snap) != self._fingerprint:
            raise Exception("snap graph has changed from the one saved in %s" % self.path)

    @property
    def snap(self):
        if self._snap_graph is None:
            self._snap_graph = snap.LoadEdgeList(
                snap.PNGraph if self.directed else snap.PUNGraph, self.path, 0, 1)
            self._fingerprint = fingerprint(self._snap_graph)
        return self._snap_graph

    def nodes(self):
        return self._snap_graph.GetNodes()

    def edges(self):
        return self._snap_graph.GetEdges()

    def add_node(self, node: int):
        return self._snap_graph.AddNode(node)

    def add_edge(self, i: int, j: int):
        return self._snap_graph.AddEdge(i, j)

    def neighbors(self, node: int):
        """
        List of neighbors of the given node in this graph. (snap wrapper for simplicity)

        :param node: node id
        :return: list of ids
        """
        if self.directed:
            raise NotImplementedError("For directed graph and all neighbors, take GetInEdges + GetOutEdges")
        return list(self.snap.GetNI(int(node)).GetOutEdges())

    def random_node(self):
        return self.random_nodes(1)

    def random_nodes(self, count=1):
        return np.random.choice([n.GetId() for n in self._snap_graph.Nodes()], count, replace=False)

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
                logging.info("Could not find stats '%s' at '%s'. Will be computed." %
                             (stat, stat_path))
                from statistics import stat_computer
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

            # raise KeyError("Unknown item type: %s" % type(item))
        return value

    def save(self, new_path=None):
        """ Write current edge list of snap graph into file. """
        s = self._snap_graph
        assert s
        if new_path is None:
            new_path = self.path
        if new_path == self.path:
            logging.warning("Graph file '%s' will be overwritten." % self.path)
        # snap.SaveEdgeList writes commented section, we don't want it
        with open(new_path, 'w') as f:
            for e in s.Edges():
                f.write("%s %s\n" % (e.GetSrcNId(), e.GetDstNId()))
        self._fingerprint = fingerprint(s)

    @property  # Denis:  could be useful to handle nx version of graph
    def snap_to_networkx(self):
        nx_graph = nx.Graph()
        for NI in self.snap.Nodes():
            nx_graph.add_node(NI.GetId())
            for Id in NI.GetOutEdges():
                nx_graph.add_edge(NI.GetId(), Id)

        return nx_graph
