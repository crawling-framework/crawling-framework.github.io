import abc
from abc import abstractmethod
import numpy as np

from base.cgraph import MyGraph
from crawlers.declarable import Declarable, declaration_to_filename
from search.feature_extractors import AttrHelper


class Oracle(Declarable):
    """
    Function evaluating nodes according to the target.
    In case of classification task for each node returns 1 if node is target, and 0 otherwise.
    For regression task it returns some score for each node.
    """
    __metaclass__ = abc.ABCMeta

    # Static variable to store target sets. Each set is shared for all equivalent Oracle instances.
    # FIXME could eat memory in an experiment with many graphs
    # {oracle filename -> {graph -> target set}}
    _oracle_graph_targetset = {}

    target_set_max_graph_size = 1e7  # If graph has more nodes, do not compute target set directly

    def __init__(self, name=None, **kwargs):
        super(Oracle, self).__init__(**kwargs)
        self.name = "oracle" if name is None else name

        on = declaration_to_filename(self.declaration)
        if on not in self._oracle_graph_targetset:
            self._oracle_graph_targetset[on] = {}
        self._graph_target_set = self._oracle_graph_targetset[on]  # graph -> target_set

    def __str__(self):
        return self.name

    @abstractmethod
    def __call__(self, id, graph):
        """ Evaluate node according to the target. """
        raise NotImplementedError()

    def _compute_target_set(self, graph: MyGraph):
        """ Compute and memorize target set.
        Use it when remembering set is easier than calling many times.
        """
        if graph.nodes() > self.target_set_max_graph_size:
            # Graph is too large
            raise RuntimeError(f"Graph size exceeds threshold to get target set, "
                               f"N = {graph.nodes()} > {self.target_set_max_graph_size}")
        # Check all nodes
        target_set = set(n for n in graph.iter_nodes() if self(n, graph) == 1)
        self._graph_target_set[graph.full_name] = target_set

    def target_set(self, graph):
        """ Returns a target set if known. Else returns None. """
        g = graph.full_name
        if g not in self._graph_target_set:
            self._compute_target_set(graph)

        if g in self._graph_target_set:
            return self._graph_target_set[g]
        else:
            return None

    def target_set_size(self, graph):
        """ Returns target set size if known. Else returns None. """
        if self.target_set(graph) is not None:
            return len(self.target_set(graph))
        else:
            return None

    def random_node(self, graph, **kwargs):
        """ Choose a random node from target set. """
        if self.target_set(graph) is not None:
            # NOTE converting to list each time, not supposed to use often
            return np.random.choice(list(self._graph_target_set[graph.full_name]))
        else:
            raise RuntimeError(f"Target set is unknown for graph {graph.full_name}")


class HasAttrValueOracle(Oracle):
    """
    Target nodes are nodes whose attribute has a specified value.
    For nodes with undefined attributes returns None.
    """
    def __init__(self, attribute: tuple, value, allowed_only=True):
        """
        :param attribute: attribute name as tuple
        :param value: which value is target
        :param allowed_only: if True then None will be returned for attribute values that are
         not allowed (as returned by AttrHelper.attribute_vals())
        """
        super(HasAttrValueOracle, self).__init__(
            name='%s_is_%s' % (attribute, value), attribute=attribute, value=value, allowed_only=allowed_only)
        self.attr = attribute
        self.value = value
        self.allowed_only = allowed_only

    def __call__(self, id, graph: MyGraph):
        y = graph.get_attribute(id, *self.attr)
        if self.allowed_only and y not in AttrHelper.attribute_vals(graph, self.attr):
            # Attribute takes value different from allowed ones - thus we cannot judge
            # any prediction
            return None
        return 1 if y == self.value else 0
