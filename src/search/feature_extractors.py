import logging
from collections import OrderedDict
from math import sqrt

import numpy as np

from base.cgraph import MyGraph
from crawlers.declarable import Declarable

AGE_GROUPS = [15, 20, 25, 30, 35, 40, 50, 60]


class AttrHelper:
    """
    """
    _attribute_vals_cache = {}  # (graph, attribute) -> attribute_vals
    _shared_graph = {}  # tmp graph -> permanent graph

    @staticmethod
    def attribute_vals(graph: MyGraph, attribute: [str, tuple, list]) -> list:
        """ Get a set of possible attribute values or None for textual or continuous attributes.
        """
        # Convert to tuple
        attribute = attribute if isinstance(attribute, tuple) else tuple(attribute.split())

        # Check if shared
        if graph in AttrHelper._shared_graph:
            # Replace with a known permanent graph
            graph = AttrHelper._shared_graph[graph]

        # Check if cached
        if (graph, attribute) in AttrHelper._attribute_vals_cache:
            return AttrHelper._attribute_vals_cache[(graph, attribute)]

        vk_dict = {
            # TODO see community_label_mix for other textual attr
            ('age',): np.arange(0, len(AGE_GROUPS)+1),
            ('sex',): [1, 2],
            ('relation',): np.arange(1, 9),
            ('occupation', 'type'): ['work', 'university', 'school'],
            ('personal', 'political'): np.arange(1, 10),
            ('personal', 'people_main'): np.arange(1, 7),
            ('personal', 'life_main'): np.arange(1, 9),
            ('personal', 'smoking'): np.arange(1, 6),
            ('personal', 'alcohol'): np.arange(1, 6),
            # 'personal religion': ?, text
            ('schools', 'type_str'): np.arange(0, 14),
        }
        res = None
        if graph.full_name[0] == 'vk_samples':
            res = vk_dict[attribute]

        # elif graph.full_name[0] == 'synthetic':
        #     res = graph[Stat.ATTRIBUTE_VALS][attribute[0]]

        elif graph.full_name == ('attributed', 'twitter'):
            tw_dict = {
                # ('income',): ?
                ('occupation',): [str(i) for i in range(1, 10)]
            }
            res = tw_dict[attribute]

        elif graph.full_name == ('attributed', 'vk_10_classes'):
            if attribute in vk_dict:
                res = vk_dict[attribute]
            elif attribute == "community_class":
                raise NotImplementedError()
            else:  # Бизнес, Программирование, etc
                res = [1, 0]

        elif graph.full_name in {('snap', 'dblp'), ('snap', 'livejournal')}:
            res = [1, 0]

        elif graph.full_name in {('sel_harv', 'donors'), ('sel_harv', 'kickstarter')}:
            res = [1, 0]

        elif graph.full_name[0] == "citation":
            # Datasets from pytorch-geometric
            if graph.full_name[1] == "pubmed":
                if attribute == ("feature",):
                    res = np.arange(500)
                elif attribute == ("label",):
                    res = np.arange(3)
            elif graph.full_name[1] == "dblp":
                if attribute == ("feature",):
                    res = np.arange(1639)
                elif attribute == ("label",):
                    res = np.arange(4)
            elif graph.full_name[1] == "cora":
                if attribute == ("feature",):
                    res = np.arange(8710)
                elif attribute == ("label",):
                    res = np.arange(70)
            elif graph.full_name[1] == "cora_ml":
                if attribute == ("feature",):
                    res = np.arange(2879)
                elif attribute == ("label",):
                    res = np.arange(7)
            elif graph.full_name[1] == "citeseer":
                if attribute == ("feature",):
                    res = np.arange(602)
                elif attribute == ("label",):
                    res = np.arange(6)

        else: raise NotImplementedError(f"Attribute values for graph {graph.full_name}")

        AttrHelper._attribute_vals_cache[(graph, attribute)] = res
        return res

    @staticmethod
    def share_graph_attributes(src_graph: MyGraph, dst_graph: MyGraph):
        """ Use attributes from source graph when work with destination graph.
        For instance, destination graph is the observed/crawled part of source graph.
        """
        AttrHelper._shared_graph[dst_graph] = src_graph

    @staticmethod
    def one_hot(graph: MyGraph, attribute: [str, tuple, list], value, add_none=False):
        """ 1-hot encoding feature. If no such value, return all zeros or with 1 it in last element.
        :param graph: MyGraph
        :param attribute: attribute name, e.g. 'sex', ('personal', 'smoking').
        :param value: value of this attribute. If a list, a multiple-hot vector will be constructed.
        :param add_none: if True, last element of returned vector encodes undefined or
         not-in-the-list attribute value. If False, in case of undefined value, vector of all zeros
         will be returned.
        :return: vector of length = len(AttrHelper.attribute_vals(graph, attribute)) or +1 if
         add_none is True.
        """
        # TODO if called often, we can cache it
        allowed_vals = AttrHelper.attribute_vals(graph, attribute)

        if isinstance(value, list):  # multiple-hot vector
            vals = set(value)
            if add_none:
                none = True
                res = np.zeros(len(allowed_vals)+1)
            else:
                res = np.zeros(len(allowed_vals))
            for pos, val in enumerate(allowed_vals):
                if val in vals:
                    res[pos] = 1
                    none = False
            if add_none and none:
                res[-1] = 1
            return res

        if add_none:
            res = np.zeros(len(allowed_vals)+1)
            for pos, val in enumerate(allowed_vals):
                if val == value:
                    res[pos] = 1
                    return res
            res[-1] = 1  # last element means value is undefined or not in the list
            return res
        else:
            res = np.zeros(len(allowed_vals))
            for pos, val in enumerate(allowed_vals):  # NOTE: iteration over set not list
                # FIXME str or int?
                if val == value:
                    res[pos] = 1
                    break
            return res  # all zeros here

    @staticmethod
    def node_one_hot(graph: MyGraph, node, attribute: [str, tuple, list], add_none=False):
        """ 1-hot encoding feature for given node and attribute.
        If no such value, return all zeros or with 1 it in last element.

        :param graph: MyGraph
        :param attribute: attribute name, e.g. 'sex', ('personal', 'smoking').
        :param value: value of this attribute.
        :param add_none: if True, last element of returned vector encodes undefined or
         not-in-the-list attribute value. If False, in case of undefined value, vector of all zeros
         will be returned.
        :return: vector of length = len(AttrHelper.attribute_vals(graph, attribute)) or +1 if
         add_none is True.
        """
        # Convert to tuple
        attribute = attribute if isinstance(attribute, tuple) else tuple(attribute.split())
        val = graph.get_attribute(node, *attribute)
        return AttrHelper.one_hot(graph, attribute, val, add_none=add_none)


class NeighborsFeatureExtractor(Declarable):
    """
    Feature vector for an observed node based on its neighbors.
    To extract features it needs (node, crawler_helper).

    from https://arxiv.org/pdf/1703.05082.pdf
    Structure-and-attribute blends:
    + number and fraction of target neighbors,- replaced by TNF
    + number and fraction of triangles formed with two non-target (and with two target) neighbors, - added 3 types
    + number and fraction of neighbors mostly surrounded by target nodes,- replaced by TNF2
    * fraction of neighbors that exhibit each node attribute,
    + probability of finding a target exactly after two random walk steps from border node - replaced by TNF 1.
    """
    # Caches last several calculations to speed up - useful in multipredictors
    _call_cache = OrderedDict()  # {(self.name, node, crawler_helper) -> computed feature vector}
    _max_cache_size = 100

    def __init__(self,
                 ix=False, od=False, cc=False, cnf=False, tnf=False, tri=False,
                 # rw2=False,
                 attributes=None,
                 neighs1=False, neighs2=False, hist: int=0):
        """
        :param ix: include index (size of crawled set).
        :param od: include node degree.
        :param cc: include node clustering coefficient.
        :param cnf: include crawled neighbors fraction.
        :param tnf: include target neighbors fraction.
        :param tri: include 3 types of triangles.
        #  from the observed node.
        :param attributes: include attributes (list).
        :param neighs1: add previous features for all 1st neighbors where possible.
        :param neighs2: add previous features for all 2nd neighbors where possible.
        :param hist: applied to all features computed for 1st and 2nd neighbors:
         0 - compute average, higher positive integer - compute discrete distribution with 'hist'
         number of bins (hist=1 is senseless).
        """
        if attributes is None:
            attributes = []
        super(NeighborsFeatureExtractor, self).__init__(
            ix=ix, od=od, cc=cc, cnf=cnf, tnf=tnf, tri=tri, attributes=attributes,
            neighs1=neighs1, neighs2=neighs2, hist=hist)
        self.ix = ix
        self.od = od
        self.cc = cc
        self.cnf = cnf
        self.tnf = tnf
        self.tri = tri
        self.attributes = attributes
        self.neighs1 = neighs1
        self.neighs2 = neighs2
        self.hist = hist
        assert hist in {0, 2, 3, 4, 5}  # hist = 1 is senseless (always gives 1)
        name = (",ix" if ix else "") + (",OD" if od else "") + (",CC" if cc else "") +\
            (",CNF" if cnf else "") + (",TNF" if tnf else "") + (",Tri" if tri else "") +\
            ("".join(f",{a}" for a in attributes)) +\
            ("+overN1" if neighs1 else "") + ("+overN2" if neighs2 else "") +\
            (("-avg" if hist == 0 else f"-hist{hist}") if neighs1 or neighs2 else "")
        # if len(name) == 0:
        #     raise RuntimeError("Can't work with 0 features, add some!")
        self.name = name[1:]

        self.feature_names = None  # will be created at the first call since we need graph for that

    def _create_feature_names(self, graph):
        # Add feature names - must be same order as when forming feature vector
        # 1) ix
        self.feature_names = []  # feature elements names
        if self.ix:
            self.feature_names.append('ix')

        # For each attribute list all its values and undefined
        attr_fnames = []
        for a in self.attributes:
            for v in AttrHelper.attribute_vals(graph, a):
                attr_fnames.append((True, f"{a}={v}"))
            attr_fnames.append((True, f"{a}_is_undef"))

        # 2) features for observed node
        fnames = [
            (self.od, "OD"),
            (self.cc, "CC"),
            (self.tnf, "TNF"),
            (self.tri, "Tri2"),
            (self.tri, "Tri1"),
            (self.tri, "Tri0"),
        ] + attr_fnames
        for f, name in fnames:
            if f:
                self.feature_names.append(name)

        # 3) features for 1st neighbors of observed node
        if self.neighs1:
            fnames = [
                (self.od, "OD1"),
                (self.cc, "CC1"),
                (self.cnf, "CNF1"),
                (self.tnf, "TNF1"),
                # (tri, "Tri21"),
                # (tri, "Tri11"),
                # (tri, "Tri01"),
            ] + attr_fnames
            if self.hist == 0:
                for f, name in fnames:
                    if f:
                        self.feature_names.append(f"{name}-avg")
            else:  # 2-5
                for f, name in fnames:
                    if f:
                        for bin in range(self.hist):
                            self.feature_names.append(f"{name}-{bin}:{self.hist}")

        # 4) features for 2nd neighbors of observed node
        if self.neighs2:
            fnames = [
                (self.od, "OD2"),
                (self.cc, "CC2"),
                (self.cnf, "CNF2"),
                (self.cnf, "TNF2"),
            ]
            # ] + attr_fnames  # TODO - isn't too much ?
            if self.hist == 0:
                for f, name in fnames:
                    if f:
                        self.feature_names.append(f"{name}-avg")
            else:  # 2-5
                for f, name in fnames:
                    if f:
                        for bin in range(self.hist):
                            self.feature_names.append(f"{name}-{bin}:{self.hist}")

    def __str__(self):
        return self.name

    def _hist(self, values):
        values = np.array((values))
        if not (np.all(0 <= values) & np.all(values <= 1.001)):
            logging.error(f"Feature values exceed [0;1] interval for hist: {values}")
        if self.hist == 0:
            return [0] if len(values) == 0 else [np.mean(values)]
        if len(values) == 0:
            return [0] * self.hist
        hs = np.histogram(values, bins=self.hist, range=(0, 1), density=True)[0] / self.hist
        return hs

    def __call__(self, node, crawler_helper):
        # NOTE: faster version without creating Neighborhood
        if self.feature_names is None:
            self._create_feature_names(crawler_helper.orig_graph)

        # Check in _call_cache
        call_key = (self.name, node, crawler_helper)
        if call_key in NeighborsFeatureExtractor._call_cache:
            return NeighborsFeatureExtractor._call_cache[call_key]

        og = crawler_helper.observed_graph
        obs_deg = og.deg(node)
        if obs_deg == 0:
            return [0] * len(self.feature_names)

        cc = crawler_helper.node_clust
        cnf = crawler_helper.node_cnf
        tnf = crawler_helper.node_tnf
        anf = crawler_helper.attr_node_vec  # {attr -> {node -> fractions vector}}

        def f_od(od):
            return 1/sqrt(od)

        # Form feature vector - must be same order as listed in feature names
        # 1) ix
        X = []  # feature vector
        if self.ix:
            index = len(crawler_helper.crawled_set)
            X.append(1/np.log2(index + 1))
            # for index in [1, 10, 100, 1000, 10000, 100000]
            # its value: 1.0, 0.289, 0.150, 0.100, 0.075, 0.060

        neighs1 = set(n for n in og.neighbors(node))
        neighs2 = None

        # 2) features for the observed node
        if self.od:
            X.append(f_od(obs_deg))
        if self.cc:
            X.append(cc[node])
        if self.tnf:
            X.append(tnf[node])
        if self.tri:
            graph = crawler_helper.orig_graph
            oracle = crawler_helper.oracle
            # Get all triangles
            pairs = set()  # nodes (x, y) s.t. y >= x
            for n1 in neighs1:
                x = n1
                for n in og.neighbors(n1):
                    if n in neighs1:
                        pairs.add((min(x, n), max(x, n)))
            triangles = {0: 0, 1: 0, 2: 0}
            for x, y in pairs:
                ts = sum(1 if oracle(_, graph) == 1 else 0 for _ in [x, y])
                triangles[ts] += 1

            t = sum(triangles[i] for i in [0, 1, 2])
            t = 1 if t == 0 else t
            X.append(triangles[2] / t)  # TODO normalize???
            X.append(triangles[1] / t)
            X.append(triangles[0] / t)

        for a in self.attributes:
            X.extend(anf[a][node])

        def get_neighs2(neighs2):
            if neighs2 is None:
                neighs2 = []
                n12ids = set(n for n in og.neighbors(node))  # 1st neighs, will be ignored later
                for n1 in og.neighbors(node):
                    for n2 in og.neighbors(n1):
                        if n2 in n12ids:
                            continue
                        n12ids.add(n2)
                        neighs2.append(n2)
            return neighs2

        # 3) features for 1st neighbors of the observed node
        if self.neighs1:
            if self.od:
                vals = [f_od(og.deg(n)) for n in neighs1]
                X.extend(self._hist(vals))
            if self.cc:
                vals = [cc[n] for n in neighs1]
                X.extend(self._hist(vals))
            if self.cnf:
                vals = [cnf[n] for n in neighs1]
                X.extend(self._hist(vals))
            if self.tnf:
                vals = [tnf[n] for n in neighs1]
                X.extend(self._hist(vals))
            # if self.tri:
            # Need to know 3rd neighbors or observed_graph
            #     # Get all triangles
            #     neighs1 = neighborhood.neighs
            #     neighs2 = get_neighs2(neighs2)
            #     n2ids = set(n.node for n in neighs2)
            #     node_pairs = {}  # node from 1st neighs -> 2 nodes from 2nd neighs (x, y) s.t. y >= x
            #     for n1 in neighs1:
            #         pairs = node_pairs[n1] = set()
            #         for n11 in n1.neighs:
            #             x = n11.node
            #             for n12 in n11.neighs:
            #                 ...
            #     triangles = {0: 0, 1: 0, 2: 0}
            #     for x, y in pairs:
            #         triangles[oracle(x, graph) + oracle(y, graph)] += 1
            #
            #     X.append(triangles[2])
            #     X.append(triangles[1])
            #     X.append(triangles[0])

            for a in self.attributes:
                vals = [anf[a][n] for n in neighs1]
                for x in np.apply_along_axis(self._hist, axis=0, arr=vals):
                    X.extend(x)

        # 4) features for 2nd neighbors of the observed node
        if self.neighs2:
            # Get 2nd neighbors
            neighs2 = get_neighs2(neighs2)
            if self.od:
                vals = [f_od(og.deg(n)) for n in neighs2]
                X.extend(self._hist(vals))
            if self.cc:
                vals = [cc[n] for n in neighs2]
                X.extend(self._hist(vals))
            if self.cnf:
                vals = [cnf[n] for n in neighs2]
                X.extend(self._hist(vals))
            if self.tnf:
                vals = [tnf[n] for n in neighs2]
                X.extend(self._hist(vals))

            # TODO - isn't too much ?
            # for a in self.attributes:
            #     vals = [anf[a][n] for n in neighs2]
            #     for x in np.apply_along_axis(self._hist, axis=0, arr=vals):
            #         X.extend(x)

        if len(X) == 0:
            raise RuntimeError("Feature vector has 0 elements, add some features!")
        assert len(self.feature_names) == len(X)

        # Save in _call_cache
        NeighborsFeatureExtractor._call_cache[call_key] = X
        if len(NeighborsFeatureExtractor._call_cache) > NeighborsFeatureExtractor._max_cache_size:
            # Remove the oldest element
            NeighborsFeatureExtractor._call_cache.popitem(last=False)

        return X
