from math import sqrt

import dgl
import numpy as np
import torch.optim
from dgl.nn.pytorch import GATConv
from torch.nn.functional import log_softmax, nll_loss
from torch.nn import Module
from tqdm import tqdm
from dgl import batch as dgl_batch
from dgl import graph as dgl_graph

from crawlers.cadvanced import NodeFeaturesUpdatableCrawlerHelper
from crawlers.declarable import Declarable
from search.feature_extractors import AttrHelper
from search.predictors.simple_predictors import Predictor, import_by_name


class GNNet(Module, Declarable):
    """ GNN network based on torch.nn.Module
    """
    out_dim = 2

    def __init__(self, conv_class: str, layer_sizes: tuple, activation='torch.relu',
                 merge='mean', **conv_kwargs):
        """
        :param conv_class: name of convolution class, e.g. 'SAGEConv'.
        :param layer_sizes: layers' sizes, except output (always = GNNet.out_dim).
        :param activation: activation between layers, default is 'torch.relu'.
        :param conv_kwargs: additional arguments to DGL convolution layer.
        """
        super().__init__()
        super(Module, self).__init__(conv_class=conv_class, layer_sizes=layer_sizes,
                                     activation=activation, **conv_kwargs)
        layer_sizes = tuple(layer_sizes)  # to avoid potential list/tuple duplicates in filenames

        if 'allow_zero_in_degree' not in conv_kwargs and conv_class not in ['SAGEConv']:
            conv_kwargs['allow_zero_in_degree'] = True

        # NOTE: in case of "NameError: name '...' is not defined", explicitly add needed imports
        conv_class = import_by_name(conv_class, ["dgl.nn.pytorch"])
        # conv_class = import_by_name(conv_class, ["torch_geometric.nn"])

        assert merge in ['mean', 'cat']  # for GAT, can be 'mean' or 'cat'
        self.merge = merge

        # Stack convolutional layers, assign them to conv0, conv1, ...
        # NOTE: we do need class variables here, since torch register parameters properly looking
        # at class attributes
        self.input_dim = layer_sizes[0]
        self.n_layers = len(layer_sizes)
        prev_dim = self.input_dim

        ix = 0
        for dim in layer_sizes[1:]:
            if conv_class == GATConv and ix > 0:
                conv = conv_class(
                    (conv_kwargs['num_heads'] if self.merge == "cat" else 1) * prev_dim,
                    dim, **conv_kwargs)
            else:
                conv = conv_class(prev_dim, dim, **conv_kwargs)
            setattr(self, 'conv%s' % ix, conv)
            prev_dim = dim
            ix += 1

        # Last layer
        if conv_class == GATConv:
            num_heads = conv_kwargs['num_heads'] or 1
            conv_kwargs['num_heads'] = 1
            if ix > 0 and self.merge == 'cat':
                conv = conv_class(num_heads * prev_dim, GNNet.out_dim, **conv_kwargs)
            else:
                conv = conv_class(prev_dim, GNNet.out_dim, **conv_kwargs)
        else:
            conv = conv_class(prev_dim, GNNet.out_dim, **conv_kwargs)
        setattr(self, 'conv%s' % ix, conv)

        self.activation = eval(activation)

        self.name = "%s[%s-%s%s(%s)]" % (
            conv_class.__name__, '-'.join([str(d) for d in layer_sizes]), GNNet.out_dim,
            f"({num_heads})" if conv_class == GATConv else "", self.activation.__name__)

    def reset_parameters(self):
        for ix in range(self.n_layers):
            getattr(self, 'conv%s' % ix).reset_parameters()

    def forward(self, g, inputs):
        h = inputs
        for ix in range(0, self.n_layers):
            if ix > 0:  # Apply activation
                h = self.activation(h)

            conv = getattr(self, 'conv%s' % ix)
            h = conv(g, h)

            if isinstance(conv, GATConv):  # tensor N x H x D
                if self.merge == 'cat':
                    # Concat - N x H x D -> N x (H*D)
                    h = h.permute((1, 2, 0))
                    h = torch.concat([_ for _ in h], dim=0)
                    h = h.T

                elif self.merge == 'mean':
                    # Mean - N x H x D -> N x D
                    h = h.permute((1, 2, 0))
                    h = torch.mean(h, dim=0)
                    h = h.T

        return h


class GNNPredictor(Predictor):
    def __init__(self, conv_class: str, layer_sizes: tuple, activation='torch.relu',
                 attributes=None,
                 epochs=300, batch=100, learn_rate=0.01, name=None, **conv_kwargs):
        """

        :param conv_class:
        :param layer_sizes: hidden layers' sizes, except input (defined at first training step) and
         output (always = GNNet.out_dim). Empty `layer_sizes` corresponds to 1-layer net.
        :param activation:
        :param attributes: list of graph nodes attributes to include into feature vectors
        :param epochs: number of training epochs
        :param batch: number of dgl graphs to unite in 1 training batch
        :param learn_rate: optimizer learning rate
        :param name: name on plots
        :param name: name for plotting
        :param conv_kwargs: additional arguments to GNN convolution
        """
        if attributes is None:
            attributes = []
        attributes = sorted(attributes)
        super(GNNPredictor, self).__init__(
            conv_class=conv_class, layer_sizes=layer_sizes, activation=activation,
            attributes=attributes, epochs=epochs, batch=batch, learn_rate=learn_rate, **conv_kwargs)

        self.conv_class = conv_class
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.conv_kwargs = conv_kwargs
        self.gnn = None  # Will be initialized when graph is known
        self.name = "GNN?" if name is None else name
        self.attributes = attributes
        self.epochs = epochs
        self.batch = batch
        self.learn_rate = learn_rate

        self._trained = False

    def __str__(self):
        return self.name

    def _init_gnn(self, input_dim):
        layer_sizes = [input_dim] + list(self.layer_sizes)
        self.gnn = GNNet(self.conv_class, tuple(layer_sizes), self.activation, **self.conv_kwargs)
        self.name = self.gnn.name if self.name == "GNN?" else self.name

    @property
    def used_attributes(self):
        return self.attributes

    def reset(self):
        self.gnn.reset_parameters()

    def train(self, Xs, ys):
        """ Train the model on sequence of graphs with their inputs (Xs)
         and classification answers (ys).
        """

        # Take first max_tr_samples elements
        train_ixs = np.arange(len(Xs))

        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.learn_rate)
        pbar = tqdm(total=self.epochs, desc='%s training %s' % (
            self.name, np.bincount(np.array(ys)[train_ixs])))
        for epoch in range(self.epochs):
            avg_loss = 0
            batch = min(self.batch, len(train_ixs))
            for ix in range((len(train_ixs) + batch - 1) // batch):
                batch_ixs = train_ixs[ix * batch: (ix + 1) * batch]

                bg = dgl_batch([Xs[i][0] for i in batch_ixs])
                binputs = torch.cat([Xs[i][1] for i in batch_ixs], dim=0).float()
                target_ixs = [0]
                for i in batch_ixs[:-1]:
                    g, _ = Xs[i]
                    target_ixs.append(target_ixs[-1] + g.number_of_nodes())

                logits = self.gnn(bg, binputs)
                logp = log_softmax(logits, 1)
                loss = nll_loss(logp[target_ixs], torch.tensor([ys[i] for i in batch_ixs]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

            avg_loss /= (ix + 1)
            # print('Epoch %d | Loss: %.4f' % (epoch, avg_loss))
            pbar.update(1)

        pbar.close()
        self._trained = True

    def extract_features(self, node, crawler_helper: NodeFeaturesUpdatableCrawlerHelper) -> tuple:
        graph = crawler_helper._orig_graph
        obs_deg = crawler_helper.observed_graph.deg(node)
        if obs_deg == 0:
            return dgl.graph(([],[])), torch.tensor([])

        og = crawler_helper.observed_graph
        cc = crawler_helper.node_clust
        cnf = crawler_helper.node_cnf
        oracle = crawler_helper.oracle

        def attr_vector(id):
            res = []
            for attr in self.attributes:
                a = graph.get_attribute(id, *attr)
                res.extend(AttrHelper.one_hot(graph, attr, a))
            return res

        attr_vec_len = sum([len(AttrHelper.attribute_vals(graph, tuple(a))) for a in self.attributes])

        if self.gnn is None:
            self._init_gnn(5 + attr_vec_len)

        # Build dgl graph and input vectors for its nodes
        node_map = {}  # graph node id -> dgl node id
        src, dst = [], []
        node_map[node] = 0  # the observed node has id = 0
        N = 1  # num of nodes
        inputs = [  # input features N x F
            # [OD; CNF; CC; attrs]
            [1/sqrt(obs_deg), 1, cc[node], 0, 0, *[0] * attr_vec_len]
        ]

        # 1st neighborhood - crawled nodes
        for n1 in og.neighbors(node):
            node_map[n1] = N
            src.append(N)
            dst.append(0)
            N += 1
            t = 1 if oracle(n1, graph) == 1 else 0
            inputs.append([1/sqrt(og.deg(n1)), cnf[n1], cc[n1], t, 1-t, *attr_vector(n1)])

        # 2nd neighborhood - crawled nodes
        for n1 in og.neighbors(node):
            # neigh = n1.node
            for n2 in og.neighbors(n1):
                if n2 not in node_map:
                    node_map[n2] = N
                    N += 1
                    t = 1 if oracle(n2, graph) == 1 else 0
                    inputs.append([1/sqrt(og.deg(n2)), cnf[n2], cc[n2],
                                   t, 1-t, *attr_vector(n2)])
                src.append(node_map[n2])
                dst.append(node_map[n1])

        g = dgl_graph((torch.tensor(src), torch.tensor(dst)))

        inputs = torch.tensor(inputs, dtype=torch.float32).float()
        assert inputs.shape == torch.Size([g.number_of_nodes(), self.gnn.input_dim])
        return g, inputs

    def predict_score(self, X):
        if not self._trained:  # not trained
            # logging.warning(f"{self.name} is used without training")
            return np.random.uniform(0, 1)
        # Forward signal on dgl graph
        g, inputs = X
        logits = self.gnn(g, inputs)
        probs = torch.nn.functional.softmax(logits[0], dim=0).detach().numpy().astype(float)
        return probs[1]
