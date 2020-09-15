import glob
import json
import logging
import os
import shutil
from bisect import bisect_left, bisect_right
from math import sqrt, ceil

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from running.metrics_and_runner import TopCentralityMetric, Metric, exponential_batch_generator
from graph_io import GraphCollections
from graph_stats import Stat

from crawlers.cbasic import Crawler, definition_to_filename, filename_to_definition
from utils import RESULT_DIR


def compute_aucc(xs, ys):
    # from sklearn.metrics import auc
    # return auc(xs, ys)
    assert len(xs) == len(ys) > 0
    xs = xs / xs[-1]
    res = xs[0] * ys[0] / 2
    for i in range(1, len(xs)):
        res += (xs[i] - xs[i-1]) * (ys[i-1] + ys[i]) / 2
    return res


def compute_waucc(xs, ys):
    # res = compute_aucc(np.log(xs), ys)
    assert len(xs) == len(ys) > 0
    xs = xs / xs[-1]
    res = 0 if xs[0] == 0 else ys[0]
    norm = 0 if xs[0] == 0 else 1
    for i in range(1, len(xs)):
        res += (xs[i] - xs[i-1]) * (ys[i-1] + ys[i]) / 2 / xs[i]
        norm += (xs[i] - xs[i-1]) / xs[i]
    return res / norm


# A graph needed just to generate pretty short crawlers and metrics names for ResultsMerger.
example_graph = GraphCollections.get('example', collection='other')  # petster-hamster with 2000 nodes
# FIXME will fail for MultiCrawler with count > g.nodes()


LINESTYLES = ['-', '--', ':', '-.']
COLORS = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
          'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
          'pink', 'lime', 'wheat', 'lightsteelblue']


class ResultsMerger:
    """
    ResultsMerger can aggregate and plot results saved in files.
    Process all combinations of G graphs x C crawlers x M metrics. Averages over n instances of each.
    All missed instances are just ignored.

    Plotting functions:

    * draw_by_crawler - Draw M x G table of plots with C lines each. Ox - crawling step, Oy - metric value.
    * draw_by_metric_crawler - Draw G plots with C x M lines each. Ox - crawling step, Oy - metric value.
    * draw_by_metric - Draw C x G table of plots with M lines each. Ox - crawling step, Oy - metric value.
    * draw_aucc - Draw G plots with M lines. Ox - C crawlers, Oy - (w)AUCC value (M curves with error bars).
    * draw_winners - Draw C stacked bars (each of M elements). Ox - C crawlers, Oy - number of wins (among G) by (w)AUCC
      value.

    Additional functions:

    * missing_instances - Calculate how many instances of all configurations are missing.
    * remove_files - Remove all saved instances for current graphs, crawlers, metrics.
    * print_results_as_table - Remove all saved instances for current graphs, crawlers, metrics.

    NOTES:

    * x values must be the same for all files and are the ones generated by `exponential_batch_generator()` from
      running/metrics_and_runner.py
    * it is supposed that for all instances values lists are of equal lengthes (i.e. budgets). Otherwise normalisation
      and aggregation may fail. If so, use `x_lims` parameter for the control.

    """
    def __init__(self, graph_names, crawler_defs, metric_defs, n_instances=1, x_lims=None):
        """
        :param graph_names: list of graphs names
        :param crawler_defs: list of crawlers definitions
        :param metric_defs: list of metrics definitions
        :param n_instances: number of instances to average over
        :param x_lims: use only specified x-limits for all plots unless another value is specified in plotting function
        """
        self.graph_names = graph_names
        self.crawler_names = []  # list(map(definition_to_filename, crawler_defs))
        self.metric_names = []  # list(map(definition_to_filename, metric_defs))
        self.labels = {}  # pretty short names to draw in plots

        # Generate pretty names for crawlers and metrics for plots
        for md in metric_defs:
            m = Metric.from_definition(example_graph, md)
            f = definition_to_filename(m.definition)
            self.metric_names.append(f)
            self.labels[f] = m.name
        for cd in crawler_defs:
            c = Crawler.from_definition(example_graph, cd)
            f = definition_to_filename(c.definition)
            self.crawler_names.append(f)
            self.labels[f] = c.name
        # print(self.labels)

        self.n_instances = n_instances
        self.x_lims = x_lims
        self.instances = {}  # instances[graph][crawler][metric] -> count
        self.contents = {}  # contents[graph][crawler][metric]: 'x' -> [], 'ys' -> [[]*n_instances], 'avy' -> []
        self.auccs = {}  # auccs[graph][crawler][metric]: 'AUCC' -> [AUCC], 'wAUCC' -> [wAUCC]

        self._read()
        # missing = self.missing_instances()
        # if len(missing) > 0:
        #     logging.warning("Missing instances, will not be plotted:\n%s" % json.dumps(missing, indent=2))

    @staticmethod
    def names_to_path(graph_name: str, crawler_name: str, metric_name: str):
        """ Returns file pattern e.g.
        '/home/misha/workspace/crawling/results/ego-gplus/POD(batch=1)/TopK(centrality=BtwDistr,measure=Re,part=crawled,top=0.01)/\*.json'
        """
        path = os.path.join(RESULT_DIR, graph_name, crawler_name, metric_name, "*.json")
        return path

    def _read(self):
        total = len(self.graph_names) * len(self.crawler_names) * len(self.metric_names)
        pbar = tqdm(total=total, desc='Reading history')
        self.instances.clear()
        # self.contents.clear()
        for g in self.graph_names:
            self.instances[g] = {}
            self.contents[g] = {}
            for c in self.crawler_names:
                self.instances[g][c] = {}
                self.contents[g][c] = {}
                for m in self.metric_names:
                    path_pattern = ResultsMerger.names_to_path(g, c, m)
                    path_pattern = path_pattern.replace('[', '[[]')  # workaround for glob since '[' is a special symbol for it
                    paths = glob.glob(path_pattern)
                    self.instances[g][c][m] = len(paths)
                    self.contents[g][c][m] = contents = {}

                    count = len(paths)
                    contents['x'] = []
                    contents['ys'] = ys = [[]]*count
                    contents['avy'] = []

                    xvalues = [0]
                    total = 0
                    for b in exponential_batch_generator():
                        total += b
                        xvalues.append(total)
                    if self.x_lims:
                        x0, x1 = self.x_lims
                        i0 = bisect_right(xvalues, x0)-1
                        i1 = bisect_left(xvalues, x1)
                    else:
                        i0 = 0
                        i1 = len(xvalues)

                    for inst, p in enumerate(paths):
                        with open(p, 'r') as f:
                            imported = json.load(f)
                        if len(contents['x']) == 0:
                            xs = np.array(sorted([int(x) for x in list(imported.keys())]))[i0: i1]
                            # xs = xs / xs[-1]
                            contents['x'] = xs
                        if inst == 0:
                            contents['avy'] = np.zeros(len(xs))
                        ys[inst] = np.array([float(x) for x in list(imported.values())])[i0: i1]
                        contents['avy'] += np.array(ys[inst]) / count

                    pbar.update(1)
        pbar.close()

    def remove_files(self):
        """ Remove all saved instances for current graphs X crawlers X metrics.
        """
        total = len(self.graph_names) * len(self.crawler_names) * len(self.metric_names)
        pbar = tqdm(total=total, desc='Removing history')
        folder = None
        removed = 0
        from os.path import dirname as parent
        from os.path import exists as exist
        for g in self.graph_names:
            for c in self.crawler_names:
                for m in self.metric_names:
                    folder = os.path.dirname(ResultsMerger.names_to_path(g, c, m))
                    if exist(folder):
                        removed += 1
                    shutil.rmtree(folder, ignore_errors=True)
                    pbar.update(1)

                # Remove parent folder if exists and empty
                if exist(parent(folder)) and not os.listdir(parent(folder)):
                    os.rmdir(parent(folder))

            # Remove parent folder if exists and empty
            if exist(parent(parent(folder))) and not os.listdir(parent(parent(folder))):
                os.rmdir(parent(parent(folder)))
        pbar.close()
        logging.info("Removed %s folders" % removed)
        self.instances.clear()
        self.contents.clear()

    def missing_instances(self) -> dict:
        """ Return dict of instances where computed < n_instances.

        :return: result[graph][crawler][metric] -> missing count
        """
        missing = {}
        for g in self.graph_names:
            missing[g] = {}
            for c in self.crawler_names:
                missing[g][c] = {}
                for m in self.metric_names:
                    present = self.instances[g][c][m]
                    if self.n_instances > present:
                        missing[g][c][m] = self.n_instances - present

                if len(missing[g][c]) == 0:
                    del missing[g][c]

            if len(missing[g]) == 0:
                del missing[g]

        # print(json.dumps(missing, indent=2))
        return missing

    def draw_by_crawler(self, x_lims=None, x_normalize=True, draw_error=True, scale=3):
        """
        Draw M x G table of plots with C lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers.
        Ox - crawling step, Oy - metric value.

        :param x_lims: x-limits for plots. Overrides x_lims passed in constructor
        :param x_normalize: if True, x values are normalized to be from 0 to 1
        :param draw_error: if True, fill standard deviation area around the averaged crawling curve
        :param scale: size of plots (default 3)
        """
        x_lims = x_lims or self.x_lims

        G = len(self.graph_names)
        M = len(self.metric_names)
        nrows, ncols = M, G
        if M == 1:
            nrows = int(sqrt(G))
            ncols = ceil(G / nrows)
        if G == 1:
            nrows = int(sqrt(M))
            ncols = ceil(M / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=x_normalize, sharey=True, num="By crawler", figsize=(1 + scale * ncols, scale * nrows))

        total = len(self.graph_names) * len(self.crawler_names) * len(self.metric_names)
        pbar = tqdm(total=total, desc='Plotting by crawler')
        aix = 0
        for i, m in enumerate(self.metric_names):
            for j, g in enumerate(self.graph_names):
                if nrows > 1 and ncols > 1:
                    plt.sca(axs[aix // ncols, aix % ncols])
                elif nrows * ncols > 1:
                    plt.sca(axs[aix])
                if aix % G == 0:
                    plt.ylabel(self.labels[m])
                if i == 0:
                    plt.title(g)
                if aix // ncols == nrows-1:
                    plt.xlabel('Nodes fraction crawled' if x_normalize else 'Nodes crawled')
                aix += 1

                if x_lims:
                    plt.xlim(x_lims)
                for k, c in enumerate(self.crawler_names):
                    contents = self.contents[g][c][m]
                    # Draw each instance
                    # for inst in range(len(contents['ys'])):
                    #     plt.plot(contents['x'], contents['ys'][inst], color=colors[k % len(colors)], linewidth=0.5, linestyle=':')
                    # Draw variance
                    xs = contents['x']
                    if x_normalize and len(xs) > 0:
                        xs = xs / xs[-1]
                    if len(xs) > 0 and draw_error:
                        error = np.var(contents['ys'], axis=0) ** 0.5
                        plt.fill_between(xs, contents['avy'] - error, contents['avy'] + error, color=COLORS[k % len(COLORS)], alpha=0.2)
                    plt.plot(xs, contents['avy'], color=COLORS[k % len(COLORS)], linewidth=1, label=self.labels[c])
                    pbar.update(1)
        pbar.close()
        plt.legend()
        plt.tight_layout()

    def draw_by_metric(self, x_lims=None, x_normalize=True, draw_error=True, scale=3):
        """
        Draw C x G table of plots with M lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers
        Ox - crawling step, Oy - metric value.
        """
        x_lims = x_lims or self.x_lims

        G = len(self.graph_names)
        C = len(self.crawler_names)
        nrows, ncols = C, G
        if C == 1:
            nrows = int(sqrt(G))
            ncols = ceil(G / nrows)
        if G == 1:
            nrows = int(sqrt(C))
            ncols = ceil(C / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=x_normalize, sharey=True, num="By metric", figsize=(1 + scale * ncols, scale * nrows))

        total = len(self.graph_names) * len(self.crawler_names) * len(self.metric_names)
        pbar = tqdm(total=total, desc='Plotting by crawler')
        aix = 0
        for i, c in enumerate(self.crawler_names):
            for j, g in enumerate(self.graph_names):
                if nrows > 1 and ncols > 1:
                    plt.sca(axs[aix // ncols, aix % ncols])
                elif nrows * ncols > 1:
                    plt.sca(axs[aix])
                if aix % G == 0:
                    plt.ylabel(self.labels[c])
                if i == 0:
                    plt.title(g)
                if aix // ncols == nrows-1:
                    plt.xlabel('Nodes fraction crawled' if x_normalize else 'Nodes crawled')
                aix += 1

                if x_lims:
                    plt.xlim(x_lims)
                for k, m in enumerate(self.metric_names):
                    contents = self.contents[g][c][m]
                    # Draw each instance
                    # for inst in range(len(contents['ys'])):
                    #     plt.plot(contents['x'], contents['ys'][inst], color=colors[k % len(colors)], linewidth=0.5, linestyle=':')
                    # Draw variance
                    xs = contents['x']
                    if x_normalize and len(xs) > 0:
                        xs = xs / xs[-1]
                    if len(xs) > 0 and draw_error:
                        error = np.var(contents['ys'], axis=0) ** 0.5
                        plt.fill_between(xs, contents['avy'] - error, contents['avy'] + error, color=COLORS[k % len(COLORS)], alpha=0.2)
                    plt.plot(xs, contents['avy'], color=COLORS[k % len(COLORS)], linewidth=1, label=self.labels[m])
                    pbar.update(1)
        pbar.close()
        plt.legend()
        plt.tight_layout()

    def draw_by_metric_crawler(self, x_lims=None, x_normalize=True, swap_coloring_scheme=False, draw_error=True, scale=3):
        """
        Draw G plots with CxM lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers.
        Ox - crawling step, Oy - metric value.

        :param x_lims: x-limits for plots. Overrides x_lims passed in constructor
        :param x_normalize: if True, x values are normalized to be from 0 to 1
        :param swap_coloring_scheme: by default metrics differ in linestyle, crawlers differ in color. Set True to swap
        :param draw_error: if True, fill standard deviation area around the averaged crawling curve
        :param scale: size of plots (default 3)
        """
        x_lims = x_lims or self.x_lims

        G = len(self.graph_names)
        nrows = int(sqrt(G))
        ncols = ceil(G / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=x_normalize, sharey=True, num="By metric and crawler", figsize=(1 + scale * ncols, scale * nrows))

        total = len(self.graph_names) * len(self.crawler_names) * len(self.metric_names)
        pbar = tqdm(total=total, desc='Plotting by metric crawler')
        aix = 0
        for j, g in enumerate(self.graph_names):
            if nrows > 1 and ncols > 1:
                plt.sca(axs[aix // ncols, aix % ncols])
            elif nrows * ncols > 1:
                plt.sca(axs[aix])
            if aix % ncols == 0:
                plt.ylabel('Metrics value')
            plt.title(g)
            if aix // ncols == nrows-1:
                plt.xlabel('Nodes fraction crawled' if x_normalize else 'Nodes crawled')
            aix += 1

            if x_lims:
                plt.xlim(x_lims)
            for k, c in enumerate(self.crawler_names):
                for i, m in enumerate(self.metric_names):
                    contents = self.contents[g][c][m]
                    ls, col = (k, i) if swap_coloring_scheme else (i, k)
                    # Draw variance
                    xs = contents['x']
                    if x_normalize and len(xs) > 0:
                        xs = xs / xs[-1]
                    if len(xs) > 0 and draw_error:
                        error = np.var(contents['ys'], axis=0) ** 0.5
                        plt.fill_between(xs, contents['avy'] - error, contents['avy'] + error, alpha=0.2,
                                         color=COLORS[col % len(COLORS)])
                    plt.plot(xs, contents['avy'], linewidth=1,
                             linestyle=LINESTYLES[ls % len(LINESTYLES)],
                             color=COLORS[col % len(COLORS)],
                             label="[%s] %s, %s" % (self.n_instances, self.labels[c], self.labels[m]))
                    pbar.update(1)
        pbar.close()
        plt.legend()
        plt.tight_layout()

    def _compute_auccs(self, x_lims=None):
        """
        :param x_lims: if specified as (x_from, x_to), compute AUCC for an interval containing the specified one
        """
        x_lims = x_lims or self.x_lims
        if len(self.auccs) > 0:
            return
        # Compute AUCCs
        G = len(self.graph_names)
        C = len(self.crawler_names)
        M = len(self.metric_names)
        self.auccs.clear()
        pbar = tqdm(total=G*C*M, desc='Computing AUCCs')
        for g in self.graph_names:
            self.auccs[g] = {}
            for c in self.crawler_names:
                self.auccs[g][c] = {}
                for m in self.metric_names:
                    self.auccs[g][c][m] = aucc = {}
                    contents = self.contents[g][c][m]
                    xs = contents['x']
                    ys = contents['ys']
                    if x_lims:
                        x0, x1 = x_lims
                        i0 = bisect_right(xs, x0)-1
                        i1 = bisect_left(xs, x1)
                    else:
                        i0 = 0
                        i1 = len(xs)

                    aucc['AUCC'] = [compute_aucc(xs[i0: i1], ys[inst][i0: i1]) for inst in range(len(ys))]
                    aucc['wAUCC'] = [compute_waucc(xs[i0: i1], ys[inst][i0: i1]) for inst in range(len(ys))]
                    pbar.update(1)
        pbar.close()

    def draw_aucc(self, aggregator='AUCC', x_lims=None, scale=3, xticks_rotation=90):
        """
        Draw G plots with M lines. Ox - C crawlers, Oy - AUCC value (M curves with error bars).
        M - num of metrics, G - num of graphs, C - num of crawlers

        :param aggregator: function translating crawling curve into 1 number. AUCC (default) or wAUCC
        :param x_lims: x-limits passed to aggregator. Overrides x_lims passed in constructor
        :param scale: size of plots (default 3)
        :param xticks_rotation: rotate x-ticks (default 90 degrees)
        """
        assert aggregator in ['AUCC', 'wAUCC']
        x_lims = x_lims or self.x_lims

        self._compute_auccs(x_lims=x_lims)
        G = len(self.graph_names)
        C = len(self.crawler_names)
        M = len(self.metric_names)

        # Draw
        nrows = int(sqrt(G))
        ncols = ceil(G / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, num="%s" % aggregator, figsize=(1 + scale * ncols, 1 + scale * nrows))
        aix = 0
        pbar = tqdm(total=G*M, desc='Plotting %s' % aggregator)
        xs = list(range(1, 1 + C))
        for g in self.graph_names:
            if nrows > 1 and ncols > 1:
                plt.sca(axs[aix // ncols, aix % ncols])
            elif nrows * ncols > 1:
                plt.sca(axs[aix])
            if aix == 0:
                plt.ylabel('%s value' % aggregator)
            plt.title(g)

            for i, m in enumerate(self.metric_names):
                errors = [np.var(self.auccs[g][c][m][aggregator]) for c in self.crawler_names]
                ys = [np.mean(self.auccs[g][c][m][aggregator]) for c in self.crawler_names]
                plt.errorbar(xs, ys, errors, label=self.labels[m], marker='.', capsize=5, color=COLORS[i % len(COLORS)])
                pbar.update(1)
            plt.xticks(xs, [self.labels[c] for c in self.crawler_names], rotation=xticks_rotation)
            aix += 1
        pbar.close()
        plt.legend()
        plt.tight_layout()

    def draw_winners(self, aggregator='AUCC', x_lims=None, scale=8, xticks_rotation=90):
        """
        Draw C stacked bars (each of M elements). Ox - C crawlers, Oy - number of wins (among G) by (w)AUCC value.
        Miss graphs where not all configurations are present.

        :param aggregator: function translating crawling curve into 1 number. AUCC (default) or wAUCC
        :param x_lims: x-limits passed to aggregator. Overrides x_lims passed in constructor
        :param scale: size of plots (default 8)
        :param xticks_rotation: rotate x-ticks (default 90 degrees)
        """
        assert aggregator in ['AUCC', 'wAUCC']
        x_lims = x_lims or self.x_lims

        self._compute_auccs(x_lims=x_lims)
        G = len(self.graph_names)
        C = len(self.crawler_names)
        M = len(self.metric_names)

        # Computing winners
        winners = {}  # winners[crawler][metric] -> count
        for c in self.crawler_names:
            winners[c] = {}
            for m in self.metric_names:
                winners[c][m] = 0

        for m in self.metric_names:
            for g in self.graph_names:
                ca = [np.mean(self.auccs[g][c][m][aggregator]) for c in self.crawler_names]
                if any(np.isnan(ca)):
                    continue
                winner = self.crawler_names[np.argmax(ca)]
                winners[winner][m] += 1

        # Draw
        plt.figure(num="Winners by %s" % aggregator, figsize=(1 + scale, scale))
        xs = list(range(1, 1 + C))
        prev_bottom = np.zeros(C)
        for i, m in enumerate(self.metric_names):
            h = [winners[c][m] for c in self.crawler_names]
            plt.bar(xs, h, width=0.8, bottom=prev_bottom, color=COLORS[i % len(COLORS)], label=self.labels[m])
            prev_bottom += h

        plt.ylabel('%s value' % aggregator)
        plt.xticks(xs, [self.labels[c] for c in self.crawler_names], rotation=xticks_rotation)
        plt.legend()
        plt.tight_layout()

    def show_plots(self):
        """ Show drawed matplotlib plots """
        plt.show()


def test_merger():
    from crawlers.cbasic import filename_to_definition

    graphs = [
        'douban',
        'ego-gplus',
    ]
    p = 0.01
    crawler_defs = [
        filename_to_definition('RW()'),
        filename_to_definition('RC()'),
        filename_to_definition('BFS()'),
        filename_to_definition('DFS()'),
        filename_to_definition('SBS(p=0.1)'),
        filename_to_definition('SBS(p=0.25)'),
        filename_to_definition('SBS(p=0.5)'),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]
    crm = ResultsMerger(graphs, crawler_defs, metric_defs, n_instances=6)
    import sys
    print("Missing instances:")
    json.dump(crm.missing_instances(), sys.stdout, indent=1)
    crm.draw_by_crawler()
    # crm.draw_aucc('AUCC')
    crm.draw_winners('AUCC')
    crm.show_plots()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    test_merger()
