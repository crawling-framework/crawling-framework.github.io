import glob
import json
import logging
import os
import shutil
from math import sqrt, ceil

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from running.metrics_and_runner import TopCentralityMetric, Metric
from graph_io import GraphCollections, konect_names, netrepo_names
from statistics import Stat

from crawlers.cbasic import Crawler, CrawlerException, MaximumObservedDegreeCrawler, \
    RandomWalkCrawler, RandomCrawler, SnowBallCrawler, BreadthFirstSearchCrawler, \
    DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, \
    MaximumExcessDegreeCrawler, definition_to_filename, filename_to_definition
from crawlers.advanced import ThreeStageCrawler, AvrachenkovCrawler, ThreeStageMODCrawler
from crawlers.cadvanced import DE_Crawler
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
# FIXME what if Multi with count > g.nodes()?
example_graph = GraphCollections.get('example', 'other')


class ResultsMerger:
    def __init__(self, graph_names, crawler_defs, metric_defs, n_instances=1):
        """
        :param graph_names: list of graphs names
        :param crawler_defs: list of crawlers definitions
        :param metric_defs: list of metrics definitions
        :param n_instances: number of instances to average over
        """
        self.graph_names = graph_names
        self.crawler_names = []  # list(map(definition_to_filename, crawler_defs))
        self.metric_names = []  # list(map(definition_to_filename, metric_defs))
        self.labels = {}  # pretty short names to draw in plots

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
        self.instances = {}  # instances[graph][crawler][metric] -> count
        self.contents = {}  # contents[graph][crawler][metric]: 'x' -> [], 'ys' -> [[]*n_instances], 'avy' -> []
        self.auccs = {}  # auccs[graph][crawler][metric]: 'AUCC' -> [AUCC], 'wAUCC' -> [wAUCC]
        self.read()
        # missing = self.missing_instances()
        # if len(missing) > 0:
        #     logging.warning("Missing instances, will not be plotted:\n%s" % json.dumps(missing, indent=2))

    @staticmethod
    def names_to_path(graph_name: str, crawler_name: str, metric_name: str):
        """ Returns file pattern e.g.
        '/home/misha/workspace/crawling/results/ego-gplus/POD(batch=1)/TopK(centrality=BtwDistr,measure=Re,part=crawled,top=0.01)/*.json'
        """
        path = os.path.join(RESULT_DIR, graph_name, crawler_name, metric_name, "*.json")
        return path

    def read(self):
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
                    # # For rename
                    # p = CrawlerRunsMerger.names_to_path(g, c, m, old=True)
                    # paths = glob.glob(p)
                    # self.contents[g][c][m] = len(paths)
                    # for i, p in enumerate(paths):
                    #     new_p = CrawlerRunsMerger.names_to_path(g, c, m, old=False)
                    #     new_p = new_p.replace('*', str(i))
                    #     print(p, new_p)
                    #     if not os.path.exists(os.path.dirname(new_p)):
                    #         os.makedirs(os.path.dirname(new_p))
                    #     os.rename(p, new_p)

                    paths = glob.glob(ResultsMerger.names_to_path(g, c, m))
                    self.instances[g][c][m] = len(paths)
                    self.contents[g][c][m] = contents = {}

                    count = len(paths)
                    contents['x'] = []
                    contents['ys'] = ys = [[]]*count
                    contents['avy'] = []

                    for inst, p in enumerate(paths):
                        with open(p, 'r') as f:
                            imported = json.load(f)
                        if len(contents['x']) == 0:
                            xs = np.array(sorted([int(x) for x in list(imported.keys())]))
                            # xs = xs / xs[-1]
                            contents['x'] = xs
                        if inst == 0:
                            contents['avy'] = np.zeros(len(xs))
                        ys[inst] = np.array([float(x) for x in list(imported.values())])
                        contents['avy'] += np.array(ys[inst]) / count

                    pbar.update(1)
        pbar.close()
        # print(self.contents)
        # print(json.dumps(self.contents, indent=2))

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
        """ Return dict of instances where computed < n_instances: absent[graph][crawler][metric] -> missing count
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
        """ Draw M x G table of plots with C lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers.
        Ox - crawling step, Oy - metric value.
        """
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        G = len(self.graph_names)
        M = len(self.metric_names)
        nrows, ncols = M, G
        if M == 1:
            nrows = int(sqrt(G))
            ncols = ceil(G / nrows)
        if G == 1:
            nrows = int(sqrt(M))
            ncols = ceil(M / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=x_normalize, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
        # fig.text(0.5, 0.02, 'Число итераций краулинга', ha='center')
        # fig.text(0.02, 0.5, 'Доля собранных влиятельных вершин', va='center', rotation='vertical')

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
                        plt.fill_between(xs, contents['avy'] - error, contents['avy'] + error, color=colors[k % len(colors)], alpha=0.2)
                    plt.plot(xs, contents['avy'], color=colors[k % len(colors)], linewidth=1, label=self.labels[c])
                    pbar.update(1)
        pbar.close()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def draw_by_metric_crawler(self, x_lims=None, x_normalize=True, swap_coloring_scheme=False, draw_error=True, scale=3):
        """ Draw G plots with CxM lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers.
        Ox - crawling step, Oy - metric value.
        """
        linestyles = ['-', '--', ':', '-.']
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        G = len(self.graph_names)
        M = len(self.metric_names)
        nrows = int(sqrt(G))
        ncols = ceil(G / nrows)
        fig, axs = plt.subplots(nrows, ncols, sharex=x_normalize, sharey=True, figsize=(1 + scale * ncols, scale * nrows))

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
                                         color=colors[col % len(colors)])
                    plt.plot(xs, contents['avy'], linewidth=1,
                             linestyle=linestyles[ls % len(linestyles)],
                             color=colors[col % len(colors)],
                             label="[%s] %s, %s" % (self.n_instances, self.labels[c], self.labels[m]))
                    pbar.update(1)
        pbar.close()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def draw_by_metric(self):
        """ Draw C x G table of plots with M lines each, where
        M - num of metrics, G - num of graphs, C - num of crawlers
        Ox - crawling step, Oy - metric value.
        """
        pass

    def _compute_auccs(self):
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
                    aucc['AUCC'] = [compute_aucc(contents['x'], contents['ys'][inst]) for inst in range(len(contents['ys']))]
                    aucc['wAUCC'] = [compute_waucc(contents['x'], contents['ys'][inst]) for inst in range(len(contents['ys']))]
                    pbar.update(1)
        pbar.close()

    def draw_aucc(self, aggregator='AUCC'):
        """ Draw G plots with M lines. Ox - C crawlers, Oy - AUCC value (M curves with error bars).
        M - num of metrics, G - num of graphs, C - num of crawlers
        """
        assert aggregator in ['AUCC', 'wAUCC']
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        self._compute_auccs()
        G = len(self.graph_names)
        C = len(self.crawler_names)
        M = len(self.metric_names)

        # Draw
        nrows = int(sqrt(G))
        ncols = ceil(G / nrows)
        scale = 3
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
        aix = 0
        pbar = tqdm(total=G*M, desc='Plotting AUCC')
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
                plt.errorbar(xs, ys, errors, label=self.labels[m], marker='.', capsize=5, color=colors[i % len(colors)])
                pbar.update(1)
            plt.xticks(xs, [self.labels[c] for c in self.crawler_names], rotation=90)
            aix += 1
        pbar.close()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def draw_winners(self, aggregator='AUCC'):
        assert aggregator in ['AUCC', 'wAUCC']
        colors = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        self._compute_auccs()
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
                # print(ca, np.argmax(ca))
                winner = self.crawler_names[np.argmax(ca)]
                winners[winner][m] += 1

        # Draw
        scale = 8
        plt.figure(figsize=(1 + scale, scale))
        xs = list(range(1, 1 + C))
        prev_bottom = np.zeros(C)
        for i, m in enumerate(self.metric_names):
            h = [winners[c][m] for c in self.crawler_names]
            plt.bar(xs, h, width=0.8, bottom=prev_bottom, color=colors[i % len(colors)], label=self.labels[m])
            prev_bottom += h

        plt.ylabel('%s value' % aggregator)
        plt.xticks(xs, [self.labels[c] for c in self.crawler_names], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compute_results_as_table(self):
        with open(os.path.join(RESULT_DIR, 'ResultsAsTable'), 'a+') as f:
            for _, m in enumerate(self.metric_names):
                f.write('%s\n' % m)
                for _, g in enumerate(self.graph_names):
                    f.write('%s ' % g)
                    max_result = -1

                    for _, c in enumerate(self.crawler_names):
                        contents = self.contents[g][c][m]
                        final_results = [contents['ys'][inst][-1] for inst in range(len(contents['ys']))]
                        avg_result = np.average(final_results)
                        if avg_result > max_result:
                            max_result = avg_result
                        f.write(' & %.4f±%.4f' % (avg_result, np.std(final_results)))
                    f.write(' & %.4f \\\ \hline\r\n' % max_result)
            f.close()

def test_merger():
    g = GraphCollections.get('socfb-Bingham82', not_load=True)
    # print(g[Stat.NODES], g[Stat.ASSORTATIVITY])

    graphs = [
        # 'socfb-Bingham82',
        # 'ego-gplus',
        # 'web-sk-2005',
        # 'digg-friends',
        'douban',
        'ego-gplus',
        # g for g in netrepo_names
    # ] + [
    #     g for g in konect_names
    ]
    p = 0.01
    crawler_defs = [
        # (ThreeStageCrawler, {'s': 699, 'n': 1398, 'p': p}),
        # (ThreeStageMODCrawler, {'s': 699, 'n': 1398, 'p': p, 'b': 10}),
        filename_to_definition('RW()'),
        filename_to_definition('RC()'),
        filename_to_definition('BFS()'),
        filename_to_definition('DFS()'),
        filename_to_definition('SBS(p=0.1)'),
        filename_to_definition('SBS(p=0.25)'),
        filename_to_definition('SBS(p=0.5)'),
        # filename_to_definition('SBS(p=0.75)'),
        # filename_to_definition('SBS(p=0.89)'),
        # filename_to_definition('MOD(batch=1)'),
        # filename_to_definition('MOD(batch=10)'),
        # filename_to_definition('MOD(batch=100)'),
        # filename_to_definition('MOD(batch=1000)'),
        # filename_to_definition('MOD(batch=10000)'),
        # filename_to_definition('POD(batch=1)'),
        # filename_to_definition('POD(batch=10)'),
        # filename_to_definition('POD(batch=100)'),
        # filename_to_definition('POD(batch=1000)'),
        # filename_to_definition('POD(batch=10000)'),
        # filename_to_definition('DE(initial_budget=10)'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=2,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=3,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=4,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=5,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=10,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=30,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=100,crawler_def=DE(initial_budget=10))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=BFS())'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=MOD(batch=1))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=POD(batch=1))'),
        # filename_to_definition('MultiInstance(count=1000,crawler_def=DE(initial_budget=10))'),
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
    # crm.missing_instances()
    crm.draw_by_crawler()
    # crm.draw_aucc('AUCC')
    crm.draw_winners('AUCC')


def rename_results_old_to_new():
    crawler_replace_dict = {
        'Multi_2xBFS': 'MultiInstance(count=2,crawler_def=BFS())',
        'Multi_2xMOD': 'MultiInstance(count=2,crawler_def=MOD(batch=1))',
        'Multi_2xPOD': 'MultiInstance(count=2,crawler_def=POD(batch=1))',
        'Multi_2xDE': 'MultiInstance(count=2,crawler_def=DE(initial_budget=10))',
        'Multi_3xBFS': 'MultiInstance(count=3,crawler_def=BFS())',
        'Multi_3xMOD': 'MultiInstance(count=3,crawler_def=MOD(batch=1))',
        'Multi_3xPOD': 'MultiInstance(count=3,crawler_def=POD(batch=1))',
        'Multi_3xDE': 'MultiInstance(count=3,crawler_def=DE(initial_budget=10))',
        'Multi_4xBFS': 'MultiInstance(count=4,crawler_def=BFS())',
        'Multi_4xMOD': 'MultiInstance(count=4,crawler_def=MOD(batch=1))',
        'Multi_4xPOD': 'MultiInstance(count=4,crawler_def=POD(batch=1))',
        'Multi_4xDE': 'MultiInstance(count=4,crawler_def=DE(initial_budget=10))',
        'Multi_5xBFS': 'MultiInstance(count=5,crawler_def=BFS())',
        'Multi_5xMOD': 'MultiInstance(count=5,crawler_def=MOD(batch=1))',
        'Multi_5xPOD': 'MultiInstance(count=5,crawler_def=POD(batch=1))',
        'Multi_5xDE': 'MultiInstance(count=5,crawler_def=DE(initial_budget=10))',
        'Multi_10xBFS': 'MultiInstance(count=10,crawler_def=BFS())',
        'Multi_10xMOD': 'MultiInstance(count=10,crawler_def=MOD(batch=1))',
        'Multi_10xPOD': 'MultiInstance(count=10,crawler_def=POD(batch=1))',
        'Multi_10xDE': 'MultiInstance(count=10,crawler_def=DE(initial_budget=10))',
        'Multi_30xBFS': 'MultiInstance(count=30,crawler_def=BFS())',
        'Multi_30xMOD': 'MultiInstance(count=30,crawler_def=MOD(batch=1))',
        'Multi_30xPOD': 'MultiInstance(count=30,crawler_def=POD(batch=1))',
        'Multi_30xDE': 'MultiInstance(count=30,crawler_def=DE(initial_budget=10))',
        'Multi_100xBFS': 'MultiInstance(count=100,crawler_def=BFS())',
        'Multi_100xMOD': 'MultiInstance(count=100,crawler_def=MOD(batch=1))',
        'Multi_100xPOD': 'MultiInstance(count=100,crawler_def=POD(batch=1))',
        'Multi_100xDE': 'MultiInstance(count=100,crawler_def=DE(initial_budget=10))',
        'Multi_1000xBFS': 'MultiInstance(count=1000,crawler_def=BFS())',
        'Multi_1000xMOD': 'MultiInstance(count=1000,crawler_def=MOD(batch=1))',
        'Multi_1000xPOD': 'MultiInstance(count=1000,crawler_def=POD(batch=1))',
        'Multi_1000xDE': 'MultiInstance(count=1000,crawler_def=DE(initial_budget=10))',
        'RW_': 'RW()',
        'RC_': 'RC()',
        'BFS': 'BFS()',
        'DFS': 'DFS()',
        'SBS10': 'SBS(p=0.1)',
        'SBS25': 'SBS(p=0.25)',
        'SBS75': 'SBS(p=0.75)',
        'SBS89': 'SBS(p=0.89)',
        'SBS': 'SBS(p=0.5)',
        'MOD10000': 'MOD(batch=10000)',
        'MOD1000': 'MOD(batch=1000)',
        'MOD100': 'MOD(batch=100)',
        'MOD10': 'MOD(batch=10)',
        'MOD': 'MOD(batch=1)',
        'POD10000': 'POD(batch=10000)',
        'POD1000': 'POD(batch=1000)',
        'POD100': 'POD(batch=100)',
        'POD10': 'POD(batch=10)',
        'POD': 'POD(batch=1)',
        'DE': 'DE(initial_budget=10)',
    }
    metric_replace_dict = {
        'Re_crawled_BtwDistr_0.01': 'TopK(centrality=BtwDistr,measure=Re,part=crawled,top=0.01)',
        'Re_crawled_ClsnDistr_0.01': 'TopK(centrality=ClsnDistr,measure=Re,part=crawled,top=0.01)',
        'Re_crawled_DegDistr_0.01': 'TopK(centrality=DegDistr,measure=Re,part=crawled,top=0.01)',
        'Re_crawled_EccDistr_0.01': 'TopK(centrality=EccDistr,measure=Re,part=crawled,top=0.01)',
        'Re_crawled_KCorDistr_0.01': 'TopK(centrality=KCorDistr,measure=Re,part=crawled,top=0.01)',
        'Re_crawled_PgrDistr_0.01': 'TopK(centrality=PgrDistr,measure=Re,part=crawled,top=0.01)',
    }
    for graph_name in (konect_names + netrepo_names)[2:]:
        path_pattern = os.path.join(RESULT_DIR, graph_name, '*', '*', "*.json")
        paths = glob.glob(path_pattern)
        for p in paths:
            parts = p.split('/')
            parts[5] = '_results'
            # parts[7] = crawler_replace_dict[parts[7]]
            # parts[8] = metric_replace_dict[parts[8]]
            new_path = "/".join(parts)

            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            os.rename(p, new_path)
            print(p, new_path)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    test_merger()
    # rename_results_old_to_new()

