import logging
from math import sqrt, ceil
from operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt

from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric

from base.cgraph import MyGraph
from crawlers.cbasic import CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler, \
    BreadthFirstSearchCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler, \
    definition_to_filename
from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
    ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs
from crawlers.multiseed import MultiInstanceCrawler
from running.animated_runner import AnimatedCrawlerRunner, Metric
from graph_io import GraphCollections, netrepo_names, konect_names
from statistics import Stat, get_top_centrality_nodes


def short_str(graph: MyGraph):
    name = graph.name
    n = graph[Stat.NODES]
    if n > 1e9:
        n = "%.1fB" % (n / 1e9)
    elif n > 1e8:
        n = "%.fM" % (n / 1e6)
    elif n > 1e6:
        n = "%.1fM" % (n / 1e6)
    elif n > 1e5:
        n = "%.fK" % (n / 1e3)
    elif n > 1e3:
        n = "%.1fK" % (n / 1e3)
    else:
        n = "%s" % n
    e = graph[Stat.EDGES]
    if e > 1e9:
        e = "%.1fB" % (e / 1e9)
    elif e > 1e8:
        e = "%.fM" % (e / 1e6)
    elif e > 1e6:
        e = "%.1fM" % (e / 1e6)
    elif e > 1e5:
        e = "%.fK" % (e / 1e3)
    elif e > 1e3:
        e = "%.1fK" % (e / 1e3)
    else:
        e = "%s" % e
    d = graph[Stat.MAX_DEGREE]
    d = "%.1fK" % (d / 1e3) if d > 1e3 else "%s" % d
    return "%s\n" % (name) + r"N=%s, E=%s, $d_{max}$=%s" % (n, e, d)


def three_stage_n_s():
    p = 0.1
    budget_coeff = [
        # 0.00001, 0.00003, 0.00005,
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = konect_names
    # graph_names = netrepo_names
    finals = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
           (ThreeStageCrawler, {'s': int(s*budget), 'n': budget, 'p': p}) for s in seed_coeff for budget in budgets
        ]

        # Collect final results
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p})
                finals[i][j][k] = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                # print(b, s, finals[i][j][k])

        # Draw finals
        if nrows > 1 and ncols > 1:
            plt.sca(axs[aix // ncols, aix % ncols])
        elif nrows * ncols > 1:
            plt.sca(axs[aix])
        if aix % ncols == 0:
            plt.ylabel('n / N')
        if i == 0:
            plt.title(g)
        if aix // ncols == nrows - 1:
            plt.xlabel('s / n')
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_mod_b_s():
    p = 0.01
    budget_coeff = 0.03
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    batch = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    # graph_names = konect_names[:12]
    graph_names = netrepo_names
    finals = np.zeros((len(graph_names), len(batch), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budget = int(budget_coeff * n)
        crawler_defs = [
           (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'b': b, 'p': p}) for s in seed_coeff for b in batch
        ]

        # Collect final results
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
        for j, b in enumerate(batch):
            budget = int(budget_coeff * n)
            for k, s in enumerate(seed_coeff):
                cd = (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'b': b, 'p': p})
                finals[i][j][k] = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                # print(b, s, finals[i][j][k])

        # Draw finals
        if nrows > 1 and ncols > 1:
            plt.sca(axs[aix // ncols, aix % ncols])
        elif nrows * ncols > 1:
            plt.sca(axs[aix])
        if aix % ncols == 0:
            plt.ylabel('b')
        if i == 0:
            plt.title(g)
        if aix // ncols == nrows - 1:
            plt.xlabel('s / n')
        plt.ylim((len(batch)-0.5, -0.5))
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(batch)), batch)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)

    # plt.colorbar()
    plt.tight_layout()
    plt.show()


def three_stage_best_n_s():
    p = 0.01
    budget_coeff = [
        # 0.00001, 0.00003, 0.00005,
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = konect_names + netrepo_names
    worst = np.ones((len(budget_coeff), len(seed_coeff)))
    res = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))
    f_20_max = np.zeros(len(graph_names))  # 20th degree / max degree
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
           (ThreeStageCrawler, {'s': int(s*budget), 'n': budget, 'p': p}) for s in seed_coeff for budget in budgets
        ]
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p})
                v = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                res[i][j][k] = v
                if v < worst[j][k]:
                    worst[j][k] = v
        node_cent = list(g[Stat.DEGREE_DISTR].items())
        f_20 = sorted(node_cent, key=itemgetter(1), reverse=True)[20][1]
        f_20_max[i] = g[Stat.NODES] / f_20
        print(graph_name, f_20_max[i])
    avg = np.mean(res, axis=0)
    var = np.var(res, axis=0) ** 0.5

    print(worst)
    print(avg)
    print(var)
    print(list(zip(graph_names, f_20_max)))
    # plt.title('3-Stage worst in konect')
    # plt.imshow(worst, cmap='inferno', vmin=0, vmax=1)
    plt.title('3-Stage average p=%s' % p)
    plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
    plt.xlabel('s / n')
    plt.ylabel('n / N')
    plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
    plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff)-0.5, -0.5))
    plt.show()


def three_stage_best_n_s_all():
    scale = 3.5
    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for p in [0.0001]:
        plt.sca(axs[aix])

        budget_coeff = [
            # 0.00001, 0.00003, 0.00005,
            0.0001, 0.0003, 0.0005,
            0.001, 0.003, 0.005,
            0.01, 0.03, 0.05, 0.1, 0.3
        ]
        seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        metric_defs = [
            (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
        ]

        n_instances = 8
        graph_names = (konect_names + netrepo_names)[:1]
        worst = np.ones((len(budget_coeff), len(seed_coeff)))
        avg = np.zeros((len(budget_coeff), len(seed_coeff)))
        for i, graph_name in enumerate(graph_names):
            g = GraphCollections.get(graph_name, not_load=True)
            n = g[Stat.NODES]
            budgets = [int(b*n) for b in budget_coeff]
            crawler_defs = [
               (ThreeStageCrawler, {'s': int(s*budget), 'n': budget, 'p': p}) for s in seed_coeff for budget in budgets
            ]
            rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

            for j, b in enumerate(budget_coeff):
                budget = int(b*n)
                for k, s in enumerate(seed_coeff):
                    start_seeds = int(s*budget)
                    cd = (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p})
                    res = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                    avg[j][k] += res / len(graph_names)
                    if res < worst[j][k]:
                        worst[j][k] = res

        plt.title(r'$p=%s$' % p, fontsize=14)
        plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
        plt.xlabel('s / n', fontsize=14)
        if aix == 0:
            plt.ylabel('n / N', fontsize=14)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.tight_layout()
        plt.ylim((len(budget_coeff)-0.5, -0.5))
        plt.grid(False)
        aix += 1

    plt.colorbar()
    # plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_mod_best_b_s():
    p = 0.01
    budget_coeff = 0.03
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    batch = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = konect_names + netrepo_names
    worst = np.ones((len(batch), len(seed_coeff)))
    avg = np.zeros((len(batch), len(seed_coeff)))
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budget = int(budget_coeff * n)
        crawler_defs = [
            (ThreeStageMODCrawler, {'s': int(s * budget), 'n': budget, 'b': b, 'p': p}) for s in seed_coeff for b in
            batch
        ]
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

        for j, b in enumerate(batch):
            for k, s in enumerate(seed_coeff):
                cd = (ThreeStageMODCrawler, {'s': int(s * budget), 'n': budget, 'b': b, 'p': p})
                res = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                avg[j][k] += res / len(graph_names)
                if res < worst[j][k]:
                    worst[j][k] = res

    # print(worst)
    print(avg)
    # plt.title('3-StageMOD worst in netrepo')
    # plt.imshow(worst, cmap='inferno', vmin=0, vmax=1)
    plt.title('3-StageMOD average p=%s' % p)
    plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
    plt.xlabel('s / n')
    plt.ylabel('b')
    plt.yticks(np.arange(0, len(batch)), batch)
    plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(batch)-0.5, -0.5))
    plt.show()


def three_stage_comparison():
    p = 0.01
    budget_coeff = 0.03
    seed_coeff = 0.2
    batch = 100

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = konect_names[:12]
    # graph_names = netrepo_names[:20]

    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name)
        n = g[Stat.NODES]
        budget = int(budget_coeff * n)
        crawler_defs = [
           (ThreeStageCrawler, {'s': int(0.7 * budget), 'n': budget, 'p': p}),
           (ThreeStageMODCrawler, {'s': int(0.2 * budget), 'n': budget, 'b': batch, 'p': p}),
        ]

        # # Run
        # chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
        # chr.run_missing(n_instances, max_cpus=8, max_memory=26)
        # print('\n\n')

        # Collect final results
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
        rm.draw_by_metric_crawler(x_lims=(0, budget), x_normalize=False, scale=12)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    # three_stage_n_s()
    # three_stage_mod_b_s()
    three_stage_best_n_s()
    # three_stage_best_n_s_all()
    # three_stage_mod_best_b_s()
    # three_stage_comparison()
