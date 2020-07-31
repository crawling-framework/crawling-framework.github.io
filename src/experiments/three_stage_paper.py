import glob
import logging
import os
import sys
from math import sqrt, ceil
from operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt

from crawlers.cadvanced import DE_Crawler
from running.history_runner import CrawlerHistoryRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric

from base.cgraph import MyGraph
from crawlers.cbasic import CrawlerException, MaximumObservedDegreeCrawler, RandomWalkCrawler, \
    BreadthFirstSearchCrawler, DepthFirstSearchCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler, \
    definition_to_filename
from crawlers.advanced import ThreeStageCrawler, CrawlerWithAnswer, AvrachenkovCrawler, \
    ThreeStageMODCrawler, ThreeStageCrawlerSeedsAreHubs, EmulatorWithAnswerCrawler
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


social_names = [
    'socfb-Bingham82',          # N=10001,   E=362892,   d_avg=72.57
    'soc-brightkite',           # N=56739,   E=212945,   d_avg=7.51
    'socfb-Penn94',             # N=41536,   E=1362220,  d_avg=65.59
    'socfb-wosn-friends',       # N=63392,   E=816886,   d_avg=25.77
    'soc-slashdot',             # N=70068,   E=358647,   d_avg=10.24
    'soc-themarker',            # N=69317,   E=1644794,  d_avg=47.46
    'soc-BlogCatalog',          # N=88784,   E=2093195,  d_avg=47.15
    'soc-anybeat',              # N=12645,   E=49132,    d_avg=7.77
    'soc-twitter-follows',      # N=404719,  E=713319,   d_avg=3.53
    'petster-hamster',          # N=2000,    E=16098,    d_avg=16.10
    'ego-gplus',                # N=23613,   E=39182,    d_avg=3.32
    'slashdot-threads',         # N=51083,   E=116573,   d_avg=4.56
    'douban',                   # N=154908,  E=327162,   d_avg=4.22
    'digg-friends',             # N=261489,  E=1536577,  d_avg=11.75
    'loc-brightkite_edges',     # N=58228,   E=214078,   d_avg=7.35
    'epinions',                 # N=119130,  E=704267,   d_avg=11.82
    'livemocha',                # N=104103,  E=2193083,  d_avg=42.13
    'petster-friendships-cat',  # N=148826,  E=5447464,  d_avg=73.21
    'petster-friendships-dog',  # N=426485,  E=8543321,  d_avg=40.06
    'munmun_twitter_social',    # N=465017,  E=833540,   d_avg=3.58
    'com-youtube',              # N=1134890, E=2987624,  d_avg=5.27
    'flixster',                 # N=2523386, E=7918801,  d_avg=6.28
    'youtube-u-growth',         # N=3216075, E=9369874,  d_avg=5.83
    'soc-pokec-relationships',  # N=1632803, E=22301964, d_avg=27.32
]


def two_stage_n_s():
    p = 0.01
    budget_coeff = [
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    # graph_names = konect_names
    graph_names = social_names
    finals = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(0.7 * len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows), num='2-Stage_p=%s' % p)
    aix = 0
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
           (AvrachenkovCrawler, {'n1': int(s*budget), 'n': budget, 'k': int(p*n)}) for s in seed_coeff for budget in budgets
        ]

        # Collect final results
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (AvrachenkovCrawler, {'n1': start_seeds, 'n': budget, 'k': int(p*n)})
                finals[i][j][k] = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                # print(b, s, finals[i][j][k])

        # Draw finals
        if nrows > 1 and ncols > 1:
            plt.sca(axs[aix // ncols, aix % ncols])
        elif nrows * ncols > 1:
            plt.sca(axs[aix])
        if aix % ncols == 0:
            plt.ylabel('n / |V|')
        if i == 0:
            plt.title(g)
        if aix // ncols == nrows - 1:
            plt.xlabel('s / n')
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.grid(False)

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def two_stage_avg_n_s():
    p = 0.01
    budget_coeff = [
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = social_names
    worst = np.ones((len(budget_coeff), len(seed_coeff)))
    res = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
           (AvrachenkovCrawler, {'n1': int(s*budget), 'n': budget, 'k': int(p*n)}) for s in seed_coeff for budget in budgets
        ]
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (AvrachenkovCrawler, {'n1': start_seeds, 'n': budget, 'k': int(p*n)})
                v = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                res[i][j][k] = v
                if v < worst[j][k]:
                    worst[j][k] = v
    avg = np.mean(res, axis=0)
    var = np.var(res, axis=0) ** 0.5

    print(worst)
    print(avg)
    print(var)
    plt.title('2-Stage average p=%s' % p)
    plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
    plt.xlabel('s / n')
    plt.ylabel('n / |V|')
    plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
    plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
    plt.grid(False)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff)-0.5, -0.5))
    plt.show()


def three_stage_n_s():
    p = 0.01
    budget_coeff = [
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
        # (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    # graph_names = konect_names
    graph_names = social_names
    finals = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows), num='3-Stage_p=%s' % p)
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
            plt.ylabel('n / |V|')
        if i == 0:
            plt.title(g)
        if aix // ncols == nrows - 1:
            plt.xlabel('s / n')
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.grid(False)

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_mod_n_s():
    p = 0.01
    budget_coeff = [
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    graph_names = social_names
    finals = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows), num='3-Stage_p=%s' % p)
    aix = 0
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
            (ThreeStageMODCrawler, {'s': int(s * budget), 'n': budget, 'b': 100, 'p': p}) for s in seed_coeff for budget in budgets
        ]

        # Collect final results
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (ThreeStageMODCrawler, {'s': start_seeds, 'n': budget, 'b': 100, 'p': p})
                finals[i][j][k] = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                # print(b, s, finals[i][j][k])

        # Draw finals
        if nrows > 1 and ncols > 1:
            plt.sca(axs[aix // ncols, aix % ncols])
        elif nrows * ncols > 1:
            plt.sca(axs[aix])
        if aix % ncols == 0:
            plt.ylabel('n / |V|')
        if i == 0:
            plt.title(g)
        if aix // ncols == nrows - 1:
            plt.xlabel('s / n')
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.grid(False)

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_mod_b_s():
    p = 0.01
    budget_coeff = 0.05
    # budget_coeff = 0.005
    seed_coeff = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    batch = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    # graph_names = konect_names + netrepo_names
    graph_names = social_names
    finals = np.zeros((len(graph_names), len(batch), len(seed_coeff)))  # finals[graph][n][s] -> F1

    nrows = int(sqrt(len(graph_names)))
    ncols = ceil(len(graph_names) / nrows)
    scale = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(1 + scale * ncols, scale * nrows), num='3-StageMOD_p=%s_budget=%s' % (p, budget_coeff))
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
        aix += 1
        plt.title(short_str(g))
        plt.imshow(finals[i], cmap='inferno', vmin=0, vmax=1)
        plt.yticks(np.arange(0, len(batch)), batch)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.grid(False)

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(batch) - 0.5, -0.5))
    plt.show()


def three_stage_avg_n_s():
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
    graph_names = social_names
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
        # node_cent = list(g[Stat.DEGREE_DISTR].items())
        # f_20 = sorted(node_cent, key=itemgetter(1), reverse=True)[20][1]
        # f_20_max[i] = g[Stat.NODES] / f_20
        # print(graph_name, f_20_max[i])
    avg = np.mean(res, axis=0)
    var = np.var(res, axis=0) ** 0.5
    from sys import stdout as sout
    print(" & ".join([str(s) for k, s in enumerate(seed_coeff)]))
    for j, b in enumerate(budget_coeff):
        print(" & ".join([str(b)] + ["$ %.2f $" % (avg[j][k], ) for k, s in enumerate(seed_coeff)]) + "\\\\ \\hline")

    # print(worst)
    print(avg)
    print(var)
    print(list(zip(graph_names, f_20_max)))
    # plt.title('3-Stage worst in konect')
    # plt.imshow(worst, cmap='inferno', vmin=0, vmax=1)
    plt.title('3-Stage average p=%s' % p)
    plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
    plt.xlabel('s / n')
    plt.ylabel('n / |V|')
    plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
    plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
    plt.grid(False)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff)-0.5, -0.5))
    plt.show()


def three_stage_mod_avg_n_s():
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
    graph_names = social_names
    worst = np.ones((len(budget_coeff), len(seed_coeff)))
    res = np.zeros((len(graph_names), len(budget_coeff), len(seed_coeff)))
    # f_20_max = np.zeros(len(graph_names))  # 20th degree / max degree
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        budgets = [int(b*n) for b in budget_coeff]
        crawler_defs = [
            (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'b': 100, 'p': p}) for s in seed_coeff for budget in budgets
        ]
        rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

        for j, b in enumerate(budget_coeff):
            budget = int(b*n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s*budget)
                cd = (ThreeStageMODCrawler, {'s': start_seeds, 'n': budget, 'b': 100, 'p': p})
                v = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                res[i][j][k] = v
                if v < worst[j][k]:
                    worst[j][k] = v
        # node_cent = list(g[Stat.DEGREE_DISTR].items())
        # f_20 = sorted(node_cent, key=itemgetter(1), reverse=True)[20][1]
        # f_20_max[i] = g[Stat.NODES] / f_20
        # print(graph_name, f_20_max[i])
    avg = np.mean(res, axis=0)
    var = np.var(res, axis=0) ** 0.5

    print(worst)
    print(avg)
    print(var)
    # print(list(zip(graph_names, f_20_max)))
    plt.title('3-StageMOD average p=%s' % p)
    plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
    plt.xlabel('s / n')
    plt.ylabel('n / |V|')
    plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
    plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
    plt.grid(False)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(budget_coeff)-0.5, -0.5))
    plt.show()


def three_stage_mod_avg_b_s():
    p = 0.01
    budget_coeff = 0.05
    # budget_coeff = 0.005
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    batch = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000]

    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
    ]

    n_instances = 8
    # graph_names = konect_names + netrepo_names
    graph_names = social_names
    scale = 3.5
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for budget_coeff in [0.05, 0.005]:
        plt.sca(axs[aix])
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
        plt.title('n/|V|=%s' % (budget_coeff), fontsize=15)
        # plt.title('3-StageMOD average p=%s, n/|V|=%s' % (p, budget_coeff), fontsize=14)
        plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
        plt.xlabel('s / n', fontsize=15)
        if aix == 0:
            plt.ylabel('b', fontsize=15)
        # plt.ylabel('b', fontsize=14)
        plt.yticks(np.arange(0, len(batch)), batch, fontsize=14)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90, fontsize=14)
        plt.grid(False)
        aix += 1

    # plt.colorbar()
    plt.tight_layout()
    plt.ylim((len(batch)-0.5, -0.5))
    plt.show()


def three_stage_avg_n_s_all():
    scale = 3.5
    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for p in [0.0001, 0.001, 0.01, 0.1]:
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
        graph_names = social_names
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
            plt.ylabel('n / |V|', fontsize=14)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.tight_layout()
        plt.ylim((len(budget_coeff)-0.5, -0.5))
        plt.grid(False)
        aix += 1

    # plt.colorbar()
    # plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_mod_avg_n_s_all():
    scale = 3.5
    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    for p in [0.0001, 0.001, 0.01, 0.1]:
        plt.sca(axs[aix])

        budget_coeff = [
            0.0001, 0.0003, 0.0005,
            0.001, 0.003, 0.005,
            0.01, 0.03, 0.05, 0.1, 0.3
        ]
        seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        batch = 100

        metric_defs = [
            (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
        ]

        n_instances = 8
        graph_names = social_names
        worst = np.ones((len(budget_coeff), len(seed_coeff)))
        avg = np.zeros((len(budget_coeff), len(seed_coeff)))
        for i, graph_name in enumerate(graph_names):
            g = GraphCollections.get(graph_name, not_load=True)
            n = g[Stat.NODES]
            budgets = [int(b*n) for b in budget_coeff]
            crawler_defs = [
               (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'p': p, 'b': batch}) for s in seed_coeff for budget in budgets
            ]
            rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

            for j, b in enumerate(budget_coeff):
                budget = int(b*n)
                for k, s in enumerate(seed_coeff):
                    start_seeds = int(s*budget)
                    cd = (ThreeStageMODCrawler, {'s': start_seeds, 'n': budget, 'p': p, 'b': batch})
                    res = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                    avg[j][k] += res / len(graph_names)
                    if res < worst[j][k]:
                        worst[j][k] = res

        plt.title(r'$p=%s$' % p, fontsize=14)
        plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
        plt.xlabel('s / n', fontsize=14)
        if aix == 0:
            plt.ylabel('n / |V|', fontsize=14)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90)
        plt.tight_layout()
        plt.ylim((len(budget_coeff)-0.5, -0.5))
        plt.grid(False)
        aix += 1

    # plt.colorbar()
    # plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_both_avg_n_s():
    scale = 3.5
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    batch = 100
    p = 0.01

    def three_cd(s, budget, p):
        return (ThreeStageCrawler, {'s': int(s*budget), 'n': budget, 'p': p})

    def threemod_cd(s, budget, p):
        return (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'p': p, 'b': batch})

    for three in [three_cd, threemod_cd]:
        plt.sca(axs[aix % ncols])

        budget_coeff = [
            0.0001, 0.0003, 0.0005,
            0.001, 0.003, 0.005,
            0.01, 0.03, 0.05, 0.1, 0.3
        ]
        seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        metric_defs = [
            (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
        ]

        n_instances = 8
        graph_names = social_names
        worst = np.ones((len(budget_coeff), len(seed_coeff)))
        avg = np.zeros((len(budget_coeff), len(seed_coeff)))
        for i, graph_name in enumerate(graph_names):
            g = GraphCollections.get(graph_name, not_load=True)
            n = g[Stat.NODES]
            budgets = [int(b*n) for b in budget_coeff]
            crawler_defs = [
                three(s, budget, p) for s in seed_coeff for budget in budgets
            ]
            rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

            for j, b in enumerate(budget_coeff):
                budget = int(b*n)
                for k, s in enumerate(seed_coeff):
                    # start_seeds = int(s*budget)
                    cd = three(s, budget, p)
                    res = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                    avg[j][k] += res / len(graph_names)
                    if res < worst[j][k]:
                        worst[j][k] = res

        plt.title('3-Step' if three == three_cd else '3-StepBatch (b=100)', fontsize=15)
        plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
        if aix // ncols == nrows-1:
            plt.xlabel('s / n', fontsize=15)
        if aix % ncols == 0:
            plt.ylabel('n / |V|', fontsize=15)
        plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff, fontsize=14)
        plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90, fontsize=14)
        plt.tight_layout()
        plt.ylim((len(budget_coeff)-0.5, -0.5))
        plt.grid(False)
        aix += 1

    # plt.colorbar()
    # plt.ylim((len(budget_coeff) - 0.5, -0.5))
    plt.show()


def three_stage_both_avg_n_s_all():
    scale = 3.5
    nrows = 2
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=(1 + scale * ncols, scale * nrows))
    aix = 0
    batch = 100

    def three_cd(s, budget, p):
        return (ThreeStageCrawler, {'s': int(s*budget), 'n': budget, 'p': p})

    def threemod_cd(s, budget, p):
        return (ThreeStageMODCrawler, {'s': int(s*budget), 'n': budget, 'p': p, 'b': batch})

    for three in [three_cd, threemod_cd]:
        for p in [0.0001, 0.001, 0.01, 0.1]:
            plt.sca(axs[aix // ncols, aix % ncols])

            budget_coeff = [
                0.0001, 0.0003, 0.0005,
                0.001, 0.003, 0.005,
                0.01, 0.03, 0.05, 0.1, 0.3
            ]
            seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            metric_defs = [
                (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short}),
            ]

            n_instances = 8
            graph_names = social_names
            worst = np.ones((len(budget_coeff), len(seed_coeff)))
            avg = np.zeros((len(budget_coeff), len(seed_coeff)))
            for i, graph_name in enumerate(graph_names):
                g = GraphCollections.get(graph_name, not_load=True)
                n = g[Stat.NODES]
                budgets = [int(b*n) for b in budget_coeff]
                crawler_defs = [
                    three(s, budget, p) for s in seed_coeff for budget in budgets
                ]
                rm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)

                for j, b in enumerate(budget_coeff):
                    budget = int(b*n)
                    for k, s in enumerate(seed_coeff):
                        # start_seeds = int(s*budget)
                        cd = three(s, budget, p)
                        res = rm.contents[graph_name][definition_to_filename(cd)][definition_to_filename(metric_defs[0])]['avy'][-1]
                        avg[j][k] += res / len(graph_names)
                        if res < worst[j][k]:
                            worst[j][k] = res

            if aix // ncols == 0:
                plt.title(r'$p=%s$' % p, fontsize=15)
            plt.imshow(avg, cmap='inferno', vmin=0, vmax=1)
            if aix // ncols == nrows-1:
                plt.xlabel('s / n', fontsize=15)
            if aix % ncols == 0:
                plt.ylabel('n / |V|', fontsize=15)
            plt.yticks(np.arange(0, len(budget_coeff)), budget_coeff, fontsize=14)
            plt.xticks(np.arange(0, len(seed_coeff)), seed_coeff, rotation=90, fontsize=14)
            plt.tight_layout()
            plt.ylim((len(budget_coeff)-0.5, -0.5))
            plt.grid(False)
            aix += 1

    # plt.colorbar()
    # plt.ylim((len(budget_coeff) - 0.5, -0.5))
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


def data_copy():
    budget_coeff = [
        0.0001, 0.0003, 0.0005,
        0.001, 0.003, 0.005,
        0.01, 0.03, 0.05, 0.1, 0.3
    ]
    seed_coeff = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    p = 0.01
    md = (TopCentralityMetric, {'top': p, 'measure': 'F1', 'part': 'answer', 'centrality': Stat.DEGREE_DISTR.short})
    m = definition_to_filename(md)

    graph_names = social_names
    for i, graph_name in enumerate(graph_names):
        g = GraphCollections.get(graph_name, not_load=True)
        n = g[Stat.NODES]
        for j, b in enumerate(budget_coeff):
            budget = int(b * n)
            for k, s in enumerate(seed_coeff):
                start_seeds = int(s * budget)
                cd = (AvrachenkovCrawler, {'n1': start_seeds, 'n': budget, 'k': int(p*n)})
                # cd = (ThreeStageCrawler, {'s': start_seeds, 'n': budget, 'p': p})
                # cd = (ThreeStageMODCrawler, {'s': start_seeds, 'n': budget, 'p': p, 'b': 100})
                c = definition_to_filename(cd)

                path = ResultsMerger.names_to_path(graph_name, c, m)
                paths = glob.glob(path)
                for src in paths:
                    dst = src.replace('results', '2-Stage')
                    if not os.path.exists(os.path.dirname(dst)):
                        os.makedirs(os.path.dirname(dst))
                    os.system("cp '%s' '%s'" % (src, dst))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    # two_stage_n_s()
    # two_stage_avg_n_s()
    # three_stage_n_s()
    # three_stage_mod_b_s()
    # three_stage_mod_n_s()
    # three_stage_avg_n_s()
    # three_stage_avg_n_s_all()
    # three_stage_mod_avg_n_s_all()
    # three_stage_both_avg_n_s_all()
    three_stage_both_avg_n_s()
    # three_stage_mod_avg_b_s()
    # three_stage_mod_avg_n_s()
    # three_stage_comparison()
    # data_copy()

    # g = GraphCollections.get('loc-brightkite_edges')
    # print(g[Stat.DEGREE_DISTR])
