import logging
from math import log

from sklearn.linear_model import LinearRegression

from crawlers.cadvanced import DE_Crawler
from crawlers.cbasic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler, RandomCrawler, \
    definition_to_filename, filename_to_definition
from crawlers.ml.regression_reward import RegressionRewardCrawler
from experiments.three_stage import social_names
from graph_io import GraphCollections, rename_results_files
from crawlers.ml.knn_ucb import KNN_UCB_Crawler
from running.animated_runner import AnimatedCrawlerRunner
from running.merger import ResultsMerger
from running.metrics_and_runner import TopCentralityMetric, Metric
from running.history_runner import CrawlerHistoryRunner
from statistics import Stat


def analyze_features():
    # g = GraphCollections.get('soc-BlogCatalog')
    g = GraphCollections.get('socfb-Bingham82')
    # g = GraphCollections.get('ego-gplus')
    # g = GraphCollections.get('livemocha')
    # g = GraphCollections.get('digg-friends')
    crawler = RegressionRewardCrawler(g, initial_seed=2, features=['OD', 'CC', 'CNF'])
    # crawler = BreadthFirstSearchCrawler(g, initial_seed=2, features=['OD', 'CC'])

    logging.info("Crawling...")
    crawler.crawl_budget(500)
    logging.info("done.")

    node_feature = crawler._node_feature
    node_reward = crawler._node_reward

    from matplotlib import pyplot as plt
    # # Draw features for crawled and observed nodes
    #
    # crawled = []  # list of pairs (x, y)
    # observed = []  # list of pairs (x, y)
    # for node, (feat_dict, reward) in node_feature.items():
    #     x = feat_dict['OD']
    #     y = feat_dict['CC']
    #     if node in crawler.crawled_set:
    #         crawled.append((x, y))
    #     else:
    #         observed.append((x, y))
    #
    # xs, ys = zip(*crawled)
    # plt.plot(xs, ys, color='g', marker='.', linestyle='', label='crawled')
    # xs, ys = zip(*observed)
    # plt.plot(xs, ys, color='b', marker='.', linestyle='', label='observed')
    #
    # plt.legend()

    # Draw heapmap for feature-reward
    import numpy as np
    bins = 100
    reward_dict = {}
    rewards = np.zeros((bins, bins))
    max_x = log(1+g[Stat.MAX_DEGREE])
    for node, feat_dict in node_feature.items():
        cc = feat_dict[0]
        cnf = feat_dict[1]
        od = feat_dict[2]
        x = min(bins-1, int(bins * od / max_x))  # [1, max_deg]
        y = min(bins-1, int(bins * cnf))  # [0,1]
        if node in crawler.crawled_set:
            if (x, y) not in reward_dict:
                reward_dict[(x, y)] = []
            r = node_reward[node]
            reward_dict[(x, y)].append(r)
        # else:
        #     observed.append((x, y))
    print(reward_dict)

    # reward_dict[(7, 2)] = 1
    for x in range(bins):
        for y in range(bins):
            if (x, y) in reward_dict:
                rewards[y][x] = np.mean(reward_dict[(x, y)])
            else:
                rewards[y][x] = np.nan
    rewards /= np.nanmax(rewards)
    plt.imshow(rewards, cmap='inferno', vmin=0, vmax=1)
    plt.gca().set_facecolor((0.8, 0.8, 0.8))
    plt.colorbar()
    plt.grid(False)
    # plt.hist2d(rewards)

    plt.title("%s\n%s" % (definition_to_filename(crawler.definition), g.name))
    plt.xlabel('log OD')
    # plt.xscale('log')
    # plt.ylabel('CC')
    plt.ylabel('CNF')
    plt.ylim((-0.5, bins+0.5))
    plt.tight_layout()
    plt.show()


def test_knnucb():
    # g = GraphCollections.get('dolphins')
    # g = GraphCollections.get('Pokec')
    # g = GraphCollections.get('livemocha')
    # g = GraphCollections.get('digg-friends')
    # g = GraphCollections.get('socfb-Bingham82')
    g = GraphCollections.get('soc-BlogCatalog')

    p = 0.01
    # budget = int(0.005 * g.nodes())
    # s = int(budget / 2)

    crawler_defs = [
        # (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        (MaximumObservedDegreeCrawler, {'initial_seed': 2}),
        # (KNN_UCB_Crawler, {'initial_seed': 2, 'features': ['OD', ]}),
        # (KNN_UCB_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF']}),
        # (KNN_UCB_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF', 'CC'], 'tau': -1}),
        # (LinReg_Crawler, {'initial_seed': 2, 'features': ['OD'], 'tau': -1}),
        # (LinReg_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF'], 'tau': -1}),
        # (RandomCrawler, {'initial_seed': 2}),
        # (LinReg_Crawler, {'initial_seed': 2, 'features': ['OD', 'CC'], 'tau': -1}),
        # (LinReg_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF', 'CC'], 'tau': -1}),
        # (LinReg_Crawler, {'initial_seed': 1, 'features': ['OD', 'CNF', 'CC', 'MND', 'AND'], 'tau': -1}),
        # (KNN_UCB_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF']}),
        # (KNN_UCB_Crawler, {'initial_seed': 2, 'features': ['OD', 'CNF', 'CC']}),
        # (MaximumObservedDegreeCrawler, {'initial_seed': 2}),
        # (DE_Crawler, {'name': 'DE'}),

        (RegressionRewardCrawler, {'initial_seed': 2, 'features': ['OD', 'CNF', 'CC'], 'regr': 'LinearRegression', 'regr_args': {}}),
        (RegressionRewardCrawler, {'initial_seed': 2, 'features': ['OD', 'CNF', 'CC'], 'regr': 'KNeighborsRegressor', 'regr_args': {'n_neighbors': 10}}),
    ]
    cd = crawler_defs[1]
    print(cd)
    f = definition_to_filename(cd)
    print(f)
    cd = filename_to_definition(f)
    print(cd)
    crawler_defs[1] = cd

    # RewardMonitor = Metric('RM', lambda crawler: )

    metric_defs = [
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'crawled'}),
        (TopCentralityMetric, {'top': 1, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=1000, )
    acr.run()


def run_comparison():
    p = 0.01

    crawler_defs = [
        # (KNN_UCB_Crawler, {'initial_seed': 1, 'alpha': 0, 'k': 1, 'n0': 50}),
        (MaximumObservedDegreeCrawler, {'name': 'MOD'}),
        # (KNN_UCB_Crawler, {'features': ['OD'], 'name': "KNN-UCB"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CC'], 'name': "KNN-UCB\n[OD+CC]"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CNF'], 'name': "KNN-UCB\n[OD+CNF]"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CNF', 'CC'], 'name': "KNN-UCB\n[OD+CNF+CC]"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CNF', 'CC', 'MND', 'AND'], 'name': "KNN-UCB\n[all 5]"}),
        (RegressionRewardCrawler, {'features': ['OD'], 'tau': -1, 'name': "LinReg"}),
        # (LinReg_Crawler, {'features': ['OD', 'CC'], 'name': "LinReg\n[OD+CC]"}),
        (RegressionRewardCrawler, {'features': ['OD', 'CNF'], 'name': "LinReg\n[OD+CNF]"}),
        # (LinReg_Crawler, {'features': ['OD', 'CNF', 'CC'], 'tau': -1, 'name': "LinReg\n[OD+CNF+CC]"}),
        # (LinReg_Crawler, {'features': ['OD', 'CNF', 'CC', 'MND', 'AND'], 'tau': -1, 'name': "LinReg\n[all 5]"}),
        # (DE_Crawler, {'name': 'DE'}),
    ] + [
        # (LinReg_Crawler, {'features': ['OD', 'CNF'], 'name': "LinReg\n[OD+CNF]"}),
        # (LinReg_Crawler, {'features': ['OD', 'CC'], 'name': "LinReg\n[OD+CC]"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CC'], 'name': "KNN-UCB\n[OD+CC]"}),
        # (KNN_UCB_Crawler, {'features': ['OD', 'CNF', 'CC', 'MND', 'AND'], 'name': "KNN-UCB\n[all 5]"}),

        # (KNN_UCB_Crawler, {'alpha': a, 'k': k}) for a in [0.2, 0.5, 1.0, 5.0] for k in [3, 10, 30]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 10, 'n0': 0})
        # (KNN_UCB_Crawler, {'alpha': a, 'k': 30}) for a in [0.2, 0.5, 1.0, 5.0]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 30, 'n_features': f, 'n0': 30}) for f in [1, 2, 3, 4]
        # (KNN_UCB_Crawler, {'alpha': 0.5, 'k': 30, 'n_features': 1, 'n0': n0}) for n0 in [0]
        # (MaximumObservedDegreeCrawler, {}),
    ]
    metric_defs = [
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'crawled'}),
        (TopCentralityMetric, {'top': 1, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    n_instances = 8
    graph_names = social_names[:20]
    # for graph_name in graph_names:
    #     g = GraphCollections.get(graph_name)
    #     chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs, budget=1000)
    #     chr.run_missing(n_instances, max_cpus=8, max_memory=30)

    crm = ResultsMerger(graph_names, crawler_defs, metric_defs, n_instances, x_lims=(0, 1000))
    crm.draw_by_crawler(x_normalize=False, draw_error=False, scale=3)
    crm.draw_winners('AUCC', scale=3)
    # crm.draw_winners('wAUCC', scale=3)
    # crm.draw_aucc()


def run_original_knnucb():
    # git clone https://bitbucket.org/kau_mad/bandits.git
    # needs pip3 install pymc3
    command = "PYTHONPATH=/home/misha/soft/bandits/:/home/misha/soft/mab_explorer/mab_explorer python3 mab_explorer/sampling.py data/CA-AstroPh.txt -s 0.05 -b 100 -m rn -e 1 -plot ./results"


def reproduce_paper():
    """ Try to reproduce experiments from "A multi-armed bandit approach for exploring partially observed networks"
    https://link.springer.com/article/10.1007/s41109-019-0145-0
    """
    # g = GraphCollections.get('ca-dblp-2012')
    g = GraphCollections.get('ca-AstroPh', 'netrepo')
    n = g[Stat.NODES]

    # Sample = 5% of nodes
    bfs = BreadthFirstSearchCrawler(g, initial_seed=1)
    bfs.crawl_budget(int(0.05 * n))
    observed_graph = bfs._observed_graph
    node_set = bfs.nodes_set
    initial_size = len(node_set)
    print(initial_size)

    # Metric which counts newly observed nodes
    class AMetric(Metric):
        def __init__(self, graph):
            super().__init__(name='new obs nodes', callback=lambda crawler: len(crawler.nodes_set) - initial_size)

    crawler_defs = [
        (MaximumObservedDegreeCrawler, {'observed_graph': observed_graph.copy(), 'observed_set': set(node_set)}),
        (KNN_UCB_Crawler, {'n0': 20, 'n_features': 4, 'observed_graph': observed_graph.copy(), 'observed_set': set(node_set)}),
    ]
    metric_defs = [
        (AMetric, {}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
    ]

    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=1000, step=20)
    acr.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # analyze_features()
    test_knnucb()
    # run_comparison()
    # reproduce_paper()

