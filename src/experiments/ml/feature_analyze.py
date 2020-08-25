import logging
from math import log

from base.cgraph import MyGraph
from crawlers.cbasic import definition_to_filename, CrawlerWithInitialSeed, Crawler, MaximumExcessDegreeCrawler, \
    RandomCrawler
from crawlers.ml.regression_reward import RegressionRewardCrawler
from crawlers.ml.with_features import CrawlerWithFeatures
from graph_io import GraphCollections
from graph_stats import Stat


class FeatureAnalyzerCrawler(CrawlerWithFeatures):
    """
    Runs a given crawler and measures features of each nodes before it is crawled.
    """
    short = "FAC"

    def __init__(self, graph: MyGraph, crawler_def, features=['OD', 'CC', 'CNF'], **kwargs):
        self.crawler = c = Crawler.from_definition(graph, crawler_def)
        super().__init__(graph=graph, crawler_def=crawler_def, features=features,
                         observed_graph=c._observed_graph, observed_set=c._observed_set, crawled_set=c._crawled_set,
                         **kwargs)
        self._node_reward = {}
        self.next_seed = c.next_seed

    def crawl(self, seed: int):
        self._node_clust[seed] = self._observed_graph.clustering(seed)
        self.update_feature(seed)

        res = self.crawler.crawl(seed)  # do not call CrawlerWithFeatures.crawl()
        self._node_reward[seed] = log(1 + len(res))
        return res


def fac():
    # g = GraphCollections.get('soc-BlogCatalog')
    # g = GraphCollections.get('socfb-Bingham82')
    # g = GraphCollections.get('ego-gplus')
    g = GraphCollections.get('livemocha')
    # g = GraphCollections.get('digg-friends')

    # crawler = FeatureAnalyzerCrawler(g, crawler_def=(RandomCrawler, {'initial_seed': 2}), features=['OD', 'CC'])
    crawler = FeatureAnalyzerCrawler(g, crawler_def=(MaximumExcessDegreeCrawler, {'initial_seed': 2}), features=['OD', 'CC'])
    # crawler = FeatureAnalyzerCrawler(g, crawler_def=(RegressionRewardCrawler, {'initial_seed': 2, 'features': ['OD', 'CC']}), features=['OD', 'CC'])

    logging.info("Crawling...")
    crawler.crawl_budget(1000)
    logging.info("done.")

    node_feature = crawler._node_feature
    # cnode_feature = crawler.crawler._node_feature
    # for n, f in node_feature.items():
    #     assert f == cnode_feature[n]
    node_reward = crawler._node_reward
    cc_ix = crawler.features.index('CC')
    # cnf_ix = crawler.features.index('CNF')
    od_ix = crawler.features.index('OD')

    from matplotlib import pyplot as plt
    # Draw heapmap for feature-reward
    import numpy as np
    bins = 100
    reward_dict = {}
    rewards = np.zeros((bins, bins))
    max_x = log(1+g[Stat.MAX_DEGREE])

    for node, feat_vec in node_feature.items():
        cc = feat_vec[cc_ix]
        # cnf = feat_dict[cnf_ix]
        od = feat_vec[od_ix]
        x = min(bins-1, int(bins * od / max_x))  # [1, max_deg]
        y = min(bins-1, int(bins * cc))  # [0,1]
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


def analyze_features():
    # g = GraphCollections.get('soc-BlogCatalog')
    g = GraphCollections.get('socfb-Bingham82')
    # g = GraphCollections.get('ego-gplus')
    # g = GraphCollections.get('livemocha')
    # g = GraphCollections.get('digg-friends')
    crawler = RegressionRewardCrawler(g, initial_seed=2, features=['OD', 'CC'])
    # crawler = BreadthFirstSearchCrawler(g, initial_seed=2, features=['OD', 'CC'])

    logging.info("Crawling...")
    crawler.crawl_budget(300)
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
        # cnf = feat_dict[1]
        od = feat_dict[1]
        x = min(bins-1, int(bins * od / max_x))  # [1, max_deg]
        y = min(bins-1, int(bins * cc))  # [0,1]
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


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # analyze_features()
    fac()
