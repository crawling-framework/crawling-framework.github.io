# ###
# Make sure you have followed all the instructions in Readme.md
#

import os
import logging
from utils import PICS_DIR

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
# logging.getLogger().setLevel(logging.INFO)


def graph_handing():
    """ 1. Automatic graph downloading
    """
    print("\n1. Automatic graph downloading\n")
    from graph_io import GraphCollections

    # Graph name is searched in konect then networkrepository namespaces.
    # If not present on disk, the graph will be downloaded.
    g = GraphCollections.get(name='dolphins')
    print("%s. V=%s, E=%s. At '%s'" % (g.name, g.nodes(), g.edges(), g.path))

    # We can explicitly specify collection, whether to extract giant component, leave self-loops.
    # Flags are applied only once when the graph is downloaded.
    g = GraphCollections.get(name='soc-wiki-Vote', collection='netrepo', giant_only=True, self_loops=False)
    print("%s. V=%s, E=%s. At '%s'" % (g.name, g.nodes(), g.edges(), g.path))

    # One can also use own graph if put it to "data/other/graph_name.ij" directory.
    # Edge list format is supported.
    g = GraphCollections.get(name='example', collection='other', giant_only=True, self_loops=False)
    print("%s. V=%s, E=%s. At '%s'" % (g.name, g.nodes(), g.edges(), g.path))


def stats_computing():
    """ 2. Statistics computation
    """
    print("\n2. Statistics computation\n")
    from graph_io import GraphCollections
    from graph_stats import Stat

    # Once graph is downloaded, it will be loaded from file
    g = GraphCollections.get(name='dolphins')

    # Iterate over all available statistics
    # Once statistics are computed, they are saved to file
    for stat in Stat:
        print("%s = %s" % (stat.description, g[stat]))

    # Statistics computation can be run from terminal, e.g.:
    # (run 'python3 graph_stats.py -h' to see help)
    command = 'python3 graph_stats.py -n soc-wiki-Vote -s DEGREE_DISTR BETWEENNESS_DISTR'
    print("running command '%s'" % command)
    os.system(command)


def visualize_crawling():
    """ 3. Visualize crawler
    """
    print("\n3. Visualize crawler\n")
    from crawlers.cbasic import BreadthFirstSearchCrawler
    from running.metrics_and_runner import TopCentralityMetric
    from running.visual_runner import CrawlerVisualRunner
    from graph_stats import Stat, get_top_centrality_nodes
    from graph_io import GraphCollections

    g = GraphCollections.get('dolphins')

    # Define BFS crawler starting at node 1
    crawler_def = (BreadthFirstSearchCrawler, {'initial_seed': 1})
    # crawler_def = (MaximumObservedDegreeCrawler, {'initial_seed': 1})

    # Top-p nodes are the target
    p = 0.1

    # Define recall metric for all nodes (observed+crawled) with target set equal to top-10% by degree centrality.
    # For args description see constructor docs.
    metric_def = (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'})

    # Target set is top-p degree nodes
    target_set = get_top_centrality_nodes(g, Stat.DEGREE_DISTR, count=int(p*g.nodes()))

    # Create visualizer for the graph which will make 20 crawling steps by 1 step
    cvr = CrawlerVisualRunner(g, crawler_def, metric_def, budget=20, step=1)

    # Run visualizer with drawing parameters.
    cvr.run(draw_orig=True, bold_edges=True, labels=False, make_gif=True, target_set=target_set)

    # Picture for each step is saved to file as well as a gif.
    print("See pics at", cvr._save_dir())


def animated_crawler_runner():
    """ 4. Animated crawler runner
    """
    print("\n4. Animated crawler runner\n")
    from crawlers.cbasic import MaximumObservedDegreeCrawler, BreadthFirstSearchCrawler
    from crawlers.cadvanced import DE_Crawler
    from running.animated_runner import AnimatedCrawlerRunner
    from running.metrics_and_runner import TopCentralityMetric
    from graph_stats import Stat
    from graph_io import GraphCollections

    g = GraphCollections.get('petster-hamster')

    # Top-p nodes are the target.
    p = 0.01

    # Define several crawler configurations.
    crawler_defs = [
        (BreadthFirstSearchCrawler, {'initial_seed': 1}),
        (MaximumObservedDegreeCrawler, {'batch': 1, 'initial_seed': 1}),
        (DE_Crawler, {'initial_seed': 1}),
    ]

    # Define several metrics.
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'crawled'}),
    ]

    # Create runner which will dynamically visualize measurements.
    acr = AnimatedCrawlerRunner(g, crawler_defs, metric_defs, budget=200, step=10)
    # Run and save the result picture.
    acr.run(ylims=(0, 1), xlabel='iteration, n', ylabel='metrics value', swap_coloring_scheme=False,
            save_to_file=os.path.join(PICS_DIR, g.name, 'demo'))


def parallel_crawler_runner():
    """ 5. Run series of experiments in parallel. For every crawling method calculates every metric.
    """
    print("\n5. Run series of experiments in parallel\n")
    from crawlers.cbasic import RandomWalkCrawler, RandomCrawler, MaximumObservedDegreeCrawler, \
        BreadthFirstSearchCrawler, DepthFirstSearchCrawler, SnowBallCrawler
    from crawlers.cadvanced import DE_Crawler
    from crawlers.multiseed import MultiInstanceCrawler
    from running.history_runner import CrawlerHistoryRunner
    from running.metrics_and_runner import TopCentralityMetric
    from graph_stats import Stat
    from graph_io import GraphCollections

    # Define several crawler configurations.
    crawler_defs = [
        (RandomWalkCrawler, {}),
        (RandomCrawler, {}),
        (BreadthFirstSearchCrawler, {}),
        (DepthFirstSearchCrawler, {}),
        (SnowBallCrawler, {'p': 0.1}),
        (MaximumObservedDegreeCrawler, {'batch': 1}),
        (MaximumObservedDegreeCrawler, {'batch': 10}),
        (DE_Crawler, {}),
        (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {})}),
    ]

    # Top-p nodes are the target.
    p = 0.01
    # Define recall metrics corresponding 6 node centralities.
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]

    # Set the number of random seeds to start from.
    n_instances = 8

    # Run crawling for several graphs
    for graph_name in ['petster-hamster', 'soc-wiki-Vote']:
        g = GraphCollections.get(graph_name)
        print("%s. V=%s, E=%s." % (g.name, g.nodes(), g.edges()))

        # Create runner which will run all and save measurements history to file.
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)

        # Run parallel execution with limitations on the number of concurrent processes and the amount of memory.
        chr.run_parallel_adaptive(n_instances, max_cpus=4, max_memory=6)
        print('\n\n')


def run_missing():
    """ 6. Run missing configurations
    """
    print("\n6. Run missing configurations\n")
    from crawlers.cbasic import SnowBallCrawler
    from crawlers.cadvanced import DE_Crawler
    from running.history_runner import CrawlerHistoryRunner
    from running.metrics_and_runner import TopCentralityMetric
    from graph_stats import Stat
    from graph_io import GraphCollections

    # Define some new configuration or an old one that contains uncomputed combinations.
    crawler_defs = [
        (SnowBallCrawler, {'p': 0.5}),
        (DE_Crawler, {}),
    ]
    p = 0.01
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
    ]

    # Set the number of random seeds to start from.
    n_instances = 10

    # If some of the combinations were already computed earlier, we can run only missing ones.
    for graph_name in ['petster-hamster', 'soc-wiki-Vote']:
        g = GraphCollections.get(graph_name)
        print("%s. V=%s, E=%s." % (g.name, g.nodes(), g.edges()))

        # Again create runner which will run all and save measurements history to file.
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)

        # Run missing configurations (detected after searching files in corresponding folders).
        # Again using limitations on the number concurrent processes and the amount of memory.
        chr.run_missing(n_instances, max_cpus=4, max_memory=6)


def merge_and_visualize():
    """ 7. Merge and visualize results of a series of experiments
    """
    print("\n7. Merge and visualize results of a series of experiments\n")
    from running.merger import ResultsMerger
    from crawlers.cbasic import RandomWalkCrawler, RandomCrawler, MaximumObservedDegreeCrawler, \
        BreadthFirstSearchCrawler, DepthFirstSearchCrawler, SnowBallCrawler
    from crawlers.cadvanced import DE_Crawler
    from crawlers.multiseed import MultiInstanceCrawler
    from running.history_runner import CrawlerHistoryRunner
    from running.metrics_and_runner import TopCentralityMetric
    from graph_stats import Stat
    from graph_io import GraphCollections

    # Define a configuration of crawlers and metrics.
    crawler_defs = [
        (RandomWalkCrawler, {}),
        (BreadthFirstSearchCrawler, {}),
        (MaximumObservedDegreeCrawler, {'batch': 1}),
        (MaximumObservedDegreeCrawler, {'batch': 10}),
        (DE_Crawler, {}),
        (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {}), 'name': "Multi-5-MOD"}),
    ]
    p = 0.01
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
    ]
    graph_names = ['petster-hamster', 'soc-wiki-Vote']
    n_instances =8

    # Run missing combination if any. It is useful when some runs of the initial configuration failed.
    for graph_name in graph_names:
        g = GraphCollections.get(graph_name)
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
        chr.run_missing(n_instances, max_cpus=4, max_memory=6)

    # Create a merger which can read measurements data from files, aggregate and draw various plots.
    # Note it takes 3-dimensional input: graphs X crawlers X metrics
    crm = ResultsMerger(graph_names, crawler_defs, metric_defs, n_instances=6)

    # Draw crawler curves for each graph and each metric
    crm.draw_by_crawler(x_lims=(0, 1), x_normalize=True, draw_error=True, scale=4)

    # Draw metric curves for each graph and each crawler
    crm.draw_by_metric(x_lims=(0, 1), x_normalize=True, draw_error=True, scale=4)

    # Draw crawler/metric curves for each graph
    crm.draw_by_metric_crawler(x_lims=(0, 1), x_normalize=True, draw_error=False, scale=6)

    # Draw AUCC values for each graph and each metric
    crm.draw_aucc(aggregator='AUCC', scale=3, xticks_rotation=90)

    # Draw weighted AUCC values for each graph and each metric
    crm.draw_aucc(aggregator='wAUCC', scale=3, xticks_rotation=90)

    # Draw for each crawler weighted AUCC scores aggregated by all graphs and metrics
    crm.draw_winners(aggregator='wAUCC', scale=6, xticks_rotation=60)

    # Show all plots
    crm.show_plots()


if __name__ == '__main__':
    # 1. Automatic graph downloading
    graph_handing()

    # 2. Statistics computation
    stats_computing()

    # 3. Visualize crawler
    visualize_crawling()

    # 4. Animated crawler runner
    animated_crawler_runner()

    # 5. Run series of experiments in parallel
    parallel_crawler_runner()

    # 6. Run missing configurations
    run_missing()

    # 7. Merge and visualize results of a series of experiments
    merge_and_visualize()
