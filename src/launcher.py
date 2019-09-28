import json
import multiprocessing as mp
import os
import random
import sys

from tqdm import tqdm

from crawling_algorythms import Crawler_MOD, Crawler_RW, Crawler_RC, Crawler_DFS, Crawler_BFS, Crawler_DE
from utils import import_graph, get_percentile


def get_dump_name(crawler, node_seed, counter, budget):
    return f'{crawler.method}-seed_{node_seed}-iter_{counter}_of_{budget}.json'


def treading_crawler(big_graph, crawling_method, node_seed, budget, percentile_set, dumps_dir):
    """
    Paralleling algorythm. 1 process for 1 seed. Using global multiprocessing.Queue for export results
    :param seed_num:
    :param big_graph:
    :param crawling_method:
    :param node_seed:
    :param budget:
    :param percentile_set:
    :return:
    """
    crawler = crawling_method(big_graph, node_seed=node_seed, budget=budget,
                              percentile_set=percentile_set)

    # total = dict({'nodes': big_graph.number_of_nodes()},
    # **{i: len(percentile_set[i]) for i in percentile_set})
    # print('Total:', total)
    try:
        # for counter in atpbar(range(budget), name=f'{crawling_method}-{node_seed}'):
        # for counter in range(budget):
        for counter in tqdm(range(budget), desc=f'{crawler.method}-{node_seed}'):
            # if (counter % 1000 == 1) or (counter == budget - 1):
            #     # print('Dumping history:', crawling_method, counter, seed_num)
            #     dump_file_name = get_dump_name(crawler, node_seed, counter, budget)
            #     with open(os.path.join(dumps_dir, dump_file_name), 'w') as thread_dump_file:
            #         json.dump(crawler.observed_history, thread_dump_file)
            crawler.sampling_process()
    finally:
        dump_file_name = get_dump_name(crawler, node_seed,
                                       len(crawler.observed_history['nodes']), budget)
        with open(os.path.join(dumps_dir, dump_file_name), 'w') as thread_dump_file:
            json.dump(crawler.observed_history, thread_dump_file)


def crawl_one_graph(graph_name, methods, budget, seed_count, top_percentile):
    big_graph = import_graph(graph_name)
    budget = min(big_graph.number_of_nodes(), budget)
    percentile, percentile_set = get_percentile(big_graph, graph_name,
                                                top_percentile)  # берём топ 10 процентов вершин
    if graph_name == 'gnutella':  # большой костыль.Мы брали не тот эксцентриситет
        percentile_set['eccentricity'] = set(big_graph.nodes()).difference(
            percentile_set['eccentricity'])  # XXX

    dumps_dir = '../results/dumps/' + graph_name + '/'
    os.makedirs(dumps_dir, exist_ok=True)

    seeds = random.sample(set(big_graph.nodes), seed_count)

    # pool = mp.Pool(processes=seed_count)
    #
    # for method in methods:
    #     for seed_num in range(seed_count):
    #         pool.apply_async(treading_crawler,
    #                          args=(
    #                              big_graph, CRAWLING_METHODS[method], seeds[seed_num],
    #                              budget, percentile_set, dumps_dir),
    #                          error_callback=print)
    #
    # pool.close()
    # pool.join()

    threads = []
    for method in methods:
        for seed in seeds:
            # treading_crawler(big_graph, CRAWLING_METHODS[method], seed, budget, percentile_set,
            #                  dumps_dir)
            thread = mp.Process(
                target=treading_crawler,
                args=(
                    big_graph, CRAWLING_METHODS[method], seed,
                    budget, percentile_set, dumps_dir),
            )

            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


CRAWLING_METHODS = {
    'RW': Crawler_RW,
    'RC': Crawler_RC,
    'DFS': Crawler_DFS,
    'BFS': Crawler_BFS,
    'MOD': Crawler_MOD,
    # 'MED': Crawler_MED,
    # 'MEUD': Crawler_MEUD,
    'DE': Crawler_DE
}
BUDGET = 1000000
SEED_COUNT = 8
TOP_PERCENTILE = 10

GRAPH_NAMES = [
    'importing',
    'wikivote',
    'hamsterster',
    'DCAM',
    'gnutella',
    'dblp2010',
    'github'
]  # 'slashdot',

if __name__ == '__main__':
    graph_name = sys.argv[1]
    methods = sys.argv[2:]
    if not methods or 'all' in methods:
        methods = CRAWLING_METHODS.keys()
    crawl_one_graph(graph_name, methods, BUDGET, SEED_COUNT, TOP_PERCENTILE)
