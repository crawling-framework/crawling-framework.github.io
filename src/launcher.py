import argparse
import json
import multiprocessing as mp
import os
import random
import time

from crawling_algorithms import CrawlerMOD, CrawlerRW, CrawlerRC, CrawlerDFS, CrawlerBFS, \
    CrawlerDE
from tqdm import tqdm
from utils import import_graph, get_percentile


def get_dump_name(crawler, node_seed, counter, budget):
    return f'{crawler.method}-seed_{node_seed}-iter_{counter}_of_{budget}.json'


def treading_crawler(big_graph, crawling_method, node_seed, budget,
                     percentile_set, calc_metrics_on, dumps_dir):
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
                              percentile_set=percentile_set,
                              calc_metrics_on_closed=(calc_metrics_on == 'closed'))

    # total = dict({'nodes': big_graph.number_of_nodes()},
    # **{i: len(percentile_set[i]) for i in percentile_set})
    # print('Total:', total)
    progress = tqdm(range(budget), desc=f'{crawler.method}-{node_seed}')
    try:
        for counter in progress:
            # if (counter % 1000 == 1) or (counter == budget - 1):
            #     # print('Dumping history:', crawling_method, counter, seed_num)
            #     dump_file_name = get_dump_name(crawler, node_seed, counter, budget)
            #     with open(os.path.join(dumps_dir, dump_file_name), 'w') as thread_dump_file:
            #         json.dump(crawler.observed_history, thread_dump_file)
            crawler.sampling_process()
    finally:
        crawler.observed_history['time'] = time.time() - progress.start_t
        progress.close()
        dump_file_name = get_dump_name(crawler, node_seed,
                                       len(crawler.observed_history['nodes']), budget)
        with open(os.path.join(dumps_dir, dump_file_name), 'w') as thread_dump_file:
            json.dump(crawler.observed_history, thread_dump_file)


def crawl_one_graph(graph_name, methods, budget, seed_count,
                    top_percentile, calc_metrics_on, output_dir):
    big_graph = import_graph(graph_name)
    budget = min(big_graph.number_of_nodes(), budget)
    percentile, percentile_set = get_percentile(big_graph, graph_name,
                                                top_percentile)  # берём топ 10 процентов вершин
    if graph_name == 'gnutella':  # большой костыль.Мы брали не тот эксцентриситет
        percentile_set['eccentricity'] = set(big_graph.nodes()).difference(
            percentile_set['eccentricity'])  # XXX

    dumps_dir = os.path.join(output_dir, graph_name)
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
                    big_graph, CRAWLING_METHODS[method], seed, budget,
                    percentile_set, calc_metrics_on, dumps_dir),
            )

            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


CRAWLING_METHODS = {
    'RW': CrawlerRW,
    'RC': CrawlerRC,
    'DFS': CrawlerDFS,
    'BFS': CrawlerBFS,
    'MOD': CrawlerMOD,
    # 'MED': CrawlerMED,
    # 'MEUD': CrawlerMEUD,
    'DE': CrawlerDE
}
BUDGET = 1000000
SEED_COUNT = 8
TOP_PERCENTILE = 10

GRAPH_NAMES = [
    'wikivote',
    'hamsterster',
    'DCAM',
    'slashdot',
    'facebook',
    'dblp2010',
    'github'
    # 'importing',
    # 'gnutella',
]


def main():
    parser = argparse.ArgumentParser(description='Traverse graph with given methods '
                                                 'and dump some stats.')
    parser.add_argument('graph', choices=GRAPH_NAMES, help='graph name')
    parser.add_argument('methods', nargs='*', choices=['all', *CRAWLING_METHODS.keys()],
                        default='all', help='crawling methods')
    parser.add_argument('-o', '--output-dir', default='../results/dumps', dest='out',
                        help='output directory')
    parser.add_argument('-m', '--metrics', choices=['closed', 'discovered'], default='closed',
                        help='whether to calculate metrics on closed or discovered vertices',
                        dest='calc_metrics_on')

    args = parser.parse_args()
    if not args.methods or 'all' in args.methods:
        args.methods = CRAWLING_METHODS.keys()
    crawl_one_graph(args.graph, args.methods, BUDGET, SEED_COUNT, TOP_PERCENTILE,
                    args.calc_metrics_on, args.out)


if __name__ == '__main__':
    main()
