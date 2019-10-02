import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from utils import dump_results
from launcher import GRAPH_NAMES


def find_dump_files(base_dir, graph_name):
    graph_dump_dir = os.path.join(base_dir, graph_name)
    file_pattern = r'(\w+)-seed_(\d+)-iter_(\d+)_of_\d+.json'
    for filename in os.listdir(graph_dump_dir):
        match = re.match(file_pattern, filename)
        if match:
            method = match.group(1)
            seed = int(match.group(2))
            budget = int(match.group(3))
            yield method, seed, budget, os.path.join(graph_dump_dir, filename)


def collect_avg_stats(base_dir, graph_name):
    history = defaultdict(lambda: defaultdict(dict))  # budget > method > seed > props

    for method, seed, budget, path in find_dump_files(base_dir, graph_name):
        with open(path) as stats_dump_file:
            history[budget][method][seed] = json.load(stats_dump_file)

    for budget, budget_history in history.items():
        crawler_avg = defaultdict(dict)
        for method, method_history in budget_history.items():
            accumulator = defaultdict(list)
            for seed, seed_history in method_history.items():
                for prop, prop_history in seed_history.items():
                    accumulator[prop].append(prop_history)
            for prop, accumulated_values in accumulator.items():
                if not accumulated_values:
                    continue
                crawler_avg[method][prop] = np.array(accumulated_values).mean(axis=0)
        dump_results(graph_name, crawler_avg, budget_history, budget, base_dir)


def main():
    parser = argparse.ArgumentParser(description='Average traversal metrics by seeds')
    parser.add_argument('graph', choices=GRAPH_NAMES, help='graph name')
    parser.add_argument('-d', '--dumps-dir', default='../results/dumps', dest='dumps_dir',
                        help='dumps directory')
    args = parser.parse_args()
    collect_avg_stats(args.dumps_dir, args.graph)


if __name__ == '__main__':
    main()
