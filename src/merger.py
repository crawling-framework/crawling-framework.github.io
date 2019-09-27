import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from utils import dump_results


def find_dump_files(base_dir, graph):
    graph_dump_dir = os.path.join(base_dir, graph)
    file_pattern = '(\w+)-seed_(\d+)-iter_(\d+)_of_\d+.json'
    for filename in os.listdir(graph_dump_dir):
        match = re.match(file_pattern, filename)
        if match:
            method = match.group(1)
            seed = int(match.group(2))
            budget = int(match.group(3))
            yield method, seed, budget, os.path.join(graph_dump_dir, filename)


def collect_avg_stats(base_dir, graph):
    history = defaultdict(lambda: defaultdict(dict))  # budget > method > seed > props

    for method, seed, budget, path in find_dump_files(base_dir, graph):
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
                crawler_avg[method][prop] = np.array(accumulated_values).mean(axis=0)
        dump_results(graph, crawler_avg, budget_history, budget)


DUMPS_DIR = '../results/dumps'

if __name__ == '__main__':
    graph = sys.argv[1]
    collect_avg_stats(DUMPS_DIR, graph)