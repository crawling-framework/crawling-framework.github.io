import json
import os
import re
from glob import glob

from graph_io import GraphCollections
from utils import RESULT_DIR

import numpy as np


def answer(graph_name='petster-hamster'):
    print('\\\ \hline')
    print('%s ' % graph_name, end="")
    graph_path = os.path.join(RESULT_DIR, graph_name) + '/'
    # print(graph_path)

    crawlers = [file.replace(graph_path, '') for file in glob(graph_path + '*')]
    crawlers.sort()
    # print(crawlers)

    max = -1
    for _, crawler_name in enumerate(crawlers):
        # print(crawler_name)
        crawler_path = os.path.join(graph_path, crawler_name)
        experiments_path = [file.replace(crawler_path, '') for file in glob(crawler_path + '/*')]
        # print(experiments_path[0][1:])
        experiments = os.path.join(graph_path, crawler_name, experiments_path[0][1:])
        # print(experiments)
        exp = [file.replace(experiments, '') for file in glob(experiments + '/*')]
        exp.sort()
        # print(exp)
        count = len(exp)
        average_metrics = []
        for experiment in exp:
            with open(os.path.join(experiments, experiment[1:]), 'r') as f:
                # print(os.path.join(experiments, experiment[1:]))
                imported = json.load(f)
                _, y_arr = zip(*imported.items())
                average_metrics.append(y_arr[-1])
                # print(x_arr)
        avg = np.average(average_metrics)
        if avg > max:
            max = avg
        print(' & %.4f±%.4f' % (avg, np.std(average_metrics)), end="")
    print(' & %.4f' % max, end="")


if __name__ == '__main__':

    social_names = [
        'socfb-Bingham82',  # N=10001,   E=362892,   d_avg=72.57
        'soc-brightkite',  # N=56739,   E=212945,   d_avg=7.51
        'socfb-Penn94',  # N=41536,   E=1362220,  d_avg=65.59
        'socfb-wosn-friends',  # N=63392,   E=816886,   d_avg=25.77
        'soc-slashdot',  # N=70068,   E=358647,   d_avg=10.24
        'soc-themarker',  # N=69317,   E=1644794,  d_avg=47.46
        'soc-BlogCatalog',  # N=88784,   E=2093195,  d_avg=47.15
        'soc-anybeat',
        'soc-twitter-follows',  # N=404719,  E=713319,   d_avg=3.53
        # konect
        'petster-hamster',  # N=2000,    E=16098,    d_avg=16.10
        'ego-gplus',  # N=23613,   E=39182,    d_avg=3.32
        'slashdot-threads',  # N=51083,   E=116573,   d_avg=4.56
        'douban',  # N=154908,  E=327162,   d_avg=4.22
        'digg-friends',  # N=261489,  E=1536577,  d_avg=11.75
        'loc-brightkite_edges',  # N=
        'epinions',  # N=
        'livemocha',  # N=
        'petster-friendships-cat',  # N=148826,  E=5447464,  d_avg=73.21
        'petster-friendships-dog',  # N=426485,  E=8543321,  d_avg=40.06
        'munmun_twitter_social',  # N=465017,  E=833540,   d_avg=3.58
        'com-youtube',  # N=1134890, E=2987624,  d_avg=5.27
        'flixster',  # N=2523386, E=7918801,  d_avg=6.28
        'youtube-u-growth',  # N=3216075, E=9369874,  d_avg=5.83
        'soc-pokec-relationships',  # N=1632803, E=22301964, d_avg=27.32
    ]

    for name in social_names:

         answer(name)