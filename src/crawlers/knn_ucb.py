import logging
from operator import itemgetter

import numpy as np


from base.cgraph import MyGraph
from crawlers.cbasic import Crawler

logger = logging.getLogger(__name__)


def count_dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


class KNN_UCB_Crawler(Crawler):
    short = 'KNN-UCB'

    def __init__(self, graph: MyGraph, alpha, delta=0.5, k2=10, n0=20):
        # TODO
        self.dct_observed = dict()  # node -> (feature vector, need_update, y_j)
        self.dct_crawled = dict()  # node -> (feature vector, need_update, y_j)

    def crawl(self, seed: int):
        res = super().crawl(seed)
        # TODO
        # y = 0
        # for j in MyGraph.neighbors(self, best_node):
        #     if not(j in self.crawled_set or j in self.observed_set):
        #         y += 1
        # self.dct_crawled.update({best_node, (np.array((1, 1, 1)), False, y)})

        return res

    def next_seed(self, k, alf):

        # initial_seed = self._orig_graph.random_node()
        # self.observed_set.add(initial_seed)

        best_node = -1
        best_expected_gain = -1

        # TODO comment
        for i in self.crawled_set:

            if self.dct_crawled.get(i) is not None:
                # if dct_crawled.get(i)[1]:
                if 0 == 0:
                    # TODO comment
                    vert_degree = self._observed_graph.deg(i)
                    max_neb_degree = 0
                    aver_neb_degree = 0
                    kol_craw_neb = 0
                    for j in MyGraph.neighbors(self, i):
                        if j in self.crawled_set:
                            aver_neb_degree += MyGraph.deg(self, j)
                            kol_craw_neb += 1
                            if MyGraph.deg(self, j) > max_neb_degree:
                                max_neb_degree = MyGraph.deg(self, j)

                    aver_neb_degree /= kol_craw_neb

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # TODO comment
                    # вычислить метрику для треугольников
                    # kol_triangl = 0
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                    vec = np.array((vert_degree, max_neb_degree, aver_neb_degree))
                    # vec = np.array((vert_degree, max_neb_degree, aver_neb_degree, kol_triangl))
                    # dct_crawled.update({i: (vec, False, dct_crawled.get(i)[2])})
                    # dct_crawled[i] = vec, False, dct_crawled.get(i)[2]
                    self.dct_crawled[i][0] = vec
                    self.dct_crawled[i][1] = False

        # TODO comment
        for i in self.observed_set:
            if i in self.dct_observed:
            # if dct_observed.get(i) is not None:
                # if dct_observed.get(i)[1]:
                if 0 == 0:
                    vert_degree = MyGraph.deg(self, i)
                    max_neb_degree = 0
                    aver_neb_degree = 0
                    kol_craw_neb = 0
                    for j in MyGraph.neighbors(self, i):
                        if j in self.crawled_set:
                            aver_neb_degree += MyGraph.deg(self, j)
                            kol_craw_neb += 1
                            if MyGraph.deg(self, j) > max_neb_degree:
                                max_neb_degree = MyGraph.deg(self, j)

                    aver_neb_degree /= kol_craw_neb
                    aver_neb_degree = round(aver_neb_degree)

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # вычислить метрику для треугольников
                    # kol_triangl = 0
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                    vec = np.array((vert_degree, max_neb_degree, aver_neb_degree))
                    # vec = np.array((vert_degree, max_neb_degree, aver_neb_degree, kol_triangl))
                    self.dct_observed.update({i, (vec, False)})

        # TODO comment
        for key, value in self.dct_observed.items():
            # TODO comment
            arr = []
            for i in self.crawled_set:
                arr.append((key, count_dist(value[0], self.dct_crawled.get(i)[0])))
            arr.sort(key=itemgetter(1))

            f = 0
            sigm = 0

            # TODO comment
            if len(self.crawled_set) >= k:
                flag = 0
                kol = 0
                for j in range(k):
                    if arr[j][1] == 0:
                        flag = 1
                        kol += 1
                        f += self.dct_crawled.get(j)[2]
                        sigm += arr[j][1]
                    elif flag == 0:
                        f += self.dct_crawled.get(j)[2] / arr[j][1]
                        sigm += arr[j][1]
                if flag == 0:
                    f /= k
                    sigm /= k
                else:
                    f /= kol
                    sigm /= kol
            else:
                kol = 0
                flag = 0
                for j in range(len(self.crawled_set)):
                    if arr[j][1] == 0:
                        flag = 1
                        kol += 1
                        f += self.dct_crawled.get(j)[2]
                        sigm += arr[j][1]
                    elif flag == 0:
                        f += self.dct_crawled.get(j)[2] / arr[j][1]
                        sigm += arr[j][1]
                        kol += 1
                f /= kol
                sigm /= kol

            if(f + alf * sigm > best_expected_gain):
                best_expected_gain = f + alf * sigm
                best_node = key

        del self.dct_observed[best_node]

        assert best_node != -1
        return best_node
        # return self.crawled_set