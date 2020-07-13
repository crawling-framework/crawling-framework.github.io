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

    def __init__(self, graph: MyGraph, initial_seed=-1, alpha=0.5, delta=0.5, k=10, n0=20, **kwargs):

        if initial_seed != -1:
            kwargs['initial_seed'] = initial_seed
        super().__init__(graph, alpha=alpha, delta=delta, k=k, n0=n0, **kwargs)
        # pick a random seed from original graph
        if len(self._observed_set) == 0:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        # k - number of nearest neighbors to estimate expected reward
        # alpha - search ratio
        # dictionaries that store observed and crawled nodes feature vectors
        # and for crawled nodes number of open nodes after crawling
        self.dct_observed = dict()  # node -> (feature vector, need_update)
        self.dct_crawled = dict()  # node -> (feature vector, need_update, y_j)
        self.k = k
        self.alp = alpha

        # self.observe(initial_seed)
        # initial_seed = self._orig_graph.random_node()
        # self.observed_set.add(initial_seed)

    def crawl(self, seed: int):

        # calculate the number of open nodes after crawling current node
        vert_degree_old = self._observed_graph.deg(seed)
        res = super().crawl(seed)
        vert_degree_new = self._observed_graph.deg(seed)
        self.dct_crawled.update({seed: (np.array([1.0, 1.0, 1.0]), False, vert_degree_new - vert_degree_old)})
        # print(res)
        # print("!!!")
        return res

    def next_seed(self):

        # print("!!!")
        # print(self._observed_set)
        # print(self.crawled_set)
        # print("!!!")
        # initial_seed = self._orig_graph.random_node()
        # self.observed_set.add(initial_seed)

        best_node = -1
        best_expected_gain = -1

        # calculation of feature vector for crawled nodes
        if len(self.crawled_set) > 1:
            for i in self.crawled_set:

                if self.dct_crawled.get(i) is not None:
                # if dct_crawled.get(i)[1]:
                # if 0 == 0:
                    # vert_degree - degree of current node
                    # aver_neb_degree - average degree of adjacent crawled nodes
                    # max_neb_degree - maximum degree of adjacent crawled nodes
                    # kol_triangle - the number of triangles containing the current node
                    vert_degree = self._observed_graph.deg(i)
                    max_neb_degree = 0
                    aver_neb_degree = 0
                    kol_craw_neb = 0
                    for j in self._observed_graph.neighbors(i):
                        if j in self.crawled_set:
                            aver_neb_degree += self._observed_graph.deg(j)
                            kol_craw_neb += 1
                            if self._observed_graph.deg(j) > max_neb_degree:
                                max_neb_degree = self._observed_graph.deg(j)

                        # print(j)
                    aver_neb_degree /= kol_craw_neb

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # вычислить метрику для треугольников
                    # kol_triangle = 0
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                    vec = np.array([vert_degree, max_neb_degree, aver_neb_degree])
                    # vec = np.array([vert_degree, max_neb_degree, aver_neb_degree, kol_triangl])
                    self.dct_crawled.update({i: (vec, False, self.dct_crawled.get(i)[2])})
                    # dct_crawled[i] = vec, False, dct_crawled.get(i)[2]
                    # self.dct_crawled[i][0] = vec
                    # self.dct_crawled[i][1] = False
        else:
            for i in self.observed_set:
                best_node = i

        # calculation of feature vector for observed nodes
        if len(self.crawled_set) != 0:
            for i in self.observed_set:
                # if i in self.dct_observed:
            # if dct_observed.get(i) is not None:
                # if dct_observed.get(i)[1]:
                if 0 == 0:
                    # vert_degree - degree of current node
                    # aver_neb_degree - average degree of adjacent crawled nodes
                    # max_neb_degree - maximum degree of adjacent crawled nodes
                    # kol_triangle - the number of triangles containing the current node
                    vert_degree = self._observed_graph.deg(i)
                    max_neb_degree = 0
                    aver_neb_degree = 0
                    kol_craw_neb = 0
                    for j in self._observed_graph.neighbors(i):
                        if j in self.crawled_set:
                            aver_neb_degree += self._observed_graph.deg(j)
                            kol_craw_neb += 1
                            if self._observed_graph.deg(j) > max_neb_degree:
                                max_neb_degree = self._observed_graph.deg(j)

                    aver_neb_degree /= kol_craw_neb
                    aver_neb_degree = round(aver_neb_degree)

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # вычислить метрику для треугольников
                    # kol_triangle = 0
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                    vec = np.array([vert_degree, max_neb_degree, aver_neb_degree])
                    # vec = np.array([vert_degree, max_neb_degree, aver_neb_degree, kol_triangl])
                    self.dct_observed.update({i: (vec, False)})
        # print(self.dct_observed)
        # print(self.dct_crawled)
        # choosing the best node from observed nodes for crawling
        if len(self.crawled_set) > 1:
            for key, value in self.dct_observed.items():
                # find k nearest neighbors
                arr = []
                for i in self.crawled_set:
                    arr.append((i, count_dist(value[0], self.dct_crawled.get(i)[0])))
                arr.sort(key=itemgetter(1))
                # print(arr)
                f = 0
                sigm = 0

                # calculate expected reward from crawling current node
                if len(self.crawled_set) >= self.k:
                    flag = 0
                    kol = 0
                    for j in range(self.k):
                        if arr[j][1] == 0:
                            flag = 1
                            kol += 1
                            f += self.dct_crawled.get(arr[j][0])[2]
                            sigm += arr[j][1]
                        elif flag == 0:
                            f += self.dct_crawled.get(arr[j][0])[2] / arr[j][1]
                            sigm += arr[j][1]
                    if flag == 0:
                        f /= self.k
                        sigm /= self.k
                    else:
                        f /= kol
                        sigm /= kol
                else:
                    kol = 0
                    flag = 0
                    # print(len(self.crawled_set))
                    for j in range(len(self.crawled_set)):
                        if arr[j][1] == 0:
                            flag = 1
                            kol += 1
                            f += self.dct_crawled.get(arr[j][0])[2]
                            sigm += arr[j][1]
                        elif flag == 0:
                            # print(self.dct_crawled.get(j)[2])
                            f += self.dct_crawled.get(arr[j][0])[2] / arr[j][1]
                            sigm += arr[j][1]
                            kol += 1
                    f /= kol
                    sigm /= kol

                if(f + self.alp * sigm > best_expected_gain):
                    best_expected_gain = f + self.alp * sigm
                    best_node = key
        # print(self.dct_observed)
        # print(self.dct_crawled)
        # print(best_node)
        if len(self.crawled_set) > 0:
            del self.dct_observed[best_node]

        assert best_node != -1
        return best_node
        # return self.crawled_set