import logging
from operator import itemgetter

import kdtree
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import numpy as np


from base.cgraph import MyGraph
from crawlers.cbasic import Crawler

logger = logging.getLogger(__name__)


def count_dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


class KNN_UCB_Crawler(Crawler):
    short = 'KNN-UCB'

    def __init__(self, graph: MyGraph, initial_seed=-1, alpha=0.5, delta=0.5, k=10, n0=0, **kwargs):
        """

        :param graph: investigated graph
        :param initial_seed: start node
        :param alpha: search ratio
        :param delta: -
        :param k: number of nearest neighbors to estimate expected reward
        :param n0: number of starting nodes
        """

        if initial_seed != -1:
            kwargs['initial_seed'] = initial_seed
        super().__init__(graph, alpha=alpha, delta=delta, k=k, n0=n0, **kwargs)

        # dictionaries that store observed and crawled nodes feature vectors
        # and for crawled nodes number of open nodes after crawling
        self.dct_observed = dict()  # node -> (feature vector, need_update)
        self.dct_crawled = dict()  # node -> (feature vector, need_update, y_j)
        self.k = k
        self.alp = alpha
        self.vec_len = 1
        self.update_period = 1
        self.update_add_per = 15
        self.n0 = n0
        self.flag_tree = True
        self.knn_model = NearestNeighbors(n_neighbors=k)
        self.arr = [[], []]

        # pick a random seed from original graph
        if len(self._observed_set) == 0 and n0 < 1:
            if initial_seed == -1:
                initial_seed = self._orig_graph.random_node()
            self.observe(initial_seed)

        # self.observe(initial_seed)
        # initial_seed = self._orig_graph.random_node()
        # self.observed_set.add(initial_seed)

    def expected_reward(self, arr):
        """
        Calculation of the expected reward for the node
        :param arr: array of distances from the calculated node to the crawled nodes
        """
        f = 0
        sigm = 0
        if len(self.crawled_set) >= self.k:
            kol_kNN = self.k
        else:
            kol_kNN = len(self.crawled_set)
        # calculate expected reward from crawling current node
        flag = 0
        kol = 0
        for j in range(kol_kNN):
            if arr[j][1] == 0:
                flag = 1
                kol += 1
                f += self.dct_crawled.get(arr[j][0])[2]
                sigm += arr[j][1]
            elif flag == 0:
                f += self.dct_crawled.get(arr[j][0])[2] / arr[j][1]
                sigm += arr[j][1]
                kol += 1

        f /= kol
        sigm /= kol
        return f + self.alp * sigm

    def calc_node_feat_vec(self, seed):
        """
        Calculation of feature vector for node.
        vert_degree - degree of current node
        aver_neb_degree - average degree of adjacent crawled nodes
        max_neb_degree - maximum degree of adjacent crawled nodes
        kol_triangle - the number of triangles containing the current node

        :param seed: the node for which the feature vector is calculated
        """

        vert_degree = self._observed_graph.deg(seed)
        max_neb_degree = 0
        med_neb_degree = 0
        arr = []
        aver_neb_degree = 0
        kol_craw_neb = 0
        per_craw_neb = 0
        for j in self._observed_graph.neighbors(seed):
            if j in self.crawled_set:
                aver_neb_degree += self._observed_graph.deg(j)
                arr.append(self._observed_graph.deg(j))
                kol_craw_neb += 1
                if self._observed_graph.deg(j) > max_neb_degree:
                    max_neb_degree = self._observed_graph.deg(j)
        if len(arr) != 0:
            med_neb_degree = np.median(arr)
        else:
            med_neb_degree = 0
        if kol_craw_neb != 0:
            aver_neb_degree /= kol_craw_neb
        aver_neb_degree = round(aver_neb_degree)
        per_craw_neb = kol_craw_neb / vert_degree

        # vec = np.array([vert_degree, med_neb_degree, max_neb_degree, per_craw_neb])
        # vec = np.array([vert_degree, med_neb_degree, aver_neb_degree, per_craw_neb])
        vec = np.array([vert_degree])
        # vec = np.array([vert_degree, aver_neb_degree, med_neb_degree])
        return vec
        # dct_crawled[i] = vec, False, dct_crawled.get(i)[2]
        # self.dct_crawled[i][0] = vec
        # self.dct_crawled[i][1] = False

    def knn(self, seed, vec):
        len_crawl = len(self.crawled_set)
        n_neighbors = self.k if len_crawl >= self.k else len_crawl
        if self.flag_tree:
            self.arr = [[], []]
            for i in self.crawled_set:
                self.arr[0].append(i)
                # print(self.dct_observed.get(seed)[1])
                dist = count_dist(vec, self.dct_crawled.get(i)[0])
                # print(dist)
                self.arr[1].append([dist])
            # print(seed)
            # print(self.arr)

            # len_crawl = len(self.crawled_set)
            # n_neighbors = self.k if len_crawl >= self.k else len_crawl
            self.knn_model = NearestNeighbors(n_neighbors=n_neighbors)
            self.knn_model.fit(self.arr[1])
            self.flag_tree = False
        arr_ind = self.knn_model.kneighbors([[0]], n_neighbors=n_neighbors)[1][0]

        # if len_crawl >= self.k:
        #     knn_model = NearestNeighbors(n_neighbors=self.k)
        #     knn_model.fit(arr[1])
        #     arr_ind = knn_model.kneighbors([[0]], n_neighbors=self.k)[1][0]
        # else:
        #     knn_model = NearestNeighbors(n_neighbors=len_crawl)
        #     knn_model.fit(arr[1])
        #     arr_ind = knn_model.kneighbors([[0]], n_neighbors=len_crawl)[1][0]
        # print(arr_ind)
        arr_dist = []
        for i in arr_ind:
            # print(i)
            arr_dist.append((self.arr[0][i], self.arr[1][i][0]))
        # print(arr_dist)
        return arr_dist
        # return []

    def crawl(self, seed: int):
        # calculate the number of open nodes after crawling current node
        res = super().crawl(seed)
        bool = True
        self.dct_crawled.update({seed: (np.zeros((self.vec_len)), bool, len(res))})

        # for key, value in self.dct_observed.items():
        #     print(key)
        # print(self._observed_set)
        # print(self.crawled_set)
        # print(self.dct_crawled)
        for i in self._observed_graph.neighbors(seed):
            # print(i)
            if i in self.crawled_set:
                # print(self.crawled_set)
                # print(i)
                self.dct_crawled.update({i: (self.dct_crawled.get(i)[0], bool, self.dct_crawled.get(i)[2])})
            else:
                # print(self.crawled_set)
                # print(i, ' !')
                if self.dct_observed.get(i) is not None:
                    self.dct_observed.update({i: (self.dct_observed.get(i)[0], True, 0, True)})
                else:
                    self.dct_observed.update({i: (np.ones((self.vec_len)), True, 0, True)})
        # print(self.dct_crawled)
        # print(res)
        # print("!!!")
        return res

    def next_seed(self):

        if len(self.crawled_set) >= 20 and len(self.crawled_set) < 40:
            self.update_period = 5
        elif len(self.crawled_set) % (30 + 2 * self.update_period) == 0:
            self.update_period += self.update_add_per
        # elif self.update_period < 50 and len(self.crawled_set) % 50 == 0:
        #     self.update_period += 10

        # print("!!!")
        # print(self._observed_set)
        # print(self.crawled_set)
        # print("!!!")
        # initial_seed = self._orig_graph.random_node()
        # self.observed_set.add(initial_seed)

        if self.n0 >= 1:
            self.n0 -= 1
            while True:
                random_seeds = self._orig_graph.random_nodes().pop()
                # print(random_seeds)
                if self.crawled_set is None:
                    break
                elif random_seeds not in self.crawled_set:
                    break
            return random_seeds
        else:
            best_node = -1
            best_expected_gain = -1

            # calculation of feature vector for crawled nodes
            if len(self.crawled_set) > 1:
                for i in self.crawled_set:

                    # if self.dct_crawled.get(i) is not None:
                    # if i in self.dct_crawled:
                    if self.dct_crawled.get(i)[1]:
                        # if 0 == 0:
                        vec = self.calc_node_feat_vec(i)
                        # print(self.crawled_set)
                        # print(i)
                        self.dct_crawled.update({i: (vec, False, self.dct_crawled.get(i)[2])})
            else:
                for i in self.observed_set:
                    best_node = i

            # print(self.dct_observed)
            # print(self.dct_crawled)

            # calculation of feature vector for observed nodes
            if len(self.crawled_set) != 0:
                for i in self.observed_set:
                    # if i in self.dct_observed:
                    # print(self.crawled_set)
                    # print(i)
                    if self.dct_observed.get(i) is not None:
                        if self.dct_observed.get(i)[1]:
                            # if 0 == 0:
                            vec = self.calc_node_feat_vec(i)
                            self.dct_observed.update({i: (vec, False, self.dct_observed.get(i)[2], self.dct_observed.get(i)[3])})
                    elif len(self.crawled_set) == 1:
                        vec = self.calc_node_feat_vec(i)
                        self.dct_observed.update({i: (vec, False, self.dct_observed.get(i)[2], self.dct_observed.get(i)[3])})
            # print(self.dct_observed)
            # print(self.dct_crawled)

            # choosing the best node from observed nodes for crawling
            if len(self.crawled_set) > 1:
                for key, value in self.dct_observed.items():
                    if value[3] or len(self.crawled_set) % self.update_period == 0:
                        # find k nearest neighbors
                        arr = []
                        # for i in self.crawled_set:
                        #     arr.append((i, count_dist(value[0], self.dct_crawled.get(i)[0])))
                        # arr.sort(key=itemgetter(1))
                        # print(arr)
                        arr = self.knn(key, value[0])

                        exp_rev = self.expected_reward(arr)
                        self.dct_observed.update({key: (value[0], value[1], exp_rev, len(self.crawled_set) < 20)})
                        # if len(self.crawled_set) >= 20:
                        #     self.dct_observed.update({key: (value[0], value[1], exp_rev, False)})
                        # else:
                        #     self.dct_observed.update({key: (value[0], value[1], exp_rev, True)})
                    else:
                        exp_rev = value[2]

                    if(exp_rev > best_expected_gain):
                        best_expected_gain = exp_rev
                        best_node = key
            self.flag_tree = True
            # print(self.dct_observed)
            # print(self.dct_crawled)
            # print(best_node)
            if len(self.crawled_set) > 0:
                if self.dct_observed.get(best_node) is not None:
                    self.dct_crawled.update({best_node: (self.dct_observed.get(best_node)[0], False, 0)})
                    del self.dct_observed[best_node]


            assert best_node != -1
            return best_node
        # return self.crawled_set
