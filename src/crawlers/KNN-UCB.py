import logging
import snap
import numpy as np


from base.cgraph import MyGraph
from crawlers.cbasic import Crawler

logger = logging.getLogger(__name__)

class KNN_UCB_Crawler(Crawler):

    def count_dist(self, v1, v2):
        return np.sqrt(np.sum((v1-v2)**2))

    def next_seed(self, k, alf):

        initial_seed = self._orig_graph.random_node()
        self.observed_set.add(initial_seed)

        dct_observed = dict()
        dct_crawled = dict()

        while(len(self.observed_set) > 0):

            best_vert = -1
            best_expected_gain = -1

            for i in range(self.crawled_set):

                if dct_crawled.get(i) is not None:
                    # if dct_crawled.get(i)[1]:
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
                        dct_crawled.update({i, (vec, False, dct_crawled.get(i)[2])})


            for i in range(self.observed_set):
                if dct_observed.get(i) is not None:
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
                        dct_observed.update({i, (vec, False)})

            for key, value in dct_observed:
                arr = []
                for i in range(self.crawled_set):
                    arr.append((key, self.count_dist(value[0], dct_crawled.get(i)[0])))
                arr.sort(key=lambda x: x[1])

                f = 0
                sigm = 0

                if len(self.crawled_set) >= k:
                    flag = 0
                    kol = 0
                    for j in range(k):
                        if arr[j][1] == 0:
                            flag = 1
                            kol += 1
                            f += dct_crawled.get(j)[2]
                            sigm += arr[j][1]
                        elif flag == 0:
                            f += dct_crawled.get(j)[2] / arr[j][1]
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
                            f += dct_crawled.get(j)[2]
                            sigm += arr[j][1]
                        elif flag == 0:
                            f += dct_crawled.get(j)[2] / arr[j][1]
                            sigm += arr[j][1]
                    f /= kol
                    sigm /= kol

                if(f + alf * sigm > best_expected_gain):
                    best_expected_gain = f + alf * sigm
                    best_vert = key

            y = 0
            for j in MyGraph.neighbors(self, best_vert):
                if not(j in self.crawled_set or j in self.observed_set):
                    y += 1
            dct_crawled.update({best_vert, (np.array((1, 1, 1)), False, y)})

            dct_observed.pop(best_vert)
            self.crawl(best_vert)

        return self.crawled_set