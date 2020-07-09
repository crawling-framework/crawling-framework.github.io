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


        while(len(self.observed_set) > 0):

            best_vert = -1
            expected_gain = -1

            dct = dict()

            for i in range(self.observed_set):

                if dct.get(i) is not None:
                    if dct.get(i)[1]:
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # вычислить метрики для вершины
                        vert_degree = 1
                        max_neb_degree = 10
                        aver_neb_degree = 5
                        kol_triangl = 0
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                        vec = np.array((vert_degree, max_neb_degree, aver_neb_degree, kol_triangl))
                        dct.update({i, (vec, False)})

            for i in range(self.observed_set):
                arr = []
                for key, value in dct:
                    if key != i:
                        arr.append((key, self.count_dist(value[0], dct.get(i)[0])))
                arr.sort(key=lambda x: x[1])

                f = 0
                sigm = 0

                for j in range(k):
                    # !!!!!!!!!!!!!!!!!!!
                    # f += y[j] / arr[j][1]
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!
                    f += 1 / (k * arr[j][1] + 1)  # <-------- +1
                    sigm += arr[j][1] / k

                if(f + alf * 1 > expected_gain):  # <--------- !!!!!!!!!!!!!
                    expected_gain = f + alf * 1
                    best_vert = i

            self.crawl(best_vert)

        return self.crawlers