#!/usr/bin/env python
# coding: utf-8

# In[90]:

import random
import networkx as nx
import matplotlib.pyplot as plt
import time
#from .vkprint import vkprint
METRICS_LIST = ['degrees', 'k_cores', 'eccentricity', 'betweenness_centrality']

class Crawler:
    """
    Crawler class with a lot of stuff (to comment)
    """
    def __init__(self, big_graph, node_seed, budget, percentile_set):

        self.big_graph = big_graph  # сам конечный граф
        self.total = self.big_graph.number_of_nodes()  # total это общее число вершин
        self.b = min(budget, int(self.total * 0.99))  # бюджет запросов, которые можно сделать
        self.node_seed = node_seed  # начальная сида, с которой начинаем
        self.percentile_set = percentile_set  # множество вершин графа, удовлетворяющих квантилю
        self.method = 'RC'

        self.current = node_seed  # в начале текущая вершина это стартовая нода
        self.v_observed = set()  # множество увиденных вершин
        self.v_observed.add(node_seed)
        self.v_closed = set()  # множество уже обработанных вершин
        self.node_degree = []  # общее число друзей вершины. А не в бюджетном
        self.node_array = []  # последовательность вершин, которые мы обходим
        self.G = nx.Graph()  # наш насемплированный граф
        self.G.add_node(node_seed)  # добавляем начальную вершину

        #self.METRICS_LIST = METRICS_LIST  # список названий полей из properties графа, которые исследуем
        self.counter = 0  # счётчик итераций
        self.observed_history = dict({i: [] for i in METRICS_LIST}, **{'nodes': []})
        ##'degrees': [], 'k_cores': [], 'eccentricity': [], 'betweenness_centrality': []}    # история изменения количества видимых вершин
        ## вот тут мб надо немного переделать
        self.property_history = dict(
            (i, [] * (self.total + 1)) for i in METRICS_LIST)  # история изменения параметров с итерациями
        for prop in METRICS_LIST:
            self.property_history[prop].append(len(self.v_closed.intersection(self.percentile_set[prop])))

        self.node_degree = []  # общее число друзей вершины. А не в бюджетном
        self.node_array = []  # список вершин в порядке обработки

        # self.plot, _ = plt.subplots()
        # print(type())

    def _observing(self):
        """
        пробегаемся по всем друзьям и добавляем их в наш граф и в v_observed
        """
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id, self.current)

    def _change_current(self):  # выбор новой вершины. по умолчанию считаем random crawling
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def sampling_process(self):  # сам  процесс сэмплирования
        """
        One step of sampling process. Noting closed, remembering history for every metric
        :return:
        """
        self._change_current()
        self._observing()
        if self.current in self.v_observed:  # условие для пустых итераций рандом волка tbd - надо исправить
            self.v_observed.remove(self.current)  # теперь она обработанная, а не просто видимая
        self.v_closed.add(self.current)
        self.node_array.append(self.current)  # отметили, что обработали эту вершину
        for prop in METRICS_LIST:# для каждой метрики считаем, сколько вершин из бюджетного множества попало в граф
            self.observed_history[prop].append(len(self.percentile_set[prop].intersection(set(self.v_closed))))
            # self.observed_history.append(len(self.v_observed) + len(self.v_closed))

        self.observed_history['nodes'].append(len(self.v_closed)+len(self.v_observed))
        return self.observed_history  # ,self.property_history)
    # раскомментить если захочется экспортировать граф
    # nx.write_gml(G, "./Sampling/"+ graph_name +"/"+method+'_b='+str(b)+'_seed='+str(n_seed)+'.graph')


class Crawler_RC(Crawler):
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)

    def _change_current(self):
        self.current = random.sample(self.v_observed, 1)[0]


class Crawler_DFS(Crawler):
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)
        self.method = 'DFS'
        self.dfs_counter = 0
        self.dfs_queue = [self.node_seed]

    def _change_current(self):  # полностью переопределяем метод
        while (self.dfs_queue[0] not in self.v_observed):
            self.dfs_queue.pop(0)
        self.current = self.dfs_queue[0]

    def _observing(self):
        self.dfs_counter = 0
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id, self.current)
                self.dfs_counter += 1
                self.dfs_queue.insert(self.dfs_counter, friend_id)
        if self.current in self.v_observed:
            self.dfs_queue.remove(self.current)


class Crawler_BFS(Crawler):
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)
        self.method = 'BFS'
        self.bfs_queue = [self.node_seed]

    def _change_current(self):  # полностью переопределяем метод
        while (self.bfs_queue[0] not in self.v_observed):
            self.bfs_queue.pop(0)
        self.current = self.bfs_queue[0]

    def _observing(self):
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id, self.current)
                self.bfs_queue.append(friend_id)
        if self.current in self.bfs_queue:
            self.bfs_queue.remove(self.current)


class Crawler_MOD(Crawler):
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)
        self.method = 'MOD'
        self.graph_for_max_deg = self.G

    def _change_current(self):
        maximal, max_id = 0, random.sample(self.v_observed, 1)[0]  # берём случайную вершину за основу
        for i in self.v_observed:
            if maximal < self.graph_for_max_deg.degree(i):
                maximal, max_id = self.graph_for_max_deg.degree(i), i
        self.current = max_id


class Crawler_MED(Crawler_MOD):  # унаследовали от MOD только степени берём из большого графа
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)
        self.method = 'MED'
        self.graph_for_max_deg = self.big_graph


class Crawler_RW(Crawler):
    def __init__(self, big_graph, node_seed, budget, percentile_set):
        Crawler.__init__(self, big_graph, node_seed, budget, percentile_set)
        self.method = 'RW'
        self.previous = self.current

    def sampling_process(self):
        self.previous = self.node_seed
        super().sampling_process()
        self.previous = self.current

    def _change_current(self):
        # print(self.previous)
        new_node = random.sample(set(self.big_graph.adj[self.previous]), 1)[0]
        while new_node in self.v_closed:
            self.previous = new_node
            new_node = random.sample(set(self.big_graph.adj[self.previous]), 1)[0]
            # print('whiling in ',new_node,'and his friends',big_graph.adj[previous])
        self.current = new_node


