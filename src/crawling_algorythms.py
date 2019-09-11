#!/usr/bin/env python
# coding: utf-8

# In[90]:

import random
import networkx as nx


class Crawler:
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        
        self.big_graph = Graph # сам конечный граф
        self.total = self.big_graph.number_of_nodes() # total это общее число вершин
        self.b = min(budget,int(self.total*0.99)) # бюджет запросов, которые можно сделать
        self.node_seed = node_seed # начальная сида, с которой начинаем
        self.percentile_set = percentile_set # множество вершин графа, удовлетворяющих квантилю
        self.method = 'RC'
        
        self.current = node_seed  # в начале текущая вершина это стартовая нода
        self.v_observed = set()   # множество увиденных вершин
        self.v_observed.add(node_seed)  
        self.v_closed = set()  # множество уже обработанных вершин
        self.node_degree = [] # общее число друзей вершины. А не в бюджетном
        self.node_array = []  # последовательность вершин, которые мы обходим
        self.G = nx.Graph()         # наш насемплированный граф
        self.G.add_node(node_seed)  # добавляем начальную вершину
        
        self.metrics_list = metrics_list # список названий полей из properties графа, которые исследуем
        self.counter = 0            # счётчик итераций
        self.observed_history = [0] # история изменения количества видимых вершин
## вот тут мб надо немного переделать
        self.property_history = dict((i,[]*(self.total+1)) for i in metrics_list) # история изменения параметров с итерациями
        for prop in self.metrics_list:
            self.property_history[prop].append(len(self.v_closed.intersection(self.percentile_set[prop])))
            
        self.node_degree = [] # общее число друзей вершины. А не в бюджетном
        self.node_array = []  # список вершин в порядке обработки
        
        #self.plot, _ = plt.subplots()
        #print(type())
        
    

    def _observing(self):
        """
        пробегаемся по всем друзьям и добавляем их в наш граф и в v_observed
        """
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id,self.current)
            
        
    def _change_current(self): # выбор новой вершины. по умолчанию считаем random crawling
        raise NotImplementedError()
    
    def draw(self):
        #print(self.observed_history)
        pass
        return plt.plot(self.observed_history)
        
    def sampling_process(self): # сам  процесс сэмплирования
        
        #print(self.current, self.v_observed, self.v_closed)
        self._change_current()
        self._observing()
        if self.current in self.v_observed: # условие для пустых итераций рандом волка tbd - надо исправить
            self.v_observed.remove(self.current) # теперь она обработанная, а не просто видимая
        self.v_closed.add(self.current)
        self.node_array.append(self.current) # отметили, что обработали эту вершину
        self.observed_history.append(len(self.v_observed)+len(self.v_closed))

        for prop in self.metrics_list:
            self.property_history[prop].append(len(self.v_closed.intersection(self.percentile_set[prop])))
            #print(self.property_history,prop,self.counter)
            #self.property_history[prop][self.counter]+=len(self.v_closed.intersection(self.percentile_set[prop]))      
        #self.draw()
        return self.observed_history#,self.property_history)
    # раскомментить если захочется экспортировать граф
    #nx.write_gml(G, "./Sampling/"+ graph_name +"/"+method+'_b='+str(b)+'_seed='+str(n_seed)+'.graph')

    

class Crawler_RC(Crawler):
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
    def _change_current(self):
        self.current = random.sample(self.v_observed,1)[0]

        
        
class Crawler_DFS(Crawler):
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
        self.method = 'DFS'
        self.dfs_counter = 0
        self.dfs_queue = [self.node_seed]
    
    def _change_current(self): # полностью переопределяем метод
        while (self.dfs_queue[0] not in self.v_observed):
            self.dfs_queue.pop(0)
        self.current = self.dfs_queue[0] 
         
    def _observing(self):
        self.dfs_counter = 0
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id,self.current)
                self.dfs_counter +=1
                self.dfs_queue.insert(self.dfs_counter,friend_id)
        if self.current in self.v_observed:
            self.dfs_queue.remove(self.current)
        

        
class Crawler_BFS(Crawler):
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
        self.method = 'BFS'
        self.bfs_queue = [self.node_seed]
    
    def _change_current(self): # полностью переопределяем метод
        while (self.bfs_queue[0] not in self.v_observed):
            self.bfs_queue.pop(0)
        self.current = self.bfs_queue[0] 

    def _observing(self):
        for friend_id in list(self.big_graph.adj[self.current]):
            if friend_id not in self.v_closed:
                self.v_observed.add(friend_id)
                self.G.add_node(friend_id)
                self.G.add_edge(friend_id,self.current)
                self.bfs_queue.append(friend_id)
        if self.current in self.bfs_queue:
            self.bfs_queue.remove(self.current)       
        
        
          
class Crawler_MOD(Crawler):
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
        self.method = 'MOD'
        self.graph_for_max_deg = self.G
        
    def _change_current(self):
        maximal,max_id = 0, random.sample(self.v_observed,1)[0] # берём случайную вершину за основу
        for i in self.v_observed:
            if maximal<self.graph_for_max_deg.degree(i):
                maximal,max_id = self.graph_for_max_deg.degree(i),i
        self.current = max_id


        
class Crawler_MED(Crawler_MOD): # унаследовали от MOD только степени берём из большого графа
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
        self.method = 'MED'
        self.graph_for_max_deg = self.big_graph

    

class Crawler_RW(Crawler):
    def __init__(self,Graph,node_seed,budget,percentile_set,metrics_list):
        Crawler.__init__(self,Graph,node_seed,budget,percentile_set,metrics_list)
        self.method = 'RW'
        self.previous = self.current
        
    def sampling_process(self):
        self.previous = self.node_seed
        super().sampling_process()
        self.previous = self.current
    
    def _change_current(self):
        #print(self.previous)
        new_node = random.sample(set(self.big_graph.adj[self.previous]),1)[0]
        while new_node in self.v_closed:
            self.previous = new_node
            new_node = random.sample(set(self.big_graph.adj[self.previous]),1)[0]
            #print('whiling in ',new_node,'and his friends',big_graph.adj[previous])
        self.current = new_node
            
    

