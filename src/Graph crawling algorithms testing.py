#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import vk
import json
from networkx.algorithms import community
from matplotlib.pyplot import figure
session = vk.Session(access_token='4c7c952b4c7c952b4c7c952bf14c1a736544c7c4c7c952b179d212fd48229afe166e816')
vk_api = vk.API(session)
options = {   # опции для отрисовки графа
  #  'node_color': 'black',
    'node_size': 200,
    'width': 1,
   # 'font_weight': 'bold',
    'with_labels':True
}


# In[123]:


# маленький бот, который умеет печатать в личку отчёт из print
python_bot_session = vk.Session(access_token='a8dede3a2d1a42fa9bc495db9b437ca55671824c985d1faa77371943a6d36b5508021a74cadc36b645cae')
python_bot = vk.API(python_bot_session)
# функция, которая печатает и на экран и в указанную

attachments = []
# по умолчанию мой idшник вк(Денис)
def myprint(*message_array):
    #message = [i for i in message.replace(',', ' ').split()]
    message = [i for i in message_array.replace(',', ' ').split()]
    
    peer_id = '110014788'
    attachment = ''
    # TBD: разобраться с Attachments
    print(message)
    
    return python_bot.messages.send(v = '5.89',user_id = peer_id, message = message, attachment = ','.join(attachments))


myprint('Hi', 'Hello', options)


# In[94]:


# Работа с графами. Выбирается один из списка (всё определяет graph_name и *_graph) 

#fh=open("importing.txt", 'rb')
t0 = time.time() 
#первое слово выбрать из списка: importing   dblp2010   github   wikivote   slashdot
graph_name_list = ['importing',  'dblp2010',  'github',' wikivote','  slashdot']
graph_list = []
graph_name = "wikivote"  
fh=open(graph_name+ ".edges", 'rb')

Graph=nx.read_edgelist(fh, delimiter = ' ')

graph_components = [Graph.subgraph(c) for c in nx.connected_components(Graph)]
print('components lenght', [len(i) for i in graph_components])


fh.close()
print(list(nx.selfloop_edges(Graph)))
Graph.remove_edges_from(nx.selfloop_edges(Graph))
print('Nodes: ',Graph.number_of_nodes(),' edges',Graph.number_of_edges())
#dblp201_graph = Graph
wikivote_graph = Graph
#slashdot_graph = Graph
#github_graph = Graph
#nx.draw(Graph, with_labels=False ,edge_cmap=plt.cm.Blues , node_size = 30)
print(int(time.time() -t0) , 'sec')




# In[132]:


### Функция, выдающая всевозможные параметры графа в виде словаря. Прогоняем по списку графов и делаем табличку

def graph_properties(G):
    t0 = time.time()
    print('number_of_nodes',G.number_of_nodes(),
            'number_of_edges',G.number_of_edges())
    
    # степени всех вершин и их распределение
    degrees = dict(G.degree())
    degrees_hist = [degrees[i] for i in degrees]
    degrees_hist.sort(reverse = True)
    print('Распределение степеней',degrees_hist[:100])
    d_avg = sum(degrees_hist)/G.number_of_nodes()
    print('d_avg=',d_avg, '  max_degree = ',degrees_hist[0])
    plt.hist(degrees_hist,max(degrees_hist))
    plt.title('Распределение степеней вершин')
    plt.show()
    
    
    print('time passed',time.time()-t0)
    t0 = time.time()
    
    
    # k-ядерность всех вершин и их гистограмма распределения  (сложность - O(n+m))
    print('Распределение k-ядерности всех вешин')
    k_cores = nx.algorithms.core.core_number(G)
    k_cores_hist = [k_cores[i] for i in k_cores]
    k_cores_hist.sort(reverse = True)
    print('k-ядерность вершин',k_cores_hist[:100])
    plt.hist(k_cores_hist,max(k_cores_hist))
    plt.title('k-cores')
    plt.show()
    
    
    print('time passed',time.time()-t0)
    t0 = time.time()
    

    ### Эксцентриситет - самое максимальное кратчайшее расстояние до вершины в графе (сложность О(nm) )
    eccentricity = 0
    if nx.is_connected(G):
        eccentricity = nx.algorithms.distance_measures.eccentricity(G)
        eccentricity_hist = [eccentricity[i] for i in eccentricity]
        eccentricity_hist.sort(reverse = True)
        print('eccentricity',eccentricity_hist[:100])
        plt.hist(eccentricity_hist,max(eccentricity_hist))
        plt.title('Eccentricity')
        plt.show()
    
    
    
    print('time passed',time.time()-t0)
    t0 = time.time()
    
    
    
    return {'number_of_nodes':G.number_of_nodes(),
            'number_of_edges':G.number_of_edges(),
            'degrees':degrees,
            'd_avg':d_avg,
            'k_cores':k_cores,
            'eccentricity': eccentricity,
            # clustering coeff
            #betweeness
           }
    
    
print(graph_properties(wikivote_graph))


#github_proeerties = graph_properties(github_graph)

#print(nx.k_nearest_neighbors(Graph,'2'))
#print(nx.algorithms.assortativity.degree_assortativity_coefficient(Graph))

#nx.draw(Graph, with_labels=True ) # раскоментить, если граф не убийственно большой. Это попытка его нарисовать


# In[ ]:


nx.networkx.algorithms.communicability_alg.communicability(wikivote_graph)


# In[43]:


# Определение числа компонент связности, берём самую большую. И ещё сохраняем граф

t0 = time.time() 
print(Graph.number_of_nodes())
print('is connected',nx.is_connected(Graph))

print('component count', nx.number_connected_components(Graph))
components = [Graph.subgraph(c) for c in nx.connected_components(Graph)]
almost_empty_count = 0
budget_graph = components[0]
subgraph_nodes = []
for subgraph in components:
    subgraph_nodes.append(subgraph.number_of_nodes())
    if subgraph.number_of_nodes() == 1:
        almost_empty_count +=1
print(subgraph_nodes.sort())
print('almost empty',almost_empty_count)
print( len(Graph.nodes))

#budget_set = set()
#for i in budget_graph.nodes():
#    budget_set.add(i)
total = max([Graph.subgraph(c).number_of_nodes() for c in nx.connected_components(Graph)])
print(total, 'total')


# сразу экспортируем граф
nx.write_gml(Graph, "./Sampling/"+ graph_name +'_BIG.graph')
print('exported in'+"./Sampling/"+ graph_name +'_BIG.graph')


nx.draw(Graph, with_labels=False ,edge_cmap=plt.cm.Blues , node_size = 10)
#plt.savefig('./Sampling/Graph '+filename+ '.png')
plt.show()
print(int(time.time() -t0) , 'sec')


# # Общая функция получения истории observed вершин

# In[102]:


# описания всех нужных нам функций
def max_deg(G):
    maximal = 0
    max_id = random.sample(v_observed,1)[0] # берём случайную вершину за основу
    for i in v_observed:
        if maximal<G.degree(i):
            maximal = G.degree(i)
            max_id = i
            #print(max_id, maximal, end = ' ')
    return max_id#,maximal]

def observing(node,big_graph,G):  # собираем друзей просматриваемой вершины, всё 
    friends = list(big_graph.adj[node]) 
    dfs_counter = 0
    for friend_id in friends:
        if friend_id not in v_closed:
            v_observed.add(friend_id)
            G.add_node(friend_id)
            G.add_edge(friend_id,node)
            bfs_queue.append(friend_id)  # очередь для поиска в ширину - добавляем друзей в конец 
            dfs_counter +=1
            dfs_queue.insert(dfs_counter,friend_id) # очередь для поиска в глубину - добавляем друзей поближе
    if node in dfs_queue:
        dfs_queue.remove(node)
    if node in bfs_queue:
        bfs_queue.remove(node)
    if node in v_observed: # вообще такого не должно быть
        v_observed.remove(node)
    node_array.append(node)
    v_closed.add(node)  
    friends_of_node = set(big_graph.adj[node])
    #print('observing', len(v_observed), len(v_closed))
    
    # TBD: надо пересмотреть всех друзей этой вершины и накинуть им в степени ещё по одной за родство с просматриваемой
    # и добавить те вершины, которых ещё не было


# на вход подаём граф, начальный элемент n_seed, бюджет запросов b и метод, которым прогоняем
def graph_sampling(big_graph, n_seed, b, method = 'MOD'): 
    counter = 0
    G = nx.Graph() 
    G.add_node(node_seed)
    previous = node_seed
    while (counter< total)and(len(v_observed)>0)and(counter <b):
        # в зависимости от метода мы выбираем вершину разными способами
        if (method in ('Random Walk','RW')): 
            try:
                new_node = random.sample(set(big_graph.adj[previous]),1)[0]
                while new_node in v_closed:
                    previous = new_node
                    new_node = random.sample(set(big_graph.adj[previous]),1)[0]
                    #print('whiling in ',new_node,'and his friends',big_graph.adj[previous])
                current = new_node
            except BaseException:
                print('error with friends of', previous)
            
            #print('current is', current)
        elif method in ('Rand','Random Crawling','RC'):
            current = random.sample(v_observed,1)[0]
        elif method in ('Maximum Excess Degree', 'MED'):
            current = max_deg(big_graph)
        elif method in ('Maximal Observed Degree','MOD' ): #  по умолчанию прогоняем в Maximal Observed Degree (MOD)
            current = max_deg(G)
        elif method in ('Depth First Search','DFS'):
            #print('DFS queue', dfs_queue)
            while (dfs_queue[0] not in v_observed):
                dfs_queue.pop(0)
            current = dfs_queue[0]
                
        elif method in ('Breadth First Search', 'BFS'):
            #print('BFS queue', bfs_queue)
            while (bfs_queue[0] not in v_observed):
                bfs_queue.pop(0)
            current = bfs_queue[0]
        
        #print('count',counter,'observed',len(v_observed),'closed', len(v_closed),'current' ,current)
        observing(current,big_graph,G)        
        observed_history.append( len(v_observed)+ len(v_closed))
        counter+=1
        #plt.title(method + 'running ' + str(current)+ 'observed:'+ str(v_observed))
        #nx.draw_circular(G, with_labels=True ,edge_cmap=plt.cm.Blues , node_size = 300)
        #plt.title(method + 'running ' + str(current)+ 'observed:'+ str(v_observed))
        #plt.savefig('./Sampling/'+str(method)+'/Node'+str(node_seed) +' counter:'+str(counter) +' seeds.png')  
        #plt.show()
        previous = current
    # делаем бэкап графа 
    nx.write_gml(G, "./Sampling/"+ graph_name +"/"+method+'_b='+str(b)+'_seed='+str(n_seed)+'.graph')
    
    return observed_history


# In[122]:


# Вот тут выбираем граф, и дальше он прогоняется по всем методам из methods и по всем начальным вершинам из seeds
#  ... каждую из таких итераций делается по b запросов. Если граф не большой, b = total т.е. всего вершин в графе
# Сложность  = len(seeds)*len(methods)*b (для MOD/MED * b^2 вместо *b)

# надо выбрать оба этих поля
Graph = wikivote_graph#dblp2010mtx_graph#github_graph
graph_name = 'wikivote' #'wikivote', 'github', 'DCAM'



t00 = time.time()
linestiles = [ ':', '--', '-.','--', '-.', ':','-','-', '-', '--', '-.', ':','-', '--', '-.', ':']

##  TBD:  ВОТ ОТСЮДА НАДО УБРАТЬ РАЗМЕРЫ ГРАФА ВНУТРЬ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

total = len(Graph)  # общее количество 
b = total#1000  # число запросов не больше чем total
all_history = []

plt.figure(figsize=(15,15))

#plt.plot([total]*b) # верхняя линия  # TBD: верхняя линия размером с самую большую компоненту
plt.grid(True)


seed_count = 2
seeds = random.sample(set(Graph.nodes),seed_count) # список начальных вершин, по которым мы будем проходиться
methods = ['RW','DFS', 'BFS', 'RC','MED','MOD']
method_history = np.zeros([len(methods),total+1])



method_counter = 0
for method in methods:
    real_seed_count = seed_count #если у нас сиды - аутисты, чтобы не делить на их количество
    seed_counter = 0
    myprint('Running method ' + method)
    for node_seed in seeds:
        t0 = time.time()
        # сама программа по запуску алгоритма сэмплирования
        
        v_observed = set() # 
        v_observed.add(node_seed)  
        v_closed = set()  # множество уже обработанных вершин
        node_degree = [] # общее число друзей вершины. А не в бюджетном
        node_array = []  # список нод в порядке обработки
        dfs_queue = [node_seed]
        bfs_queue = [node_seed]
        observed_history = [0]  # история изменения числа видимых вершине
        history = graph_sampling(Graph,node_seed, b, method)
        all_history.append(history)

        len_v = len(v_observed)+ len(v_closed)
        last_x = min(b,len_v)
        #print([(i,int(history[i]-history[i-1])) for i in range(1,last_x)])
        if (len(v_observed)+len(v_closed)<b):
            real_seed_count -=1
        
        #plt.plot([len_v for i in range(1,last_x)])
        if method == 'RW':
            color = 'green'
        elif method == 'MOD':
            color = 'red'  
        elif method == 'RC':
            color = 'grey'
        elif method == 'DFS':
            color = 'black'
        elif method == 'BFS':
            color = 'blue'
        elif method == 'MED':
            color = 'cyan'
        #"Количество разведанных вершин от числа итераций"
        # отрисовка для каждого стартового сида, всего будет много графиков
        plt.plot(history[0:b],color = color,linewidth = 1,linestyle = linestiles[seed_counter])
                                    #, label = method + ' id' + str(node_seed)+' nodes: '+str(history[-1])
        # подгоняем размер и суммируем по всем сидам для каждого метода. Усреднение
        history_remap = history
        for i in range(len(history),total+1):
            history_remap.append(history_remap[i-1])
        #print(len(history), len(method_history[method_counter]), total )
        method_history[method_counter] += np.array(history_remap)    
        seed_counter +=1
        myprint(method + ' seed=' + str(node_seed) + " total=",total, " Observed",len(v_observed) , " Closed",len(v_closed),' Time:',round(time.time() -t0,2) , 'sec')
    
    
    
    plt.plot(method_history[method_counter][:b]/real_seed_count, linewidth = 5, color = color, label = method)
    method_counter+=1   
    #plt.savefig('./Sampling/'+ filename+'/'+ str(seed_count)+ '_seeds' + str(b)+ '.png')  
    
    plt.savefig('./Sampling/'+graph_name+'/'+ str(seed_count)+ '_seeds' + str(b)+ '.png')  
    
    plt.legend()
plt.show()
myprint('Total time:',int(time.time() -t00) , 'sec')
#['184501','143222','191631','52146']


# ## Здесь заканчивается сама прога и начинаются эээксперименты

# In[15]:




def draw_observed_history(history):
len_v = len(v_observed)+ len(v_closed)
last_x = min(len_v)
#print([(i,int(history[i]-history[i-1])) for i in range(1,last_x)])
print("всего вершин",len(total), "observed",len(v_observed) , "closed",len(v_closed) )

print("Количество разведанных вершин от числа итераций")
plt.plot([len_v for i in range(1,last_x)])
plt.plot(history[:last_x])
plt.show()
# график производной
print("производная предыдущего графика (т.е. сколько мы разведываем вершин после обработки каждой)")
plt.bar(range(1,last_x),[history[i]-history[i-1] for i in range(1,last_x)])

plt.show()


# отображаем скорости роста для каждого метода поотдельности. TBD: он сломался и надо починить
for method_counter in range(len(methods)):
plt.plot([(method_history[method_counter][i]-method_history[method_counter][i-1])/seed_count for i in range(2,total)], color = list(['blue','red','green','black'])[method_counter], label = method)
plt.show()
print(node_array)


# In[40]:



#nx.write_gml(Graph, "test_graph")
# тестирование экспортированных графов на адекватность
import os
for root, dirs, files in os.walk("./Sampling/importing/"):
    for file in files:
        
        if file.endswith(".graph"):
            print(os.path.join(root, file))
            new = nx.read_gml("test_graph")
            #nx.draw(new, with_labels=True ,node_size = 100)
            #plt.show()
            


# In[201]:


# жадно разбиваем граф на сообщества и выводим по ним всю информацию
Graph = budget_graph
c = list(community.greedy_modularity_communities(Graph))
print(c)
community_counter = len(c)
print('community counter',community_counter)
community_size_list= [len(i) for i in c]
print('community sizes',community_size_list)
community_size_avg = sum([len(i) for i in c])/len(c)
print('Average community size ',community_size_avg)
print('coverage',community.coverage(Graph,c))
print('performance',community.performance(Graph,c))

#Graph.remove_edges_from(nx.selfloop_edges(Graph))
#nx.algorithms.core.core_number(Graph)#[110014788]
total_nodes = Graph.number_of_nodes()
total_edges = Graph.number_of_edges()
d_avg = sum([Graph.degree(I) for I in Graph])/Graph.number_of_nodes()



print(d_avg)


# In[294]:


# эксперименты. можно смело удалять 
K_5 = nx.complete_graph(5)
K_3_5 = nx.complete_bipartite_graph(3, 5)
barbell = nx.barbell_graph(10, 10)
lollipop = nx.lollipop_graph(10, 20)
graphs = [K_5, K_3_5, barbell, lollipop]
for G1 in graphs:
    nx.draw(G1, with_labels=True, font_weight='bold')
    plt.show()
    d = list(community.greedy_modularity_communities(G1))
    print(d)
    community_counter = len(d)
    print('community counter',community_counter)
    community_size_list= [len(i) for i in d]
    print('community sizes',community_size_list)
    community_size_avg = sum([len(i) for i in d])/len(d)
    print('Average community size ',community_size_avg)
    print('coverage',community.coverage(G1,d),'штука похожая на мю - community mixing')
    print('performance',community.performance(G1,d))


# In[51]:


#import tempfile
with open('mygraph.txt', 'w') as f:
    _ = f.write(b'>>graph6<<A_\n')
    _ = f.seek(0)
    G = nx.read_graph6(f)
list(G.edges())


# In[ ]:





# In[ ]:


Running method RW
RW 87645 всего вершин 120867 observed 57423 closed 10000
RW  n_seed 87645 151 sec
RW 25181 всего вершин 120867 observed 57676 closed 10000
RW  n_seed 25181 145 sec
RW 88944 всего вершин 120867 observed 56734 closed 10000
RW  n_seed 88944 145 sec
RW 90453 всего вершин 120867 observed 57262 closed 10000
RW  n_seed 90453 147 sec
RW 45384 всего вершин 120867 observed 57336 closed 10000
RW  n_seed 45384 146 sec
RW 24934 всего вершин 120867 observed 57627 closed 10000
RW  n_seed 24934 142 sec
RW 1992 всего вершин 120867 observed 57500 closed 10000
RW  n_seed 1992 143 sec
RW 23083 всего вершин 120867 observed 57385 closed 10000
RW  n_seed 23083 144 sec
RW 113741 всего вершин 120867 observed 56920 closed 10000
RW  n_seed 113741 144 sec
RW 32947 всего вершин 120867 observed 57174 closed 10000
RW  n_seed 32947 139 sec
Running method DFS
DFS 87645 всего вершин 120867 observed 62383 closed 10000
DFS  n_seed 87645 143 sec
DFS 25181 всего вершин 120867 observed 62113 closed 10000
DFS  n_seed 25181 145 sec
DFS 88944 всего вершин 120867 observed 62403 closed 10000
DFS  n_seed 88944 145 sec
DFS 90453 всего вершин 120867 observed 62404 closed 10000
DFS  n_seed 90453 144 sec
DFS 45384 всего вершин 120867 observed 62463 closed 10000
DFS  n_seed 45384 144 sec
DFS 24934 всего вершин 120867 observed 61720 closed 10000
DFS  n_seed 24934 146 sec
DFS 1992 всего вершин 120867 observed 62315 closed 10000
DFS  n_seed 1992 146 sec
DFS 23083 всего вершин 120867 observed 62139 closed 10000
DFS  n_seed 23083 147 sec
DFS 113741 всего вершин 120867 observed 62250 closed 10000
DFS  n_seed 113741 146 sec
DFS 32947 всего вершин 120867 observed 62181 closed 10000
DFS  n_seed 32947 144 sec

home(/denisaivazov/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:93:, UserWarning:, Creating, legend, with, loc="best", can, be, slow, with, large, amounts, of, data.)

Running method BFS
BFS 87645 всего вершин 120867 observed 57490 closed 10000
BFS  n_seed 87645 98 sec
BFS 25181 всего вершин 120867 observed 56827 closed 10000
BFS  n_seed 25181 124 sec
BFS 88944 всего вершин 120867 observed 56770 closed 10000
BFS  n_seed 88944 116 sec
BFS 90453 всего вершин 120867 observed 57763 closed 10000
BFS  n_seed 90453 101 sec
BFS 45384 всего вершин 120867 observed 54603 closed 10000
BFS  n_seed 45384 110 sec
BFS 24934 всего вершин 120867 observed 56393 closed 10000
BFS  n_seed 24934 116 sec
BFS 1992 всего вершин 120867 observed 57733 closed 10000
BFS  n_seed 1992 104 sec
BFS 23083 всего вершин 120867 observed 55168 closed 10000
BFS  n_seed 23083 99 sec
BFS 113741 всего вершин 120867 observed 57347 closed 10000
BFS  n_seed 113741 105 sec
BFS 32947 всего вершин 120867 observed 56282 closed 10000
BFS  n_seed 32947 104 sec
Running method RC
RC 87645 всего вершин 120867 observed 39610 closed 10000
RC  n_seed 87645 124 sec
RC 25181 всего вершин 120867 observed 38972 closed 10000
RC  n_seed 25181 124 sec
RC 88944 всего вершин 120867 observed 39496 closed 10000
RC  n_seed 88944 130 sec
RC 90453 всего вершин 120867 observed 40037 closed 10000
RC  n_seed 90453 129 sec
RC 45384 всего вершин 120867 observed 38912 closed 10000
RC  n_seed 45384 124 sec
RC 24934 всего вершин 120867 observed 40852 closed 10000
RC  n_seed 24934 128 sec
RC 1992 всего вершин 120867 observed 39466 closed 10000
RC  n_seed 1992 131 sec
RC 23083 всего вершин 120867 observed 38827 closed 10000
RC  n_seed 23083 127 sec
RC 113741 всего вершин 120867 observed 40525 closed 10000
RC  n_seed 113741 127 sec
RC 32947 всего вершин 120867 observed 39415 closed 10000
RC  n_seed 32947 125 sec
Running method MED
MED 87645 всего вершин 120867 observed 72761 closed 10000
MED  n_seed 87645 1115 sec
MED 25181 всего вершин 120867 observed 72760 closed 10000
MED  n_seed 25181 1121 sec
MED 88944 всего вершин 120867 observed 72762 closed 10000
MED  n_seed 88944 1112 sec
MED 90453 всего вершин 120867 observed 72771 closed 10000
MED  n_seed 90453 1113 sec
MED 45384 всего вершин 120867 observed 72763 closed 10000
MED  n_seed 45384 1142 sec
MED 24934 всего вершин 120867 observed 72761 closed 10000
MED  n_seed 24934 1104 sec
MED 1992 всего вершин 120867 observed 72762 closed 10000
MED  n_seed 1992 1092 sec
MED 23083 всего вершин 120867 observed 72765 closed 10000
MED  n_seed 23083 1093 sec
MED 113741 всего вершин 120867 observed 72760 closed 10000
MED  n_seed 113741 1093 sec
MED 32947 всего вершин 120867 observed 72761 closed 10000
MED  n_seed 32947 1093 sec
Running method MOD
MOD 87645 всего вершин 120867 observed 66117 closed 10000
MOD  n_seed 87645 957 sec
MOD 25181 всего вершин 120867 observed 66132 closed 10000
MOD  n_seed 25181 954 sec
MOD 88944 всего вершин 120867 observed 66146 closed 10000
MOD  n_seed 88944 951 sec
MOD 90453 всего вершин 120867 observed 66142 closed 10000
MOD  n_seed 90453 968 sec
MOD 45384 всего вершин 120867 observed 66150 closed 10000
MOD  n_seed 45384 951 sec
MOD 24934 всего вершин 120867 observed 66117 closed 10000
MOD  n_seed 24934 961 sec
MOD 1992 всего вершин 120867 observed 66151 closed 10000
MOD  n_seed 1992 957 sec
MOD 23083 всего вершин 120867 observed 66141 closed 10000
MOD  n_seed 23083 952 sec
MOD 113741 всего вершин 120867 observed 66139 closed 10000
MOD  n_seed 113741 957 sec
MOD 32947 всего вершин 120867 observed 66167 closed 10000
MOD  n_seed 32947 959 sec


# In[1]:


# создаём бюджетное множество - группа ФУПМ МФТИ. Всё, что вне её, считается несуществуюшим.
# группа фупм мфти id 1694    группа ораторки 173319156

group_id = "1694"
offset = 1000 
members = vk_api.groups.getMembers(v=5.89, group_id = group_id) 
budget_err_count = 0
DCAM_graph = nx.Graph()  # делаем большой граф из группы ФУПМ МФТИ
while True:  # т.к. вк ограничивает выгрузку за раз 1000, докидываем со сдвигом offset.
    resp = vk_api.groups.getMembers(v=5.89, group_id = group_id, offset = offset)#["response"]
    #print(resp)
    members["items"] += resp["items"]
    offset += 1000
    #print (offset)
    if offset > resp["count"] :
        break
        
budget_set = set()
for id in members["items"]:
    #print(id)
    try:
        bugs = vk_api.users.get(v=5.89, user_id=id, fields = 'deactivated,is_closed')
        if (u'deactivated' in bugs[0])or(bugs[0]['is_closed']):
            continue
        budget_set.add(id)
        DCAM_graph.add_node(id)
    except BaseException:
        budget_err_count+=1 
        
        
#budget_set = set(members['items'])
#print('budget errors', budget_err_count)
#print(len(budget_set), budget_set)

for node in DCAM_graph:
    #print('n',node)
    friends = vk_api.friends.get(v=5.52, user_id=node)['items' ]
    for friend in friends:
        #print('f',friend, type(friend))
        if friend in budget_set:
            DCAM_graph.add_edge(node,friend)
nx.write_gml(DCAM_graph, "./DCAM_BIG.graph")

