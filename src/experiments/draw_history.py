import json
import os

from matplotlib import pyplot as plt

from utils import CENTRALITIES

print(os.getcwd())

graph_name = 'petster-hamster'
crawlers_names = ['DFS', 'MOD', 'RC_', 'DFS', 'BFS', 'FFC']
total_budget = 2000

centrality_history = {centrality_name: {crawler_name: {}
                                        for crawler_name in crawlers_names}
                      for centrality_name in CENTRALITIES}
color = ['blue', 'black', 'green', 'red', 'orange', 'cyan', 'pink', 'yellow']
fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex=True,
                        sharey=True)  # TODO need to be changed if CENTRALITIES changes
for centrality_name in CENTRALITIES:
    color_iter = 0
    plot_x, plot_y = CENTRALITIES.index(centrality_name) % 3, int(CENTRALITIES.index(centrality_name) / 3)
    for crawler_name in crawlers_names:
        path = "../data/crawler_history/observed_history_{}_{}.json".format(crawler_name, centrality_name)
        with open(path) as top_set_file:
            centrality_history[centrality_name][crawler_name] = json.load(top_set_file)
            x, y = zip(*centrality_history[centrality_name][crawler_name])
            axs[plot_x, plot_y].plot(x, y, color=color[color_iter], label=crawler_name)
        color_iter += 1 % 8
    axs[plot_x, plot_y].set(title=centrality_name)
    axs[plot_x, plot_y].grid(True)
    axs[plot_x, plot_y].set_xlim(0, total_budget)

plt.xlabel('crawl iterations')
plt.ylabel('crawled+observed percent')
plt.title(graph_name)
plt.legend()
plt.savefig('/home/jzargo/PycharmProjects/crawling/crawling/results/{}_centralities_history.png'.format(graph_name))
plt.show()
