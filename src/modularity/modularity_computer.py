from graph_io import GraphCollections
from statistics import Stat
from utils import USE_NETWORKIT

if USE_NETWORKIT:
    from networkit.community import PLM, Modularity


def test(g):
    print(g['NODES'], g['EDGES'], g[Stat.AVG_DEGREE])
    m = g['EDGES']
    nk = g.networkit()
    plm = PLM(nk, refine=False, gamma=1)
    plm.run()
    partition = plm.getPartition()
    # for p in partition:
    len_num = {}
    exp_q = 0
    for i in range(partition.numberOfSubsets()):
        comm = partition.getMembers(i)
        for i in comm:
            exp_q += (g.deg(i)) ** 2

        # print(len(comm), comm)
        size = len(comm)
        if size not in len_num:
            len_num[size] = 0
        len_num[size] += 1
        if size > 1:
            print(size, list(comm)[:10])

    print(exp_q, 4*m*m)
    print(len_num)
    print('total', sum(k * v for k, v in len_num.items()))
    # print(partition.getMembers(0))
    # print(partition.getMembers(1))
    # print(partition.getMembers(2))
    mod = Modularity().getQuality(partition, nk)
    print(mod)


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    # a = [1003, 924, 930, 1674, 720, 342, 1482, 804, 252, 708, 876, 1410, 1374, 1119, 1104, 1392, 351, 939]
    # print(sum(a))

    # name = 'petster-hamster'
    # name = 'digg-friends'
    # name = 'Pokec'
    # g = GraphCollections.get(name, 'konect')

    # name = 'uk-2005'  # netrepo
    # name = 'github'  # netrepo
    # name = 'sc-shipsec5'  # netrepo  Q=0.8995
    # name = 'socfb-Bingham82'  # netrepo  Q=0.4597
    # name = 'tech-p2p-gnutella'  # netrepo  Q=0.5022
    # name = 'karate'  # netrepo
    # name = 'rec-amazon'  # netrepo Q=0.9898
    name = 'soc-slashdot'  # netrepo Q=0.362
    name = 'socfb-Bingham82'  # netrepo Q=0.4619
    name = 'ca-MathSciNet'  # netrepo Q=

    g = GraphCollections.get(name, 'netrepo')

    test(g)

