import numpy as np
from numpy import random
import scipy.stats as stats


# def power_law_generator(count, gamma, b, a=0):
#     """
#     Generator of power law values, y ~ x^{gamma-1} for a<=x<=b
#     :param gamma: power
#     :param b: max value
#     :param a: min value
#     :return:
#     """
#     from scipy.stats import powerlaw
#     # powerlaw.rvs()
#     r = powerlaw.rvs(a=a, size=1000)
#     for x in r:
#         yield x
#     # ag, bg = a ** gamma, b ** gamma
#     # print("iter")
#     # for _ in range(count):
#     #     r = random.random()
#     #     r = random.random()
#     #     yield (ag + (bg - ag) * r) ** (1. / gamma)


def truncated_power_law(count, gamma, maximal):
    """
    Generate power law integer samples y ~ x^{-gamma} for 1<=x<=maximal.
    :param count: number of samples
    :param gamma: power
    :param maximal: max value
    :return:
    """
    if maximal >= count:
        maximal = count-1
    x = np.arange(1, maximal + 1, dtype='float')
    pmf = 1 / x ** gamma
    pmf /= pmf.sum()
    d = stats.rv_discrete(values=(range(1, maximal + 1), pmf))
    return d.rvs(size=count)


def truncated_normal(count, mean, variance, min, max):
    """
    Generate normal integer samples for min<=x<=maximal.
    :param count: number of samples
    :param mean:
    :param variance:
    :param min: min value
    :param max: max value
    :return:
    """
    assert min < max
    x = np.arange(min, max + 1, dtype='float')
    pmf = np.exp(-0.5*(x-mean)**2 / variance)
    pmf /= pmf.sum()
    d = stats.rv_discrete(values=(range(min, max + 1), pmf))
    return d.rvs(size=count)


def configuration_model(deg_seq, random_seed=None):
    import snap

    DegSeqV = snap.TIntV()
    for deg in deg_seq:
        DegSeqV.Add(deg)
    Rnd = snap.TRnd(random_seed if random_seed else random.randint(1e9))

    g = snap.GenConfModel(DegSeqV, Rnd)
    print("N=%d, E=%d" % (g.GetNodes(), g.GetEdges()))
    # for EI in g.Edges():
    #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
    return g


def test():
    import matplotlib.pyplot as plt

    # deg_seq = truncated_power_law(100000, gamma=1.6, maximal=100)
    deg_seq = truncated_normal(100, mean=10, variance=1000, min=1, max=100)
    print(deg_seq)

    # plt.hist(sample, bins=np.arange(100) + 0.5)
    # plt.show()
    # g = configuration_model(deg_seq)


if __name__ == '__main__':
    test()
