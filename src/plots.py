from operator import itemgetter

import six
import seaborn as sns

from graph_models.models import ba_model
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


COLOR_CYCLE_0 = list(six.iteritems(colors.cnames))
COLOR_CYCLE_1 = ['black',
                 'green', 'red', 'blue', 'magenta', 'orange', 'cyan',
                 'darkred', 'darkgreen', 'darkblue', 'darkmagenta', 'darkorange', 'darkcyan',
                 'pink', 'lime', 'wheat', 'lightsteelblue']
COLOR_CYCLE_2 = ['blue',
                 'red', 'orange',
                 'green', 'lightgreen']


def cumulative_function(x_list, reverse=False):
    """
    Translate a distribution into cumulative distribution.

    :param x_list: list of [x]
    :param reverse: reverse ordering
    :return: (y, N(x|x<y)), sorted by y
    """
    values_dict = {}
    for x in x_list:
        if x not in values_dict:
            values_dict[x] = 0
        values_dict[x] += 1
    sorted_list = sorted(values_dict.items(), key=itemgetter(0), reverse=reverse)
    res = []
    cum_count = 0
    for x, count in sorted_list:
        cum_count += count
        res.append((x, cum_count))
    return res


def plot_degree_distribution(graph_list, direction='total', cumulative=False, normalize=False, bar_plot=False, no_logs=False,
                             fill_in_zeros=False, same_figure=False, legend=False, title=None, xlim=None, ylim=None,
                             verbose=False, lang='EN', fontsize=12, **kwargs):
    """
    Draw (in-/out-/total) degree distribution for a given list of graphs.

    :param cumulative:
    :param graph_list: list of `Graph`s
    :param direction: 'in', 'out' or 'total' (default)
    :param bar_plot: draw a bar-plot
    :param no_logs: plot in linear scale, not log-log
    :param fill_in_zeros: draw zero degrees
    :param same_figure: draw graphic in the same matplotlib figure (should be defined)
    :return:
    """
    if direction not in ['out', 'in', 'total']:
        print("WARNING: incorrect degree dist parameter '%s'; 'total' will be used.")
        direction = 'total'

    if not same_figure:
        plt.figure(direction + "-degree distributions" if title is None else title)

    # todo refactor for directed graphs
    # def nx_degree_function(g):
    #     if not g.is_directed():
    #         return g.degree_iter()
    #     switch = {'in': g.in_degree_iter,
    #               'out': g.out_degree_iter,
    #               'total': g.degree_iter}
    #     return switch[direction]()

    lines = []
    for i, gr in enumerate(graph_list):
        if verbose:
            print("DD for ", gr._name, gr.get_size())

        deg_sequence = list(gr.get_node_property_dict('degree').values())

        if cumulative:
            if lang == 'RU':
                plt.ylabel(u"вершин с большей степ.", fontsize=fontsize)
            else:
                plt.ylabel("nodes with higher degree", fontsize=fontsize)
            xy = cumulative_function(deg_sequence, reverse=True)
            if normalize:
                total = sum(deg_sequence)
                xy = [(x, 1.*y/total) for x, y in xy]
        else:
            if lang == 'RU':
                plt.ylabel(u"число вершин", fontsize=fontsize)
            else:
                plt.ylabel("number of nodes", fontsize=fontsize)
            freq = {}
            for d in deg_sequence:
                if d not in freq:
                    freq[d] = 1
                else:
                    freq[d] += 1

            # If required add internal zero values (except logarithmic case)
            if fill_in_zeros and no_logs:
                for k in range(0, max(freq.keys()) + 1):
                    if k not in freq:
                        freq[k] = 0

            # Sort by degree
            xy = sorted(freq.items(), key=itemgetter(0))

        # TODO
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        x, y = zip(*xy)
        if bar_plot:
            line = plt.bar(np.array(x) + 1. * i / len(graph_list), y, width=0.9 / len(graph_list),
                           # color=COLOR_CYCLE_1[i % len(COLOR_CYCLE_1)],
                           **kwargs)
        else:
            draw = plt.plot if no_logs else plt.loglog
            line, = draw(x, y,
                         # color=COLOR_CYCLE_1[i % len(COLOR_CYCLE_1)],
                         **kwargs)
        lines.append(line)

    if legend:
        plt.legend(lines, map(lambda gr: gr._name, graph_list), loc=0)
    if lang == 'RU':
        direction_ru = {'in': u'входящая степень', 'out': u'исходящая степень', 'total': u'степень'}[direction]
        plt.xlabel(direction_ru, fontsize=fontsize)
    else:
        plt.xlabel("Degree" if direction == 'total' else direction + "-degree", fontsize=fontsize)
    return lines
    # plt.show()


def draw_stat(ax, graph_list, stat, scale, title=False, legend=False,
              no_ylabel=False, lang='EN', fontsize=12, **kwargs):
    # fontsize = kwargs['fontsize']
    # lang = kwargs['lang']
    ax.title.set_fontsize(fontsize)
    if stat.endswith('-DD'):  # 'in' 'out' 'total', prepend 'c' for cumulative
        cumulative = False
        if stat[0] == 'c':
            cumulative = True
            stat = stat[1:]
        direction = stat[:-3]
        if title:
            if lang == 'RU':
                direction_ru = {'in': u'вход.', 'out': u'исх.', 'total': ' '}[direction]
                ax.title.set_text((u'Кумул.' if cumulative else '') + u'распр.%sстепеней' % direction_ru)
            else:
                ax.title.set_text(('Cum. ' if cumulative else '') + '%s-degree distr.' % direction)
        return plot_degree_distribution(
            graph_list, direction=direction, cumulative=cumulative, legend=legend,
            fill_in_zeros=True, same_figure=True, verbose=False, lang=lang, fontsize=fontsize,
            #marker='.',
            **kwargs)

    else:
        raise NotImplementedError()


    # elif '|' in stat:
    #     y, x = stat.split('|')
    #     if title:
    #         ax.title.set_text('%s(%s) dependency' % (y, x))
    #     return plot_graph_numeric_stat(
    #         graph_list, x_stat=x, y_stat=y, use_logs=True, same_figure=True, bars=False,
    #         no_legend=not legend, lang=lang, fontsize=fontsize, marker='.', linestyle='-', **kwargs)
    #
    # elif stat.endswith('GP'):
    #     if title:
    #         ax.title.set_text(stat + ' ' + u'распределение' if lang == 'RU' else 'distribution')
    #     size = int(stat[0])
    #     directed = stat[2] != 'u'
    #     return draw_profiles(
    #         graph_list, size=size, directed=directed, use_logs=False, same_figure=True,
    #         names_instead_of_ids=True, bars=True, no_legend=not legend,
    #         legend_size=min(2 * scale + 1, 12), normalize='l1', no_ylabel=no_ylabel,
    #         rotation=0 if scale > 3 else 50, lang=lang, fontsize=fontsize, verbose=False, **kwargs)
    #
    # elif stat == 'ass-deg':
    #     if title:
    #         ax.title.set_text('Degree assortativity')
    #     return [plot_assortativity(graph, 'deg', lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #
    # elif stat == 'ass-in':
    #     if title:
    #         ax.title.set_text('In-degree assortativity')
    #     return [plot_assortativity(graph, 'in', lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #
    # elif stat == 'ass-out':
    #     if title:
    #         ax.title.set_text('Out-degree assortativity')
    #     return [plot_assortativity(graph, 'out', lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #
    # elif stat == 'out-in':
    #     if title:
    #         ax.title.set_text('Out/in-degree joint dist')
    #     return [plot_out_in_degree(graph, lang=lang, **kwargs) for graph in graph_list]
    #
    # elif stat == 'hop':
    #     if title:
    #         ax.title.set_text(u'Достижимость вершин' if lang == 'RU' else 'Approximate hop-plot')
    #     lines = [plot_distances(
    #         graph, as_undir=True, normalize_y=True, eff_diam=False, no_ylabel=no_ylabel, lang=lang,
    #         fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'wcc' or stat == 'WCC':
    #     if title:
    #         ax.title.set_text(u'Распределение WCC' if lang == 'RU' else 'WCC distribution')
    #     lines = [plot_cc_dist(
    #         graph, type='w', lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'scc' or stat == 'SCC':
    #     if title:
    #         ax.title.set_text(u'Распределение SCC' if lang == 'RU' else 'SCC distribution')
    #     lines = [plot_cc_dist(
    #         graph, type='s', lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'cc':
    #     if title:
    #         ax.title.set_text(
    #             u'Кумул.коэф.кластериз.' if lang == 'RU' else 'Clustering coeff. distr.')
    #     lines = [plot_clustering_cumulative(
    #         graph, normalize_y=True, marker='', linestyle='-', lang=lang, fontsize=fontsize, **kwargs)
    #         for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'cc-deg':
    #     if title:
    #         ax.title.set_text('Clustering coeff of degree dist')
    #     lines = [plot_clustering_degree(
    #         graph, average=True, marker='', linestyle='-', lang=lang, fontsize=fontsize, **kwargs)
    #         for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'sing':
    #     if title:
    #         ax.title.set_text('Top singular values of A')
    #     lines = [plot_singvalues(
    #         graph, min(graph.get_size()[0] - 1, 99), lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat.startswith('singvec-'):  # singvec-right or singvec-left
    #     which = stat[8:]
    #     if title:
    #         ax.title.set_text('First %s singular vector of A' % which)
    #     lines = [plot_first_singvector(
    #         graph, which_vector=which, k=min(graph.get_size()[0], 99), lang=lang, fontsize=fontsize, **kwargs) for
    #              graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat == 'eigvec':
    #     if title:
    #         ax.title.set_text('First eigenvector of symmetric A')
    #     lines = [plot_first_eigenvector(
    #         graph, k=min(graph.get_size()[0], 99), lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines
    #
    # elif stat.startswith('eig'):  # stats='eig-A' or 'eig-un-A', also L, M
    #     matrix = stat[-1]
    #     as_symm = stat[4] == 'u'
    #     if title:
    #         ax.title.set_text('%s eigenvalues of %s%s' %
    #                           ("Bottom" if matrix == 'L' else "Top", "symmetric " if as_symm else "", matrix))
    #     lines = [plot_top_eigenvalues(
    #         graph, matrix=matrix, as_symm=as_symm, k=min(graph.get_size()[0], 99),
    #         complex=(not as_symm), lang=lang, fontsize=fontsize, **kwargs) for graph in graph_list]
    #     if legend:
    #         plt.legend(lines, [graph.name for graph in graph_list], loc=0)
    #     return lines


def plot_statistics(graph_list, title=None, stats=['in-DD', 'out-DD', '3-GP'], single_plot=False,
                    start_subplot=0, grid=None, no_legend=False, no_subtitle=False, scale=3.,
                    lang='EN', fontsize=12, **kwargs):
    """
    Plot several statistics for given list of graphs.

    :param graph_list:
    :param stats:
    :param scale: size of each subplot in inches
    :param single_plot: use same subplot for all graphs
    :param grid: specify (nrows, ncols) explicitly. By default nrows = len(graph_list), ncols = len(stats)
    :param title: use specified title
    :param no_legend: don't add graph_models legend to each subplot
    :param no_subtitle: don't add graph_models title to each subplot
    :return:
    """
    # FIXME we treat all graphs as directed
    snaps = {}

    nrows = 1 if single_plot else len(graph_list)
    ncols = len(stats)

    if grid is not None:
        nrows, ncols = grid

    fig = plt.figure("Statistics" if title is None else title, (scale * ncols, scale * nrows))
    # kwargs.update({'lang': lang, 'fontsize': fontsize})

    lines = []
    legends = []
    if single_plot:
        for j, stat in enumerate(stats):
            print("  " + stat)
            ax = plt.subplot(nrows, ncols, j + 1 + start_subplot)

            # Draw statistic
            lines = draw_stat(ax, graph_list, stat, scale, title=not no_subtitle,
                              legend=not no_legend, lang=lang, fontsize=fontsize,
                              # color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                              **kwargs)
            # if j == 0:
            #     legends = [graph.name for graph in graph_list]
            #     # legend is all graphs on 1st plot
            #     plt.legend(lines, legends, loc=0)
    else:
        count = 1
        for i, graph in enumerate(graph_list):
            print("Stats for %s" % graph._name)
            for j, stat in enumerate(stats):
                print("  " + stat)
                ax = plt.subplot(nrows, ncols, count)
                count += 1

                # Draw statistic
                line = draw_stat(ax, [graph], stat, scale, i == 0, # fixme title=not no_subtitle,
                                 color=COLOR_CYCLE_1[i % len(COLOR_CYCLE_1)], lang=lang,
                                 fontsize=fontsize, **kwargs)[0]
                if j == 0:
                    lines.append(line)
                    legends.append(graph._name)
                    # legend is  only 1 current graph on 1st plot
                    plt.legend([line], [graph._name], loc=0)

    plt.tight_layout()


def test():
    graph = ba_model(n=100000, avg_deg=50)
    plot_statistics([graph], stats=['total-DD'], scale=8, marker='o')
    plt.grid()
    plt.show()
    # sns.plot()


if __name__ == '__main__':
    test()
