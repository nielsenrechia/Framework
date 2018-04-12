# encoding=utf8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import floor
import itertools
import seaborn as sns


def get_idxs(max):
    times = int(floor(np.log10(max-1)))
    basic = [0, 1, 2, 3, 4, 5, 6, 8]
    baseten = [(10 ** i)-1 for i in xrange(1, times + 1, 1)]
    midvalues = [-1 + 2 ** i * 5 ** (i + 1) for i in xrange(1, times+1, 1)]
    if max < np.max(midvalues):
        del midvalues[-1]
    return baseten, sorted(basic + [max-1] + baseten + midvalues)


# define best apps by cumulative distribuition
def plot_best_apps(X, plotData, language, max, thresholds):
    sum_by_data = X[['pkgs', plotData]].groupby('pkgs').sum()
    matrix = pd.concat([sum_by_data, pd.DataFrame(X['pkgs'].value_counts())], axis=1, join='inner')
    matrix.columns = ['App_access_time', 'Users']
    matrix['Acess_per'] = matrix['App_access_time'] / np.sum(matrix['App_access_time'])
    matrix['Acess_per_cumsum'] = matrix['Acess_per'].sort_values(ascending=False).cumsum()
    matrix['Acess_per_by_user_cumsum'] = matrix.sort_values(by='Users', ascending=False)['Acess_per'].cumsum()

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax3 = ax2.twiny()

    baseten, idxs = get_idxs(max)

    cumsumV = matrix['Acess_per_cumsum'].sort_values(ascending=True)[idxs].values
    uCumsumV = matrix['Acess_per_by_user_cumsum'].sort_values(ascending=True)[idxs].values
    allV = matrix['Acess_per_cumsum'].sort_values(ascending=True).values
    colors = itertools.cycle(["purple", "b", "black", "c"])

    tV = []
    tL = []
    for t in thresholds:
        bigIdx = next(x[0] for x in enumerate(allV) if x[1] > t)
        bigIdxV = matrix['Acess_per_cumsum'].sort_values(ascending=True)[[bigIdx]].values
        afterIdx = next(x[0] for x in enumerate(cumsumV) if x[1] > bigIdxV)
        after = idxs[afterIdx]
        before = idxs[afterIdx-1]
        if bigIdx < (after-before):
            cut2 = (afterIdx-1) + ((1./(after-before))*bigIdx)
        else:
            cut2 = (afterIdx - 1) + ((1. / (after - before)) * (bigIdx-(after-before)))
        tV.append(cut2)
        tL.append(bigIdx)
        color = next(colors)
        ax.axhline(y=t, color=color, linestyle='--', linewidth=2)
        ax.axvline(x=cut2, color=color, linestyle='--', linewidth=2)

    if language == 'en':
        ax.plot(cumsumV, linestyle='-', marker='o', color='g', linewidth=2, markersize=2, label='usage time')
        ax.plot(uCumsumV, linestyle='--', marker='s', color='r', linewidth=2, markersize=2, label='usage time by users')
        ax.set_ylabel('cumulative contribution(%)')
        ax.set_xlabel('top apps (#)')
        # ax.set_title('Teste')

    elif language == 'pt-br':
        ax.plot(cumsumV, linestyle='-', marker='o', color='g', linewidth=2, markersize=2, label='tempo de uso')
        ax.plot(uCumsumV, linestyle='--', marker='s', color='r', linewidth=2, markersize=2, label=u'tempo de uso pelo nº de usuários')
        ax.set_ylabel('Porcentagem Acumulativa (%)')
        ax.set_xlabel('Aplicativos Principais (#)')

    yV = [i / 100. for i in xrange(10, 110, 10)]
    ax.set_yticks(yV)
    ax.set_yticklabels([str(int(i * 100)) for i in yV])
    ax2.set_yticks(thresholds)
    ax2.set_yticklabels([str(int(i * 100)) for i in thresholds])
    ax.set_ylim([0.08, 1.03])
    ax2.set_ylim([0.08, 1.03])

    xV = list(ax.get_xticks())
    xlabels = [str(i + 1) for i in sorted((set([0, 2, 4, 6] + baseten + [max - 1])))]
    ax.set_xticks(sorted(set(xV)))
    ax3.set_xticks(sorted([i for i in tV]))
    ax3.set_xlim([np.min(xV), np.max(xV)])
    ax.set_xticklabels(xlabels, horizontalalignment='center')
    ax3.set_xticklabels([str(i) for i in tL], horizontalalignment='center')

    ax.legend(loc='lower right', shadow=True)
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major')
    plt.show()


def plot_discretization(matrix, populars_pkgs, most_used_pkgs, barcodes, discretization):
    discretization_type = ['frequency', 'ip']
    populars_pkgs.sort_values(by='customers', inplace=True)
    head_pkg = populars_pkgs.loc[populars_pkgs['customers'].idxmax()]
    median_pkg = populars_pkgs[populars_pkgs['customers'] >= populars_pkgs['customers'].median()].iloc[0]
    tail_pkg = populars_pkgs.loc[populars_pkgs['customers'].idxmin()]
    populars_pkgs = populars_pkgs[populars_pkgs['pkgs'].isin([head_pkg['pkgs'], median_pkg['pkgs'], tail_pkg['pkgs']])]
    frequency_matrix, ip_matrix = discretization(matrix, populars_pkgs, most_used_pkgs, discretization_type, barcodes, plot=True)
    matrix = matrix[ip_matrix.columns.values]

    pkgs = populars_pkgs['pkgs'].unique()

    # fig, axs = plt.subplots(3, 3)
    # plt.show()
    result = pd.DataFrame()

    for pkg in pkgs:
        df = pd.DataFrame(matrix[matrix[pkg] != 0.0][pkg].rename('duration_sec'))
        df['pkg'] = pkg
        df = pd.concat((df, pd.Series(np.random.uniform(0.01, 0.09, df.shape[0]), name="y", index=df.index.values)),
                       axis=1)
        for d in discretization_type:
            dfd = df.copy()
            dfd['discretization'] = d
            if d == "frequency":
                dfd = pd.concat((dfd, frequency_matrix[frequency_matrix[pkg] != 0.0][pkg].rename("intervals")), axis=1)
            # elif d == "clustering":
            #     df = pd.concat((df, cluster_matrix[cluster_matrix[pkg] != 0.0][pkg].rename("class")), axis=1)
            else:
                dfd = pd.concat((dfd, ip_matrix[ip_matrix[pkg] != 0.0][pkg].rename("intervals")), axis=1)
            result = pd.concat((result, dfd))

    result['intervals'] = result['intervals'].str[0]

    g = sns.FacetGrid(result.reset_index(True), col="pkg", row="discretization", hue="intervals", margin_titles=True, sharey=True, sharex=False, palette="Set1").map(
        plt.scatter, "duration_sec", "y").add_legend()
    g.set(ylim=(0.003, 0.093))

    for ax, d in zip(g.axes, discretization_type):
        for a, pkg in zip(ax, pkgs):
            m = result[result['pkg'] == pkg][['duration_sec']].max().values + 0.6
            a.set_xlim(0.0, m)
            a.text(m - 10, 0.005, str(
                result[(result['pkg'] == pkg) & (result['discretization'] == d)][['intervals']].astype(int).max().values[
                    0]) + ' intervals')

    g = g.fig.subplots_adjust(wspace=.02, hspace=.06)
    plt.show()


def plot_best_apps_selection(X, ax1, ax2, ax3, wmin, wmax, fmin, fmax, gmin, gmax):
    t = X[X['pkgs'] == 'com.whatsapp']['duration_sec']
    plt.setp(ax1.get_xticklabels(), rotation=90)
    x = sns.distplot(t, ax=ax1, color='g', bins=50)
    ax1.set_xlim(wmin, wmax)
    ax1.set_ylim(0, 0.0002)
    t = X[X['pkgs'] == 'com.facebook.katana']['duration_sec']
    plt.setp(ax2.get_xticklabels(), rotation=90)
    ax2.set_xlim(fmin, fmax)
    ax2.set_ylim(0, 0.0002)
    y = sns.distplot(t, ax=ax2, color='b', bins=50)
    t = X[X['pkgs'] == 'com.android.chrome']['duration_sec']
    plt.setp(ax3.get_xticklabels(), rotation=90)
    ax3.set_xlim(gmin, gmax)
    ax3.set_ylim(0, 0.0004)
    z = sns.distplot(t, ax=ax3, color='y', bins=50)
    return x, y, z
