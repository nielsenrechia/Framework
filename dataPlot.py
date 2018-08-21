#encoding=utf8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import floor
import itertools
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


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


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    # dados = ddata['dcoord'][-1]
    # max_d = max(dados)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')


def plot_dendrogram(l, path_dendrograms, period, method, clusters):
    plt.clf()
    print 'dendogram...'
    fancy_dendrogram(
        l,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=40,  # show only the last p merged clusters # sept = 80
        show_leaf_counts=True,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=11.,  # font size for the x axis labels
        show_contracted=True,  # to get a distribution impression in truncated branches
        annotate_above=l.iloc[-13:-12, 2].values,
        max_d=l.iloc[-clusters:-clusters+1, 2].values + 0.05e+31   # sept = 158000, jan athen 1.578e+102 or 1.628e+102
    )
    plt.tight_layout()
    plt.savefig(path_dendrograms + 'dendrogram_' + period + '_' + method + '2.png', bbox_inches='tight', pad_inches=0)
    z = 0


def plot_gap(maxClusters, min_nc, optimal, gap_results, period, method, path_labels):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.2, wspace=0.6)

    # ax1.scatter(X["x"], X["y"])
    ax1.plot(xrange(min_nc, maxClusters + 1), gap_results['OrigWk'].values, linestyle='-', color='green', marker='o',
             linewidth=2)
    ax1.set_ylabel(r"$Wk$")
    # r"$W_{k} = \sum_{r=1}^{k}\frac{1}{2n_{r}}\sum_{i,i' \in C_{r}}d_{ii'}$"
    ax2.plot(xrange(min_nc, maxClusters + 1), gap_results['OrigLogWk'].values, linestyle='-', color='red', marker='o',
             linewidth=2, label=r"$log(W_{k})$")
    # r"$log(W_{k})$"
    ax2.plot(xrange(min_nc, maxClusters + 1), gap_results['ElogW'].values, linestyle='-', color='blue', marker='o',
             linewidth=2, label=r"$E^{*}_{n}log(W^{*}_{k}) = \frac{1}{B}\sum_{b=1}^{B}log(W^{*}_{kb})$")
    # r"$E^{*}_{n}log(W^{*}_{k}) = \frac{1}{B}\sum_{b=1}^{B}log(W^{*}_{kb})$"
    # ax2.set_ylabel('Raw data log(Wk) / Average of LogWk from B references')
    ax3.plot(xrange(min_nc, maxClusters + 1), gap_results['Gap'].values, linestyle='-', color='green', marker='o',
             linewidth=2)
    # r"$Gap(k) = E^{*}_{n}log(W^{*}_{k}) - log(W_{k})$"
    ax3.set_ylabel('Gap values')
    ax3.axvline(x=optimal, color='green', linestyle="--", linewidth=2)
    # ax3.text(optimal - 1, 0.01, 'Optimal K ->', ha='center', va='center', size=16)
    ax4.bar(xrange(min_nc, maxClusters + 1), gap_results['GapSdSk'].values)
    ax4.set_ylabel(r"$Gap(k)-(Gap(k+1)-s_{k+1})$")
    # r"$Gap(k)-(Gap(k+1)-s_{k+1})$"
    # ax4.plot(xrange(2, maxClusters + 1), gap_results['Sd'].values)
    fig.text(0.5, 0.01, 'Number of clusters (K)', ha='center', va='center', size=16)
    ax2.legend(loc='upper right', shadow=False)
    plt.tight_layout()
    fig.savefig(path_labels + 'gap_score_' + period + '_' + method + '.png', bbox_inches='tight', pad_inches=0)


def plot_silhouette(max_nc, min_nc, SmaxI, silhouette, period, method, path_labels):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(range(min_nc, max_nc+1), silhouette, label='silhouette',
            linestyle='-', color='blue', marker='o', linewidth=2)
    ax.set_ylabel('Silhouette score')
    ax.set_xlabel('Number of clusters (K)')
    ax.axvline(x=SmaxI, color='blue', linestyle="--", linewidth=2)
    plt.tight_layout()
    fig.savefig(path_labels + 'silhouette_score_' + period + '_' + method + '.png', bbox_inches='tight', pad_inches=0)


def plot_knee(max_nc, min_nc, l, num_clust, period, method, path_labels):
    last = l[-max_nc+min_nc-1:, 2] #ultimos k cluster na coluna das distancias
    last_rev = last[::-1]
    idxs = np.arange(min_nc, max_nc + 1)
    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Number of clusters (K)')
    ax.plot(idxs, last_rev, label='SSE %s' % method, linestyle='-', color='red', marker='o', linewidth=2)
    ax.plot(idxs[:-2] + 1, acceleration_rev, label=u'Aceleração', linestyle='-', color='c', linewidth=2)
    # ax.text(num_clust, l[::-1, 2][num_clust - 1], '     Possible\n<- knee point (%i)' % num_clust)
    ax.axvline(x=num_clust, color='red', linestyle="--", linewidth=2)
    ax.legend(loc='upper right', shadow=False)
    plt.tight_layout()
    fig.savefig(path_labels + 'knee_score_' + period + '_' + method + '.png', bbox_inches='tight', pad_inches=0)


def plot_ch(max_nc, min_nc, CHmax, calinski, period, method, path_labels):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(min_nc, max_nc+1), calinski, label='Calinski Harabaz Index',
            linestyle='-', color='m', marker='o', linewidth=2)
    ax.set_ylabel('Calinski Harabaz Index')
    ax.set_xlabel('Number of clusters (K)')
    ax.axvline(x=CHmax, color='m', linestyle="--", linewidth=2)
    plt.tight_layout()
    fig.savefig(path_labels + 'calinski_score_' + period + '_' + method + '.png', bbox_inches='tight', pad_inches=0)