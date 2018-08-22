# encoding=utf8
import pandas as pd
import numpy as np
from numpy import inf
from utils.clustering_validity import gap_statistic, swc, calculate_dispersion, ch
from dataPlot import plot_gap, plot_ch, plot_knee, plot_silhouette, plot_dendrogram
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, inconsistent, maxdists, maxinconsts, is_monotonic
from scipy.stats import hmean
from scipy.spatial.distance import pdist, squareform, is_valid_y, is_valid_dm, num_obs_y, num_obs_dm
from scipy.sparse import csr_matrix
import gc
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
from numba import jit


def get_all_groups_barcodes(path_labels, path_outliers, dates, method, clusters, barcodes):
    header = []
    weeks = []
    for d in xrange(len(dates) - 11):
        start_date = dates[d]
        end_date = dates[d + 1]
        period = str(start_date.date()) + '_' + str(end_date.date())

        header += [period]
        weeks += [1]

        if d == 0:
            all_groups = pd.read_csv(path_labels + 'labels_' + period + '_' + method + '_' + str(clusters[d]) + '.csv', index_col=0, header=None)
            outliers = pd.read_csv(path_outliers + 'outliers_' + period + 'csv.gz', index_col=None, header=None)
            not_outliers = barcodes[~barcodes['barcodes'].isin(outliers[0])]
            z = 0
        else:
            new_groups = pd.read_csv(path_labels + 'labels_' + period + '_' + method + '_' + str(clusters[d]) + '.csv', index_col=0, header=None)
            weeks += [d + 2]
            all_groups = pd.concat([all_groups, new_groups], axis=1)
    all_groups.columns = header

    all_groups = all_groups.dropna(subset=['11/dec - 17/dec'])
    all_groups['weeks'] = all_groups.isnull().sum(axis=1)
    # all_groups = all_groups[all_groups['weeks'] < 3]

    all_groups.to_csv(path + 'all_groups_barcodes_new_dist.csv', header=True, index_label='barcodes', index=True,
                      quoting=csv.QUOTE_NONE, escapechar='\\')


def get_all_behaviors_barcodes(path, all_groups):
    all_groups = pd.read_csv(path + all_groups, nrows=None, header=0, index_col=0)
    # header = ['04/sep - 10/sep', '11/sep - 17/sep', '18/sep - 24/sep', '25/sep - 01/oct', '02/oct - 08/oct',
    #           '09/oct - 15/oct', '16/oct - 22/oct', '23/oct - 29/oct', '20/oct - 05/nov', '06/nov - 11/nov']
    header = ['11/dec - 17/dec', '18/dec - 24/dec', '25/dec - 31/dec', '01/jan - 07/jan', '08/jan - 14/jan',
              '15/jan - 21/jan', '22/jan - 28/jan', '29/jan - 04/feb', '05/feb - 11/feb', '12/feb - 18/feb']
    barcodes = all_groups.index.values

    behaviors = pd.DataFrame(index=barcodes, columns=header[1:])

    for i, h in enumerate(header[:9]):
        print h
        for b in barcodes:
            actual_label = all_groups[h].loc[b]
            next_label = all_groups[header[i + 1]].loc[b]
            if next_label == 0.1:
                behaviors.loc[b, header[i + 1]] = 'outlier'
            elif np.isnan(next_label):
                behaviors.loc[b, header[i + 1]] = 'miss'
            else:
                if np.isnan(actual_label):
                    last_life = all_groups.loc[b][:i]
                    last_life = last_life[~last_life.isnull()]
                    last_week = last_life[-1:].index[0]
                    last_label = last_life[-1:].values[0]
                    last_group = all_groups[all_groups[last_week] == last_label].index
                    next_week_group = all_groups[header[i + 1]].loc[last_group]
                    next_flock_label = next_week_group.value_counts().idxmax()
                    if next_label == next_flock_label:
                        behaviors.loc[b, header[i + 1]] = 'loyal'
                    else:
                        behaviors.loc[b, header[i + 1]] = 'C'
                else:
                    actual_group = all_groups[all_groups[h] == actual_label].index
                    next_week_group = all_groups[header[i + 1]].loc[actual_group]
                    next_flock_group = next_week_group.value_counts().idxmax()
                    if next_label == next_flock_group:
                        behaviors.loc[b, header[i + 1]] = 'loyal'
                    else:
                        behaviors.loc[b, header[i + 1]] = 'C'

    behaviors.to_csv(path+'all_behaviors_barcodes_new_dist.csv', header=True,
                     index_label='barcodes', index=True, quoting=csv.QUOTE_NONE, escapechar='\\')
    z = 0