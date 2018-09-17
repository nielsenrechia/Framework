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
    header = barcodes.columns.values
    for d in xrange(len(dates) - 11):
        start_date = dates[d]
        end_date = dates[d + 1]
        period = str(start_date.date()) + '_' + str(end_date.date())
        print period

        header = np.append(header, period)

        labels_week = pd.read_csv(
            path_labels + 'barcodes_labels_' + period + '_' + method + '_' + str(clusters[d]) + '.csv', index_col=1,
            header=0)
        outliers = pd.read_csv(path_outliers + 'outliers_' + period + 'csv.gz', index_col=None, header=None)

        if d == 0:
            first_week = period
            all_labels = pd.concat([barcodes, labels_week], axis=1)
            z = 0
        else:
            all_labels = pd.concat([all_labels, labels_week], axis=1)
            z = 0

        all_labels.loc[outliers[0], 'labels'] = 0.1
        all_labels.columns = header

    all_labels_without_out_first_week = all_labels.dropna(subset=[first_week])
    all_labels_without_out_first_week['weeks_out'] = all_labels_without_out_first_week.isnull().sum(axis=1)
    all_labels_without_out_first_week.to_csv('results/all_labels_without_barcodes_out_first_week.csv', header=True,
                                             index_label='barcodes', index=True)
    all_labels['weeks_out'] = all_labels.isnull().sum(axis=1)
    # all_groups = all_groups[all_groups['weeks'] < 3]
    all_labels.to_csv('results/all_labels_barcodes.csv', header=True, index_label='barcodes', index=True)

    return all_labels_without_out_first_week


def get_all_behaviors_barcodes(all_groups):
    header = all_groups.columns[3:-1]
    # header = ['11/dec - 17/dec', '18/dec - 24/dec', '25/dec - 31/dec', '01/jan - 07/jan', '08/jan - 14/jan',
    #           '15/jan - 21/jan', '22/jan - 28/jan', '29/jan - 04/feb', '05/feb - 11/feb', '12/feb - 18/feb']
    barcodes = all_groups.index.values

    behaviors = pd.DataFrame(index=barcodes, columns=header[1:])

    for i, h in enumerate(header[:9]):
        print h
        for b in barcodes:
            # if b == 'ie31bc19f132f945c827f1f2f51bfcd156c1c00aed':
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

    behaviors.to_csv('results/all_behaviors_barcodes_without_first_week.csv', header=True, index_label='barcodes',
                     index=True)
    z = 0