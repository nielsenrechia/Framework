# encoding=utf8
import pandas as pd
import numpy as np
from numpy import inf
from math import ceil
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, inconsistent, maxdists, maxinconsts, is_monotonic
from scipy.stats import hmean
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.rlike.container as rlc
import gc
from scipy.spatial.distance import pdist, squareform, is_valid_y, is_valid_dm, num_obs_y, num_obs_dm
from scipy.sparse import csr_matrix
import gc
import csv


def barcodes_distance(rules, discretization):

    rules['len'] = pd.Series(rules.notnull().sum(axis=1), index=rules.index)
    rules_len = rules['len'].unique()
    for l in rules_len:
        if l == 2:
            rules2 = rules[rules['len'] == 2]
        elif l == 3:
            rules3 = rules[rules['len'] == 3]
        elif l == 4:
            rules4 = rules[rules['len'] == 4]
    barcodes = discretization.index.values
    m, n = discretization.shape
    # x = list(np.arange(40000))
    # barcodes_reversed = list(reversed(barcodes))
    # barcodes = [1,2,3,4,5]
    # barcodes_reversed = list(reversed(barcodes))
    # distances_2 = pd.DataFrame(index=barcodes, columns=barcodes)
    # distances_3 = pd.DataFrame(index=barcodes, columns=barcodes)
    # dm = pdist(matrix.values, lambda u, v: np.sqrt(((u - v) ** 2).sum()))
    # # m = squareform(dm)

    all_rules = []
    print 'rules ....'
    for barcode in barcodes:
        # print 'barcode i ' + str(barcode_i)
        barcode_matrix = discretization.loc[barcode]
        barcode_matrix = barcode_matrix[(barcode_matrix != '0.0') & (barcode_matrix != 0)]
        b_list = []
        b_list += [str(pkg)+'='+str(val) for pkg, val in barcode_matrix.iteritems()]
        b_rules = []
        for l in rules_len:
            if l == 2:
                b_rules.extend(rules2.loc[(rules['x1'].isin(b_list)) & (rules2['x2'].isin(b_list))].index)
            elif l == 3:
                b_rules.extend(rules3.loc[(rules['x1'].isin(b_list)) & (rules3['x2'].isin(b_list))
                                        & (rules3['x3'].isin(b_list))].index)
            elif l == 4:
                b_rules.extend(rules4.loc[(rules4['x1'].isin(b_list)) & (rules4['x2'].isin(b_list)) & (rules4['x3'].isin(b_list)) & (rules4['x4'].isin(b_list))].index)
        all_rules += [b_rules]

    all_rules = np.array(all_rules)
    for d in xrange(2,3,1):
        print d
        #
        # distances = pd.DataFrame(index=barcodes, columns=barcodes, dtype=np.float16)
        # distances.info()
        y = np.zeros((m * (m - 1)) // 2, dtype=np.float32)

        print 'distances .....'
        k = 0
        for i in xrange(0, m - 1):
            for j in xrange(i + 1, m):
                # union_old = float(len(set(all_rules[i]).union(set(all_rules[j]))))
                # intersection_old = float(len(set(all_rules[i]).intersection(set(all_rules[j]))))
                union = float(len(np.union1d(all_rules[i], all_rules[j])))
                intersection = float(len(np.intersect1d(all_rules[i], all_rules[j])))

                if union == 0:
                    # y[k] = 1.0
                    y[k] = np.inf
                    k = k + 1

                else:
                    # dist = np.float32(1 - (intersection / union))
                    dist = np.float32(round(-np.log(intersection / union),2))
                    y[k] = dist
                    k = k + 1

            print str(i) + "........................................"

        y_distances = pd.DataFrame(y, dtype=np.float32)
        # y_distances.to_csv(path_barcodes_distance + 'itensets_y_log_' + discretization_filename + ".csv",
        #                 header=False, index=False)

        print 'cleaning memory 1 ....'
        gc.collect()

        # del y_distances
        # y_t = is_valid_y(y)
        # print y_t
        # dm = squareform(y)
        return y_distances


def hac_clustering_barcodes(distances, methods, max_nc, path_labels, period, path_linkage, result):

    print "replacing inf values....."
    distances = distances.replace(inf, np.float32(1.79e+30))
    print 'tranforming distances in pdist format....'
    distances = np.float32(distances.values.reshape(distances.shape[0],))

    print 'is valid y ....'
    print str(is_valid_y(distances))

    # dm = squareform(distances)
    # dm_t = is_valid_dm(dm)

    print 'num obs in y ...'
    print str(num_obs_y(distances))
    # num_dm = num_obs_dm(dm)

    print 'cleaning memory 1....'
    gc.collect()

    for method in methods:
        # if method == 'single':
        print 'method ' + method + ' ...'
        l = linkage(distances, method=method)
        print 'cleaning memory 2 ....'
        gc.collect()
        lpd = pd.DataFrame(l)
        print 'saving linkage matrix ...'
        lpd.to_csv(path_linkage + 'linkage_y_' + period + '_' + method + '.csv.gz', index=False, header=False, compression='gzip')
        print 'cleaning memory 4 ....'
        del lpd
        gc.collect()

        # print 'dendogram...'
        # fancy_dendrogram(
        #     l,
        #     truncate_mode='lastp',  # show only the last p merged clusters
        #     p=50,  # show only the last p merged clusters # sept = 80
        #     show_leaf_counts=True,  # otherwise numbers in brackets are counts
        #     leaf_rotation=90.,  # rotates the x axis labels
        #     leaf_font_size=12.,  # font size for the x axis labels
        #     show_contracted=True,  # to get a distribution impression in truncated branches
        #     annotate_above=50,   # sept = 10000
        #     max_d=50    # sept = 158000, jan athen 1.578e+102 or 1.628e+102
        # )
        # plt.show()
        # z = 0

        print 'Starting Silhouette score...'
        silhouette = []
        stds = []
        armonicas = []
        armonicas_1 = []
        # #
        for k in xrange(2, max_nc+1):
            labels = fcluster(l, k, 'maxclust')
            z = 0
            if len(np.unique(labels)) > 1:
                silhouette += [round(silhouette_score(squareform(distances), labels, metric='precomputed'), 3)]
                print 'Silhouette score for k = ' + str(k) + ' is ' + str(silhouette[k - 2]) + ' ...'
                unique, counts = np.unique(labels, return_counts=True)
                stds += [np.std(counts)]
                armonicas += [hmean(counts)]
                armonicas_1 += [1 / np.sum(1.0/counts, axis=0)]
            else:
                silhouette += [0.0]
                stds += ['no']
                armonicas += ['no']
                armonicas_1 += ['no']
            labels = pd.DataFrame(labels)
            labels.to_csv(path_labels + 'labels_' + period + '_' + method + '_' + str(k) + '.csv', header=False,
                          index=True)
            print 'cleaning memory ....'
            gc.collect()

        #
        # sw_result = []
        # if len(silhouette) > 2:
        #     i_max_s = silhouette.index(max(silhouette))+2
        #     max_s = max(silhouette)
        #     silhouette = np.array(silhouette)
        #     i_max_s_2 = silhouette.argsort()[-2:][0] + 2
        #     i_max_s_3 = silhouette.argsort()[-3:][0] + 2
        #     max_s_2 = silhouette[i_max_s_2-2]
        #     max_s_3 = silhouette[i_max_s_3-2]
        #
        #     sw_result += [i_max_s, max_s, i_max_s_2, max_s_2, i_max_s_3, max_s_3]
        # else:
        #     sw_result += ['no', 'no', 'no', 'no', 'no', 'no']

        # aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        # knee = np.diff(l[::-1, 2], 2)[:29]
        #
        # results = np.column_stack((np.array(knee), np.array(silhouette), np.array(stds), np.array(armonicas),
        #                            np.array(armonicas_1)))
        # result[method] = results
        # aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii


        # num_clust = knee.argmax() + 2  # posicoa do maior valor (+2 pq cluster comeca em 2 e posicao em 0)
        # r_num_clust = round(knee[knee.argmax()], 2)
        # num_clust2 = knee.argsort()[-2:][0] + 2
        # r_num_clust_2 = round(knee[knee.argsort()[-2:][0]], 2)
        # num_clust3 = knee.argsort()[-3:][0] + 2
        # r_num_clust_3 = round(knee[knee.argsort()[-3:][0]], 2)
        #
        # SSE_result = []
        # SSE_result += [num_clust, r_num_clust, num_clust2, r_num_clust_2, num_clust3, r_num_clust_3]
        # z = 0

        # gc.collect()

        # distance_matrix = squareform(distances)
        #
        # n_objects, n_objects = distance_matrix.shape
        # n_clusters = np.unique(labels).shape[0]
        #
        # if n_clusters > 1:
        #
        #     # smallest inter-cluster distance divided biggest intra-cluster distance
        #
        #     inter_cluster = np.inf
        #     intra_cluster = -np.inf
        #
        #     for i in xrange(n_objects):
        #         for j in xrange(i, n_objects):
        #             if labels[i] == labels[j]:
        #                 if distance_matrix[i, j] > intra_cluster:
        #                     intra_cluster = distance_matrix[i, j]
        #             else:
        #                 if distance_matrix[i, j] < inter_cluster:
        #                     inter_cluster = distance_matrix[i, j]
        #
        #     index = float(inter_cluster) / float(intra_cluster)
        #     # if there is only one cluster, then the index is zero because the smallest distance between
        #     # two clusters is also zero
        #     if np.isnan(index):
        #         index = np.nan_to_num(index)

        #
        # all_results = SSE_result + sw_result

        # result.set_value(week, method, all_results)

    # result.to_csv(path_labels + 'clustering_evaluation' + period + '.csv', header=True, index=True,
    #               quoting=csv.QUOTE_NONE, escapechar='\\')
    print 'fim ...............................................................'
