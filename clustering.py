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
    t = barcodes.shape[0]
    # x = list(np.arange(40000))
    # barcodes_reversed = list(reversed(barcodes))
    # barcodes = [1,2,3,4,5]
    # barcodes_reversed = list(reversed(barcodes))
    # distances_2 = pd.DataFrame(index=barcodes, columns=barcodes)
    # distances_3 = pd.DataFrame(index=barcodes, columns=barcodes)
    # dm = pdist(matrix.values, lambda u, v: np.sqrt(((u - v) ** 2).sum()))
    # # m = squareform(dm)

    all_rules = []
    outliers = []
    remove = []
    print 'rules ....'
    for i, barcode in enumerate(barcodes):
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
        if not b_rules:
            outliers += [barcode]
            remove += [i]
        else:
            all_rules += [b_rules]

    all_rules = np.array(all_rules)
    barcodes = np.delete(barcodes, remove)
    m = barcodes.shape[0]
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
        return outliers, y_distances


def hac_clustering_barcodes(X, distances, methods, max_nc, min_nc, Brefs, path_labels, period, path_linkage, result, best_res):

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

        print 'method ' + method + ' ...'

        l = linkage(distances, method=method)
        print 'cleaning memory 2 ....'
        gc.collect()
        lpd = pd.DataFrame(l)
        print 'saving linkage matrix ...'
        lpd.to_csv(path_linkage + 'linkage_y_' + period + '_' + method + '.csv.gz', index=False, header=False,
                   compression='gzip')
        print 'cleaning memory 4 ....'
        del lpd
        gc.collect()

        r, f = X.shape
        origLogW = np.zeros(len(xrange(min_nc, max_nc + 1)))
        origW = np.zeros(len(xrange(min_nc, max_nc + 1)))
        gaps = np.zeros(len(xrange(min_nc, max_nc + 1)))
        ElogW = np.zeros(len(xrange(min_nc, max_nc + 1)))
        GapSdSk = np.zeros(len(xrange(min_nc, max_nc + 1)))
        Sd = np.zeros(len(xrange(min_nc, max_nc + 1)))
        Sk = np.zeros(len(xrange(min_nc, max_nc + 1)))
        bestk = []
        silhouette = [None] * (max_nc - min_nc + 1)
        stds = [None] * (max_nc - min_nc + 1)
        armonicas = [None] * (max_nc - min_nc + 1)
        armonicas_1 = [None] * (max_nc - min_nc + 1)
        calisnki = [None] * (max_nc - min_nc + 1)

        for index, k in enumerate(xrange(min_nc, max_nc + 1)):

            print 'GAP score for k = ' + str(k)

            labels = fcluster(l, k, 'maxclust').reshape(-1, 1)
            x = np.concatenate((X, labels), axis=1)
            origW[index] = calculate_dispersion(x, labels)
            origLogW[index] = np.log(origW[index])

            gaps[index], ElogW[index], Sd[index], Sk[index] = gap_statistic(X, k, origLogW[index], r, f,
                                                                            method=method, brefs=Brefs)

            if index > 0:
                GapSdSk[index - 1] = gaps[index - 1] - gaps[index] - Sk[index]
                if gaps[index - 1] >= gaps[index] - Sk[index]:
                    bestk.append(index - 1)

            print 'cleaning memory ....'
            gc.collect()

            print 'Silhouette score for k = ' + str(k)

            silhouette[index], stds[index], armonicas[index], armonicas_1[index] = swc(labels, distances)

            print 'cleaning memory ....'
            gc.collect()

            print 'Calisnki H score for k = ' + str(k)
            calisnki[index] = ch(X, labels)

            print 'cleaning memory ....'
            gc.collect()

            labels = pd.DataFrame(labels)
            labels.to_csv(path_labels + 'labels_' + period + '_' + method + '_' + str(k) + '.csv',
                          header=False, index=True)

        print 'cleaning memory ....'
        gc.collect()

        print 'Getting best GAP ...'
        gap_optimal = np.min(bestk) + min_nc
        gap_r = gaps[gap_optimal-min_nc]
        gap2 = np.partition(bestk, 1)[1]+min_nc
        gap_r2 = gaps[gap2 - min_nc]
        gap3 = np.partition(bestk, 2)[2]+min_nc
        gap_r3 = gaps[gap3 - min_nc]
        gaps_r = [gap_optimal, gap_r, gap2, gap_r2, gap3, gap_r3]

        print 'cleaning memory ....'
        gc.collect()

        print 'Getting best Silhouette ...'
        SmaxI = silhouette.index(max(silhouette))+min_nc
        Smax = max(silhouette)
        silhouette = np.array(silhouette)
        Smax2I = silhouette.argsort()[-2:][0] + min_nc
        Smax3I = silhouette.argsort()[-3:][0] + min_nc
        Smax2 = silhouette[Smax2I-min_nc]
        Smax3 = silhouette[Smax3I-min_nc]
        sw_result = [SmaxI, Smax, Smax2I, Smax2, Smax3I, Smax3]

        print 'cleaning memory ....'
        gc.collect()

        print 'Getting best Knee ...'
        knee = np.diff(l[::-1, 2], 2)[:max_nc-min_nc+1]
        num_clust = knee.argmax() + 3  # posicoa do maior valor (+3 pq cluster comeca em 2 e posicao em 0)
        r_num_clust = round(knee[knee.argmax()], 2)
        num_clust2 = knee.argsort()[-2:][0] + 3
        r_num_clust_2 = round(knee[knee.argsort()[-2:][0]], 2)
        num_clust3 = knee.argsort()[-3:][0] + 3
        r_num_clust_3 = round(knee[knee.argsort()[-3:][0]], 2)
        SSE_result = [num_clust, r_num_clust, num_clust2, r_num_clust_2, num_clust3, r_num_clust_3]

        print 'cleaning memory ....'
        gc.collect()

        print 'Getting best CH ...'
        CH_result = []
        CHmaxI = calisnki.index(max(calisnki)) + min_nc
        CHmax = max(calisnki)
        calisnki = np.array(calisnki)
        CHmax2I = calisnki.argsort()[-2:][0] + min_nc
        CHmax3I = calisnki.argsort()[-3:][0] + min_nc
        CHmax2 = calisnki[CHmax2I - min_nc]
        CHmax3 = calisnki[CHmax3I - min_nc]
        CH_result += [CHmaxI, CHmax, CHmax2I, CHmax2, CHmax3I, CHmax3]

        print 'cleaning memory ....'
        gc.collect()

        results = np.column_stack((knee, silhouette, np.array(stds), np.array(armonicas),
                                   np.array(armonicas_1), calisnki, origW, origLogW, gaps, ElogW, GapSdSk, Sk, Sd))

        result[method] = results

        best_final = np.concatenate(
            (np.array([sw_result]), np.array([SSE_result]), np.array([CH_result]), np.array([gaps_r])), axis=0)

        best_res.ix[method] = best_final

        gap_results_2 = result[method][["OrigWk", "OrigLogWk", "Gap", "ElogW", "GapSdSk"]].astype(float)

        print 'cleaning memory ....'
        gc.collect()

        print 'Saving KNEE plot ...'
        plot_knee(max_nc, min_nc, l, num_clust, period, method, path_labels)

        print 'Saving GAP plot ...'
        plot_gap(max_nc, min_nc, gap_optimal, gap_results_2, period, method, path_labels)

        print 'Saving Silhouette plot ...'
        plot_silhouette(max_nc, min_nc, SmaxI, silhouette, period, method, path_labels)

        print 'Saving CH plot ...'
        plot_ch(max_nc, min_nc, CHmaxI, calisnki, period, method, path_labels)

        print 'cleaning memory ....'
        gc.collect()

    best_res.to_csv(path_labels + 'clustering_best_' + period + '.csv', header=True, index=True)
    result.to_csv(path_labels + 'clustering_evaluation_' + period + '.csv', header=True, index=True)

    print 'fim ...............................................................'

    return best_res
