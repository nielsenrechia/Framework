# encoding=utf8
import pandas as pd
import numpy as np
from math import ceil
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from dataPlot import plot_best_apps_selection
from matplotlib import pyplot as plt
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.rlike.container as rlc
import gc
from scipy.spatial.distance import pdist, squareform, is_valid_y, is_valid_dm, num_obs_y, num_obs_dm
from scipy.sparse import csr_matrix


# from rpy2.robjects.packages import importr


def remove_natives_pkgs(X, natives):
    X = X[~X['pkgs'].isin(natives)]
    gc.collect()
    return X


def select_by_min_usage(X, barcodes, pkgs_summary, minTime, minPercent):
    # old_mean = 0.
    # new_mean = 0.
    #
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    # fig2, ((ax11, ax12), (ax13, ax14), (ax15, ax16)) = plt.subplots(3, 2)
    #
    # w = X[X['pkgs'] == 'com.whatsapp']['duration_sec']
    # f = X[X['pkgs'] == 'com.facebook.katana']['duration_sec']
    # g = X[X['pkgs'] == 'com.android.chrome']['duration_sec']
    # wmin = w.min()
    # wmax = w.max()
    # fmin = f.min()
    # fmax = f.max()
    # gmin = g.min()
    # gmax = g.max()
    #
    # t1, t4, t7 = plot_best_apps_selection(X, ax1, ax4, ax7, wmin, wmax, fmin, fmax, gmin, gmax)
    #
    # ax1.xaxis.label.set_visible(False)
    # ax1.set_ylabel('Whatsapp')
    # ax2.xaxis.label.set_visible(False)
    # ax4.set_ylabel('Facebook App')
    # plt.setp(ax2.get_yticklabels(), visible=False)
    # plt.setp(ax3.get_yticklabels(), visible=False)
    # ax7.set_ylabel('Chrome')
    # ax3.xaxis.label.set_visible(False)
    # ax4.xaxis.label.set_visible(False)
    # plt.setp(ax5.get_yticklabels(), visible=False)
    # ax5.xaxis.label.set_visible(False)
    # plt.setp(ax6.get_yticklabels(), visible=False)
    # ax6.xaxis.label.set_visible(False)
    # plt.setp(ax8.get_yticklabels(), visible=False)
    # plt.setp(ax9.get_yticklabels(), visible=False)
    # plt.subplots_adjust(wspace=.045, hspace=.312, top=0.964, bottom=0.09, right=0.99, left=0.059)
    # #
    Xn = X.copy()
    Xnn = X.copy()
    # Xdrop = pd.DataFrame()
    # Xndrop = pd.DataFrame()
    barcodes_summary = pd.DataFrame(np.zeros((len(barcodes), 4)), index=barcodes,
                                    columns=[['pkgs_before', 'pkgs_after_1%', 'pkgs_after_25%', 'pkgs_after_10%']])
    print 'selecting best apps by 1% ...'
    for barcode in barcodes:
        data = X[X['barcodes'] == barcode]
        # pkgs = data['pkgs'].unique()
        barcodes_summary.at[barcode, 'pkgs_before'] = data.shape[0]
        # old_mean += pkgs.shape[0]
        total_usage = data['duration_sec'].sum()
        t1 = total_usage * minPercent
        apps_to_remove = data.groupby(['pkgs']).sum()
        # s = apps_to_remove[(apps_to_remove['duration_sec'] < minTime) |
        #                    (apps_to_remove['duration_sec'] < t1)].index
        apps_to_remove = X[(X['barcodes'] == barcode) & (X['pkgs'].isin(
            apps_to_remove[(apps_to_remove['duration_sec'] < minTime) |
                           (apps_to_remove['duration_sec'] < t1)].index))].index
        # tt = X.loc[apps_to_remove]

        # Xdrop = pd.concat([Xdrop, X.loc[apps_to_remove]])
        X.drop(apps_to_remove, inplace=True)
        new_data = X[X['barcodes'] == barcode]
        # new_mean += new_data['pkgs'].unique().shape[0]
        barcodes_summary.at[barcode, 'pkgs_after_1%'] = new_data.shape[0]
    # barcodes_summary.to_csv('barcodes_summary_teste_1%.csv')

    # t2, t5, t8 = plot_best_apps_selection(X, ax2, ax5, ax8, wmin, wmax, fmin, fmax, gmin, gmax)
    #
    # wdrop = Xdrop[Xdrop['pkgs'] == 'com.whatsapp']['duration_sec']
    # fdrop = Xdrop[Xdrop['pkgs'] == 'com.facebook.katana']['duration_sec']
    # gdrop = Xdrop[Xdrop['pkgs'] == 'com.android.chrome']['duration_sec']
    # wmindrop = wdrop.min()
    # wmaxdrop = wdrop.max()
    # fmindrop = fdrop.min()
    # fmaxdrop = fdrop.max()
    # gmindrop = gdrop.min()
    # gmaxdrop = gdrop.max()

    # t11, t13, t15 = plot_best_apps_selection(Xdrop, ax11, ax14, ax15, wmindrop, wmaxdrop, fmindrop, fmaxdrop, gmindrop,
    #                                          gmaxdrop)

    # print 'selecting best apps by 25% ...'
    # # barcodes_summary2 = pd.DataFrame(np.zeros((len(barcodes), 2)), index=barcodes,
    # #                                  columns=[['pkgs_before', 'pkgs_after']])
    #
    # for barcode in barcodes:
    #     apps_to_remove = []
    #     data = Xn[Xn['barcodes'] == barcode]
    #     # barcodes_summary.at[barcode, 'pkgs_before'] = data.shape[0]
    #     pkgs = data['pkgs'].unique()
    #     # old_mean += pkgs.shape[0]
    #
    #     for pkg in pkgs:
    #         if data[data['pkgs'] == pkg]['duration_sec'].values[0] < pkgs_summary.at[pkg, '25%']:
    #             apps_to_remove += [data[data['pkgs'] == pkg].index.values[0]]
    #
    #     # Xndrop = pd.concat([Xndrop, Xn.loc[apps_to_remove]])
    #     Xn.drop(apps_to_remove, inplace=True)
    #     new_data = Xn[Xn['barcodes'] == barcode]
    #     # new_mean += new_data['pkgs'].unique().shape[0]
    #     barcodes_summary.at[barcode, 'pkgs_after_25%'] = new_data.shape[0]
    #
    # print 'selecting best apps by 10% ...'
    # for barcode in barcodes:
    #     apps_to_remove = []
    #     data = Xnn[Xnn['barcodes'] == barcode]
    #     # barcodes_summary.at[barcode, 'pkgs_before'] = data.shape[0]
    #     pkgs = data['pkgs'].unique()
    #     # old_mean += pkgs.shape[0]
    #
    #     for pkg in pkgs:
    #         if data[data['pkgs'] == pkg]['duration_sec'].values[0] < pkgs_summary.at[pkg, '10%']:
    #             apps_to_remove += [data[data['pkgs'] == pkg].index.values[0]]
    #
    #     # Xndrop = pd.concat([Xndrop, Xn.loc[apps_to_remove]])
    #     Xnn.drop(apps_to_remove, inplace=True)
    #     new_data = Xnn[Xnn['barcodes'] == barcode]
    #     # new_mean += new_data['pkgs'].unique().shape[0]
    #     barcodes_summary.at[barcode, 'pkgs_after_10%'] = new_data.shape[0]
    #
    # # Xn.reset_index(inplace=True, drop=True)
    # # old_mean = old_mean / barcodes.shape[0]
    # # print 'old mean ' + str(old_mean) + ' ...'
    # #
    # # new_mean = new_mean / Xn['barcodes'].unique().shape[0]
    # # print 'new mean ' + str(new_mean) + ' ...'
    # #
    # # t3, t6, t9 = plot_best_apps_selection(Xn, ax3, ax6, ax9, wmin, wmax, fmin, fmax, gmin, gmax)
    # # t12, t14, t16 = plot_best_apps_selection(Xndrop, ax12, ax14, ax16, wmindrop, wmaxdrop, fmindrop, fmaxdrop, gmindrop,
    # #                                          gmaxdrop)
    # #
    # # ax1.set_title("Raw")
    # # ax2.set_title("1% selection")
    # # ax3.set_title("25% selection")
    # #
    # # fig.savefig('temp.png', bbox_inches='tight', pad_inches=0)
    # barcodes_summary.to_csv('barcodes_summary_teste_all_.csv')
    # plt.show()
    # z = 0
    gc.collect()
    return X


def get_most_used_pkgs(X, barcodes, pkgs, mostUsedPercent):
    pkgs = pd.DataFrame(pkgs, columns=['pkgs'])
    # Get costumer and percent for each app
    print 'Get costumer and percent for each app '
    for i, app in pkgs.iterrows():
        unique = float(X[X['pkgs'] == app[0]]['barcodes'].unique().shape[0])
        pkgs.set_value(i, 'customers', unique)
        pkgs.set_value(i, 'percent', round(unique / barcodes.shape[0], 2))

    # Most Used
    print 'get most used apps'
    most_used_pkgs = pkgs[pkgs['percent'] >= mostUsedPercent]
    most_used_pkgs.reset_index(drop=True, inplace=True)

    # most_used_pkgs.to_csv('most_used_antes_teste.csv')
    # most_used_pkgs.to_csv(path_result + 'most_used_' + filename, header=True, index_label='pkgs', index=False,
    #               quoting=csv.QUOTE_NONE,
    #               escapechar='\\')

    print 'get most used data to generate matrix'
    # most used data to generate matrix
    most_used_data = X[X['pkgs'].isin(most_used_pkgs['pkgs'].values.tolist())]

    gc.collect()
    return most_used_pkgs, most_used_data


def apps_matrix(X, barcodes, pkgs, mostUsedPercent, popularPercent, select_data):

    pkgs = pd.DataFrame(pkgs, columns=['pkgs'])
    # Get costumer and percent for each app
    ####### pegar os mais usados filtrados antes da criação do summario estatistico que continuaram #####
    print 'Get costumer and percent for each app '
    for i, app in pkgs.iterrows():
        unique = float(X[X['pkgs'] == app[0]]['barcodes'].unique().shape[0])
        pkgs.set_value(i, 'customers', unique)
        pkgs.set_value(i, 'percent', round(unique/barcodes.shape[0], 2))

    # Most Used
    print 'get most used apps'
    most_used_pkgs = pkgs[pkgs['percent'] >= mostUsedPercent]
    most_used_pkgs.reset_index(drop=True, inplace=True)
    most_used_pkgs.to_csv('most_used_teste.csv')
    # most_used_pkgs.to_csv(path_result + 'most_used_' + filename, header=True, index_label='pkgs', index=False,
    #               quoting=csv.QUOTE_NONE,
    #               escapechar='\\')
    ##################################### AQUIIIII ############################################################

    # Popular apps
    print 'get popular apps'
    popular_pkgs = pkgs[pkgs['percent'] >= popularPercent]
    popular_pkgs.reset_index(drop=True, inplace=True)
    popular_pkgs.to_csv('popular_teste.csv')
    # popular_pkgs.to_csv(path_result + 'populars_' + filename, header=True, index_label='pkgs', index=False,
    #                       quoting=csv.QUOTE_NONE,
    #                       escapechar='\\')

    print 'get most used data to generate matrix'
    # most used data to generate matrix
    most_used_data = X[X['pkgs'].isin(most_used_pkgs['pkgs'].values.tolist())]

    pkgs = pd.Series(most_used_data['pkgs'].unique())
    barcodes = most_used_data['barcodes'].unique()
    pkgs.reset_index(drop=True, inplace=True)

    matrix = pd.DataFrame(np.zeros((barcodes.shape[0], pkgs.shape[0]), dtype=np.float32), index=barcodes, columns=pkgs)
    print barcodes.shape[0], 'barcodes'
    print pkgs.shape[0], 'pkgs'
    print barcodes.shape[0] * pkgs.shape[0], 'total elements'
    print 'Processing matrix entries...'

    # generate matrix
    for barcode in barcodes:
        sum_by_pkg = most_used_data[most_used_data['barcodes'] == barcode][['pkgs', select_data]].groupby(
            'pkgs', as_index=False).sum()
        matrix.loc[barcode][sum_by_pkg['pkgs']] = sum_by_pkg[select_data]
        z = 0

    matrix.to_csv('results/matrix_teste.csv')
    # matrix.to_csv(path_result + 'matrix_' + filename, header=True, index_label='barcodes', index=True, quoting=csv.QUOTE_NONE,
    #               escapechar='\\')

    gc.collect()
    return matrix, most_used_pkgs, popular_pkgs, most_used_data


def discretization(matrix, populars_pkgs, most_used_pkgs, discretization_type, barcodes, plot=None):
    # discretization
    populars_pkgs = populars_pkgs['pkgs'].values.tolist()
    most_used_pkgs = most_used_pkgs['pkgs'].values.tolist()
    n_objects, n_attributes, = matrix.shape
    print str(n_objects) + ' barcodes and ' + str(n_attributes) + " pkgs ..."

    for d in discretization_type:

        if d == 'frequency':

            if plot is None:
                resume = pd.DataFrame(index=np.asarray(populars_pkgs),
                                      columns=['barcodes', 'percent', 'intervals', 'barcodes_per_group', '1', '2', '3',
                                               '4', '5', '6', '7', '8', '9', '10'])
            print 'equal frequency discretization (until 10 intervals) approach ...'
            frequency_matrix = matrix.copy()
            for pkg in populars_pkgs:
                print 'Division to ' + str(pkg) + ' ...'
                instances = matrix[matrix[pkg] > 0][pkg]
                instances = instances.sort_values()
                total_instances = len(instances)
                barcodes_pkg = instances.index.values
                instances = instances.values.reshape(-1, 1)
                k = int(ceil(float(total_instances) / float(len(barcodes) * 0.1)))
                abs_number = abs(total_instances / 10)
                rest = total_instances
                labels = []

                if plot is None:
                    resume.set_value(pkg, 'barcodes', total_instances)
                    resume.set_value(pkg, 'percent', float(total_instances) / float(len(barcodes)))
                    resume.set_value(pkg, 'intervals', k)
                    resume.set_value(pkg, 'barcodes_per_group', str(abs_number))

                for i in xrange(k):
                    if i + 1 == k:
                        labels.extend([i+1 for j in range(rest)])
                    else:
                        labels.extend([i+1 for j in range(abs_number)])
                        rest -= abs_number

                print 'Concat instances and labels ...'
                instances_and_labes = np.hstack((instances, np.asarray(labels).reshape(-1, 1)))
                new_pkg = pd.DataFrame(instances_and_labes, index=barcodes_pkg, columns=[pkg, 'labels'], dtype=np.float32)

                for i in xrange(k):
                    x = new_pkg[new_pkg['labels'] == i+1]
                    min = np.min(new_pkg[new_pkg['labels'] == i+1][pkg])
                    max = np.max(new_pkg[new_pkg['labels'] == i+1][pkg])

                    if plot is None:
                        resume.set_value(pkg, str(i+1), [min, max])

                # new_pkg.to_csv(path_discretization + '_new_' + pkg + '_' + matrix_filename, header=True, index_label='teste', index=True,
                              # quoting=csv.QUOTE_NONE, escapechar='\\')

                print 'new data frame to pkg divided in the same column instead new features'
                for l in np.unique(labels):
                    frequency_matrix.ix[new_pkg[new_pkg['labels'] == l].index, pkg] = str(str(l) + '_' + str(k))
                    z = 0

            if plot is None:
                for pkg in most_used_pkgs:
                    if pkg not in populars_pkgs:
                        frequency_matrix.loc[frequency_matrix[pkg] != 0, pkg] = '1_1'

                return resume, frequency_matrix
            else:
                frequency_matrix = frequency_matrix[populars_pkgs]
                z = 0

            # new_matrix.to_csv(path_discretization + 'discretization_' + matrix_filename, header=True, index_label='barcodes',
            #                   index=True, quoting=csv.QUOTE_NONE, escapechar='\\')
            #
            # resume.to_csv(path_discretization + 'resume_discretization_by_frequency(2_to_10).csv', header=True, index=True, index_label='popular pkgs')

        if d == 'clustering':

            if plot is None:
                resume = pd.DataFrame(index=np.asarray(populars_pkgs),
                                      columns=['barcodes', 'percent', 'sw1', 'sw2', 'sw3', 'ch1', 'ch2', 'ch3'])

            print 'Kmeans discretization approach ...'
            for pkg in populars_pkgs:
                print 'Division to ' + str(pkg) + ' ...'
                instances = matrix[matrix[pkg] > 0][pkg]
                instances = instances.sort_values()
                total_instances = len(instances)
                barcodes_pkg = instances.index.values
                instances = np.log10(instances.values.reshape(-1, 1))
                max_nc = 15
                # max_nc = int(np.sqrt(len(instances)+1))/2
                print 'max number of cluster is ' + str(max_nc) + ' ...'

                # seeds
                np.random.seed(0)
                max_seed = 30

                all_labels = []
                all_centroids = []
                all_inertia = []
                header = []
                silhouette = []
                ch = []

                resume.set_value(pkg, 'barcodes', total_instances)
                resume.set_value(pkg, 'percent', float(total_instances) / float(len(barcodes)))

                for nc in xrange(2, max_nc + 1):
                    print 'Kmeans clustering with k = ' + str(nc) + ' ...'
                    header += [nc]
                    inst = KMeans(n_clusters=nc, max_iter=100, n_init=max_seed, init='k-means++', n_jobs=1)
                    all_labels += [inst.fit_predict(instances)]
                    all_centroids += [inst.cluster_centers_]
                    all_inertia += [inst.inertia_]

                for nc in xrange(2, max_nc + 1):
                    instances = np.float32(instances)
                    silhouette += [silhouette_score(instances, all_labels[nc - 2], metric='euclidean')]
                    print 'Silhouette score for k = ' + str(nc) + ' is ' + str(silhouette[nc - 2]) + ' ...'
                    ch += [calinski_harabaz_score(instances, all_labels[nc - 2])]
                    print 'CH score for k = ' + str(nc) + ' is ' + str(ch[nc - 2]) + ' ...'

                swr = np.asarray(silhouette)
                sw1 = np.argsort(swr)[-1]+2
                sw2 = np.argsort(swr)[-2]+2
                sw3 = np.argsort(swr)[-3]+2
                resume.set_value(pkg, 'sw1', sw1)
                resume.set_value(pkg, 'sw2', sw2)
                resume.set_value(pkg, 'sw3', sw3)
                chr = np.asarray(ch)
                ch1 = np.argsort(chr)[-1]+2
                ch2 = np.argsort(chr)[-2]+2
                ch3 = np.argsort(chr)[-3]+2
                resume.set_value(pkg, 'ch1', ch1)
                resume.set_value(pkg, 'ch2', ch2)
                resume.set_value(pkg, 'ch3', ch3)

            # resume.to_csv(path_discretization + 'resume_by_new_kmeans_' + matrix_filename, header=True, index=True, index_label='popular pkgs')

        if d == 'ip':

            if plot is None:
                resume = pd.DataFrame(index=np.asarray(populars_pkgs),
                                      columns=['barcodes', 'percent','min', 'max', 'low', 'high', 'msd', 'low_down',
                                               'high_up', 'result', 'intervals', 'covers_min', 'covers_max', '1', '2',
                                               '3', '4', '5', '6'])
            print 'discretization by intuitiuve partitioning ...'
            ip_matrix = matrix.copy()
            for pkg in populars_pkgs:
                print 'Division to ' + str(pkg) + ' ...'
                instances = matrix[matrix[pkg] > 0][pkg]
                instances = instances.sort_values()
                total_instances = len(instances)
                barcodes_pkg = instances.index.values
                instances = instances.values.reshape(-1, 1)
                v_min = np.min(instances)
                v_max = np.max(instances)
                low = np.percentile(a=instances, q=5)
                high = np.percentile(a=instances, q=95)
                msd = float(pow(10, len(str(int(high)))-1))
                low_down = 0
                high_up = float(pow(10, len(str(int(high)))-1) * (np.floor(high / (10 ** np.floor(np.log10(high)))) + 1))
                result = (high_up - low_down)/msd
                cut_points = []

                if plot is None:
                    resume.set_value(pkg, 'barcodes', total_instances)
                    resume.set_value(pkg, 'percent', float(total_instances) / float(len(barcodes)))
                    resume.set_value(pkg, 'min', v_min)
                    resume.set_value(pkg, 'max', v_max)
                    resume.set_value(pkg, 'low', low)
                    resume.set_value(pkg, 'high', high)
                    resume.set_value(pkg, 'msd', msd)
                    resume.set_value(pkg, 'low_down', low_down)
                    resume.set_value(pkg, 'high_up', high_up)
                    resume.set_value(pkg, 'result', result)

                if result == 3.0 or result == 6.0 or result == 9.0 or result == 7:
                    intervals = 3
                    if result == 7:
                        width = float(pow(10, len(str(int(high_up / 3)))-1) * (np.floor((high_up / 3)/ (10 ** np.floor(np.log10(high_up / 3))))))
                        z = 0
                    else:
                        width = high_up / 3
                elif result == 2.0 or result == 4.0 or result == 8.0:
                    intervals = 4
                    width = high_up / 4
                elif result == 1.0 or result == 5.0 or result == 10.0:
                    intervals = 5
                    width = high_up / 5
                else:
                    print "nenhuma das opções"

                if low_down > v_min:
                    print "create new range down"
                    cut_points += [v_min]
                    intervals += 1
                    if plot is None:
                        resume.set_value(pkg, 'covers_min', 'no, + 1 interval down')
                else:
                    cut_points = [0]
                    if plot is None:
                        resume.set_value(pkg, '1', str([0, width]))
                        resume.set_value(pkg, 'covers_min', 'YES')

                for n in xrange(1, int(intervals) + 1):
                    if result == 7 and n == 2:
                        cut_points += [cut_points[n - 1] + width + width/2]

                        if plot is None:
                            resume.set_value(pkg, str(n), str([cut_points[n - 1], cut_points[n - 1] + width + width/2]))
                    else:
                        cut_points += [cut_points[n - 1] + width]

                        if plot is None:
                            resume.set_value(pkg, str(n), str([cut_points[n - 1], cut_points[n - 1] + width]))

                if high_up < v_max:
                    cut_points += [v_max]
                    intervals += 1

                    if plot is None:
                        resume.set_value(pkg, str(len(cut_points)-1), str([cut_points[-2], v_max]))
                        print "create new range up"
                        resume.set_value(pkg, 'covers_max', 'no, + 1 interval up')
                else:
                    if plot is None:
                        resume.set_value(pkg, 'covers_max', 'YES')

                if plot is None:
                    resume.set_value(pkg, 'intervals', intervals)
                    # resume.to_csv(path_discretization + 'new_resume_by_ip_' + matrix_filename, header=True,
                    #               index=True,
                    #               index_label='popular pkgs')
                    #

                instances = pd.DataFrame(np.hstack((instances, np.zeros(len(barcodes_pkg)).reshape(-1, 1))),
                                         index=barcodes_pkg, columns=[pkg, 'labels'])

                print 'Concat instances and labels ...'
                for i, cut in enumerate(cut_points[:-1]):
                    instances['labels'][(instances[pkg] > cut) & (instances[pkg] <= cut_points[i + 1])] = str(
                        str(i + 1) + '_' + str(int(intervals)))
                    z = 0

                print 'new data frame to pkg divided in the same column instead new features'
                x = np.unique(instances['labels'].values)
                for l in np.unique(instances['labels'].values):
                    ip_matrix.ix[instances[instances['labels'] == l].index, pkg] = l
                    z = 0
                z = 0

            if plot is None:
                for pkg in most_used_pkgs:
                    if pkg not in populars_pkgs:
                        ip_matrix.loc[ip_matrix[pkg] != 0, pkg] = '1_1'

                # new_matrix.to_csv(path_discretization + 'new_intuitive_partitioning_' + matrix_filename, header=True,
                #                   index_label='barcodes',
                #                   index=True, quoting=csv.QUOTE_NONE, escapechar='\\')
                gc.collect()
                del matrix
                gc.collect()
                return resume, ip_matrix
            else:
                ip_matrix = ip_matrix[populars_pkgs]
                z = 0

    if plot is True:
        return frequency_matrix, ip_matrix

