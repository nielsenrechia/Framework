# encoding=utf8
"""Framework script"""
import sys
# from dateutil import parser
# from utils.bigquery import return_bq_query
from dataPlot import plot_dendrogram, plot_churn_rate, plot_cluster_distribution, plot_results
from dataPreprocess import remove_natives_pkgs, apps_matrix, select_by_min_usage, discretization, get_most_used_pkgs
from association_rules import association_rules
from clustering import barcodes_distance, hac_clustering_barcodes
import pandas as pd
from datetime import datetime as dt
import numpy as np
import gc
from matplotlib import pyplot as plt
from monitoring import get_all_groups_barcodes, get_all_behaviors_barcodes, put_same_behaviors_together, \
    get_similar_behaviors_for_less_weeks_behaviors, churn_prediction, monic_external_transistions
from utils.extra_functions import add_barcodes_to_labels


def query_data(table, bars, natives, start_date=None, end_date=None):
    if bars:
        query = """                 
                SELECT
                  *
                FROM
                  TABLE
                WHERE
                  end_timestamp >= 'START_DATE'
                  AND end_timestamp < 'END_DATE'
                  AND pkgs not in NATIVES
                  AND barcodes in BARS
                  -- GROUP BY barcodes, pkgs, start_timestamp, duration_sec, end_timestamp 
                  -- ORDER BY barcodes, start_timestamp
                """
        query = query.replace('BARS', bars)

    else:
        query = """                  
                SELECT
                  *
                FROM
                  TABLE
                WHERE
                  end_timestamp >= 'START_DATE'
                  AND end_timestamp < 'END_DATE'
                  AND pkgs not in NATIVES
                -- GROUP BY barcodes, pkgs, start_timestamp, duration_sec, end_timestamp
                """
    query = query.replace('START_DATE', start_date)
    query = query.replace('END_DATE', end_date)
    query = query.replace('TABLE', table)
    query = query.replace('NATIVES', natives)

    return return_bq_query(query)


def main():
    """Image configuration."""
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    """ Configuration to extract data from BigQuery"""
    startDate = '2017-06-05'
    endDate = '2017-10-23'
    frequency = '7D'
    dates = pd.date_range(startDate, endDate, freq=frequency)
    table = '[motorola.com:sandbox:nielsen.appusage_athene_170605_to_171022_finished]'
    model = 'athene'
    country = 'BR'
    language = 'en'
    natives = "('com.android.systemui', 'com.google.android.packageinstaller', 'com.android.packageinstaller', " \
              "'android', 'com.motorola.setup', 'com.motorola.storageoptimizer', 'com.motorola.motocit', " \
              "'com.motorola.android.provisioning', 'com.google.android.setupwizard', 'com.android.stk')"
    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c')"
    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c', 'ied733fe5ec38ee9f11931fe38d98be89865b33751')"
    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c', 'ied733fe5ec38ee9f11931fe38d98be89865b33751'," \
    #        "'ie98f1a318a84ce430fff87f5b21354bbe7dc4e4e8', 'ie0a5b3c3aeff2d1cbd9274aed1816770052667115'," \
    #        "'ie7574c01a7812d2ac12043abc50a3e4c2089294ea', 'ie6edd41b6fb2e2a7feee6a02c879160e2f4b89510'," \
    #        "'iedb938860c998fdc4ffb6b72146b75c3dc688fd2f', 'ieaed3d729b4638a80c9029df3203a1fd8aa693663')"
    bars = None

    """ Configuration to data preprocess"""
    thresholds = [0.92, 0.97]
    minTime = 70.
    minPercent = 0.01
    mostUsedPercent = 0.01
    popularsPercent = 0.1
    discretization_type = ['ip'] #['frequency', 'ip', 'clustering']
    selection_type = ['1%']

    """ Configuration to data clustering"""
    methods = ['ward']
    # methods = ['complete', 'single', 'average', 'weighted']
    max_nc = 30
    min_nc = 2
    Brefs = 20
    res = ['sse', 'sw', 'std', 'armonica', 'armonica_1', 'ch', 'OrigWk', 'OrigLogWk', 'Gap', 'ElogW', 'GapSdSk', 'Sk', 'Sd']
    best_res = ['1K', 'res1', '2K', 'res2', '3K', 'res3']
    metrics = ['SW', 'KNEE', 'CH', 'GAP']
    col = []
    col_best = []
    for i in methods:
        col += i.split(' ') * len(res)
        col_best += i.split(' ') * len(metrics)
    k = list(xrange(min_nc, max_nc+1))
    col = [col, res * len(methods)]
    col_best = [col_best, metrics * len(methods)]
    c_result = pd.DataFrame(index=k, columns=col)
    best_result = pd.DataFrame(index=col_best, columns=best_res)
    clusters = [6, 5, 5, 4, 7, 5, 4, 7, 5, 5]

    """Paths to save/read data"""
    path_labels = 'results/labels/'
    path_linkage = 'results/linkage/'
    path_rules = 'results/rules/'
    path_matrix = 'results/matrix/'
    path_discretization = 'results/discretization/'
    path_most_used = 'results/most_used/'
    path_popular = 'results/populars/'
    path_most_used_data = 'results/most_used_data/'
    path_distances = 'results/distances/'
    path_outliers = 'results/outliers/'
    path_dendrograms = 'results/dendrograms/'
    path_distribution = 'results/cluster_distribution/'

    """ Configuration of stages to be executed """
    BQ = False
    AR = False
    DM = False
    CL = False
    PL = False

    for d in xrange(len(dates) - 1):
        t1 = dt.now()
        start_date = dates[d]
        end_date = dates[d + 1]
        period = str(start_date.date()) + '_' + str(end_date.date())

        if BQ:
            print "BigQuery sql ..."
            hours = pd.date_range(start_date, end_date, freq='4H')

            for h in xrange(len(hours) - 1):
                start_hour = str(hours[h])
                end_hour = str(hours[h + 1])

                if h == 0:
                    X, bq_rows = query_data(table, bars, natives, start_hour, end_hour)
                    gc.collect()
                else:
                    Xn, bq_rows = query_data(table, bars, natives, start_hour, end_hour)
                    X = pd.concat([X, Xn])
                    gc.collect()
                    del Xn
                    gc.collect()
                print str(h)

            t2 = dt.now()
            print "BigQuery executed in " + str((t2-t1).total_seconds()/60.) + " minutes at " + str(t2) + " ..."
            del t1, start_date, end_date, hours, start_hour, end_hour, bq_rows
            gc.collect()

            # print "Removing antives pkgs ..."
            # X = remove_natives_pkgs(X, natives)

            print 'total records is ' + str(X.shape[0]) + ' ...'
            pkgs = X['pkgs'].unique()
            num_pkgs = pkgs.shape[0]
            print 'all pkgs ' + str(num_pkgs) + ' ...'
            barcodes = X['barcodes'].unique()
            num_barcodes = barcodes.shape[0]
            print 'barcodes ' + str(num_barcodes) + ' ...'
            gc.collect()
            del num_pkgs, num_barcodes
            gc.collect()

            print "groupping by ..."
            X = X.groupby(['barcodes', 'pkgs'], as_index=False).agg({'duration_sec': 'sum'})
            gc.collect()

            # print "best apps selection ..."
            # plot_best_apps(X, 'duration_sec', language, num_pkgs, thresholds)

            print "most used data first..."
            most_used_pkgs, X = get_most_used_pkgs(X, barcodes, pkgs, mostUsedPercent)
            gc.collect()

            # print "getting pkgs summary ..."
            # pkgs_summary = pd.DataFrame(X.groupby('pkgs').agg(
            #     {'duration_sec':
            #          {'count': 'count',
            #           'mean': 'mean',
            #           'std': 'std',
            #           'min': 'min',
            #           '25%': lambda x: np.percentile(x, 25),
            #           '50%': lambda x: np.percentile(x, 50),
            #           '75%': lambda x: np.percentile(x, 75),
            #           'max': 'max',
            #           '10%': lambda x: np.percentile(x, 10)
            #     }}))
            #
            # pkgs_summary.columns = pkgs_summary.columns.droplevel(0)
            pkgs_summary = None
            # pkgs_summary.to_csv("results/pkgs_summary_testando.csv")

            print "most important apps selection ..."
            X = select_by_min_usage(X, barcodes, pkgs_summary, minTime, minPercent)
            gc.collect()

            print "apps to matrix ..."
            matrix, most_used_pkgs, populars_pkgs, X = apps_matrix(X, barcodes, pkgs, mostUsedPercent, popularsPercent,
                                                                   'duration_sec')
            gc.collect()

            matrix.to_csv(path_matrix + 'matrix_' + period + '.csv.gz', index=True, header=True, compression='gzip')
            most_used_pkgs.to_csv(path_most_used + 'most_used_' + period + '.csv.gz', index=True, header=True,
                                  compression='gzip')
            populars_pkgs.to_csv(path_popular + 'populars_' + period + '.csv.gz', index=True, header=True,
                                 compression='gzip')
            X.to_csv(path_most_used_data + 'most_used_data_' + period + '.csv.gz', index=True, header=True,
                     compression='gzip')

            del X
            gc.collect()

            print "discretization ..."
            # teste = plot_discretization(matrix, populars_pkgs, most_used_pkgs, barcodes, discretization)
            summary, discratization = discretization(matrix, populars_pkgs,  most_used_pkgs, discretization_type,
                                                     barcodes)
            gc.collect()
            discratization.to_csv(path_discretization + 'discretization_' + period + '.csv.gz', index=True,
                                  header=True, compression='gzip')
            summary.to_csv(path_discretization + 'summary_' + period + '.csv.gz', index=True, header=True,
                           compression='gzip')
            # z = 0

            del most_used_pkgs
            del populars_pkgs
            del matrix
            gc.collect()

        if AR:
            gc.collect()
            print "association rules ..."
            discratization = pd.read_csv(path_discretization + 'discretization_' + period + '.csv.gz', index_col=0,
                                         header=0, compression='gzip')
            gc.collect()
            itemsets = association_rules(discratization)
            gc.collect()
            itemsets.to_csv(path_rules + 'itemsets2_' + period + '.csv.gz', index=True, header=True, compression='gzip')

        if DM:
            discratization = pd.read_csv(path_discretization + 'discretization_' + period + '.csv.gz', index_col=0,
                                         header=0, compression='gzip')
            itemsets = pd.read_csv(path_rules + 'itemsets_' + period + '.csv.gz', index_col=0,
                                         header=0, compression='gzip')
            print "distance matrix ..."
            outliers, distances = barcodes_distance(itemsets, discratization)
            gc.collect()

            del discratization
            gc.collect()
            distances.to_csv(path_distances + 'distances_' + period + '.csv.gz', index=False, header=False,
                             compression='gzip')
            pd.DataFrame(outliers).to_csv(path_outliers + 'outliers_' + period + '.csv.gz', index=False, header=False,
                                          compression='gzip')

        if CL:
            print "clustering ..."

            matrix = pd.read_csv(path_matrix + 'matrix_' + period + '.csv.gz', index_col=0, header=0,
                                 compression='gzip', nrows=None)
            outliers = pd.read_csv(path_outliers + 'outliers_' + period + 'csv.gz', index_col=None, header=None)
            matrix = matrix[~matrix.index.isin(outliers[0])]
            distances = pd.read_csv(path_distances + 'distances_' + period + '.csv.gz', index_col=None, header=None,
                                    compression='gzip', nrows=None)
            x = distances.values
            results = hac_clustering_barcodes(matrix.values, distances, methods, max_nc, min_nc, Brefs, path_labels, period,
                                    path_linkage, c_result, best_result)

            gc.collect()
        #
        if PL:
            for method in methods:
                l = pd.read_csv(path_linkage + 'linkage_y_' + period + '_' + method + '.csv.gz', index_col=None, header=None)
                plot_dendrogram(l, path_dendrograms, period, method, clusters[d])
            if d == 9:
                z = 0

    barcodes = pd.read_csv('results/barcodes_170605_to_171022_churners.csv', index_col=0, header=0, parse_dates=['last_day'])

    # barcodes['last_day'] = pd.to_datetime(barcodes['last_day'])
    # num_bar = barcodes.shape[0]
    # plot_churn_rate(barcodes, dates, num_bar)

    # add_barcodes_to_labels(path_labels, path_discretization, path_outliers, dates, clusters):
    # plot_cluster_distribution(path_labels, path_outliers, dates, clusters, barcodes, path_distribution)

    # all_labels = get_all_groups_barcodes(path_labels, path_outliers, dates, methods[0], clusters, barcodes)
    # all_behaviors = get_all_behaviors_barcodes(all_labels)

    # all_behaviors = pd.read_csv('results/all_behaviors_barcodes_without_first_week.csv', index_col=0, header=0)
    #
    # all_behaviors_together = put_same_behaviors_together(all_behaviors)
    # get_similar_behaviors_for_less_weeks_behaviors(all_behaviors_together)

    for d in xrange(len(dates) - 1):
        print d
        t1 = dt.now()
        start_date = dates[d]
        end_date = dates[d + 1]
        end_date_2 = dates[d + 2]
        periodX = str(start_date.date()) + '_' + str(end_date.date())
        periodY = str(end_date.date()) + '_' + str(end_date_2.date())
        all_labels = pd.read_csv('results/all_labels_barcodes.csv', index_col=0)
        objetcsX = all_labels[periodX]
        objectsY = all_labels[periodY]

        trashold = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        trasholdSplit = [0.2, 0.25, 0.3, 0.35, 0.4]
        variations = ['absorptions', 'survivals', 'deads', 'splits']

        results = pd.DataFrame(index=pd.MultiIndex.from_product([trashold, trasholdSplit], names=['trashold', 'trasholdSplit']),
                               columns=pd.MultiIndex.from_product([dates[:10].date, variations],names=['dates', 'variations']))
        for t in trashold:
            for ts in trasholdSplit:
                absorptionList, survivallist, deadList, splitList = monic_external_transistions(t, ts, objetcsX, objectsY)
                a = []
                su = []
                sp = []
                if absorptionList:
                    for l in absorptionList:
                        a += [l[0]]
                if survivallist:
                    for l in survivallist:
                        su += [l[0]]
                if splitList:
                    for l in splitList:
                        sp += [l[0]]

                results.loc[t, ts][dates[d].date()] = [float(len(set(a))), float(len(set(su))), float(len(deadList)), float(len(set(sp)))]
    results.to_csv('results/variations_by_trasholds.csv', index=True, columns=True)


        # plot_results(absorptionList, survivallist, deadList, splitList)

    # all_behaviors_together = pd.read_csv('results/final_behaviors_together_qtd.csv', index_col=0, header=0)
    #
    # churn_prediction(all_behaviors, all_behaviors_together)
    #
    # z = 0


if __name__ == '__main__':
    sys.exit(main())