# encoding=utf8
"""Framework script"""
import sys
# from dateutil import parser
# from utils.bigquery import return_bq_query
# from dataPlot import plot_best_apps, plot_discretization
from dataPreprocess import remove_natives_pkgs, apps_matrix, select_by_min_usage, discretization, get_most_used_pkgs
from association_rules import association_rules
from clustering import barcodes_distance, hac_clustering_barcodes
import pandas as pd
from datetime import datetime as dt
import numpy as np
import gc


# from matplotlib import pyplot as plt


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
    # SMALL_SIZE = 13
    # MEDIUM_SIZE = 18
    # BIGGER_SIZE = 23
    #
    # plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

    """ Configuration to extract data from BigQuery"""
    startDate = '2017-06-05'
    endDate = '2017-10-23'
    frequency = '7D'
    table = '[motorola.com:sandbox:nielsen.appusage_athene_170605_to_171022_finished]'
    model = 'athene'
    country = 'BR'
    language = 'en'
    natives = "('com.android.systemui', 'com.google.android.packageinstaller', 'com.android.packageinstaller', " \
              "'android', 'com.motorola.setup', 'com.motorola.storageoptimizer', 'com.motorola.motocit', " \
              "'com.motorola.android.provisioning', 'com.google.android.setupwizard', 'com.android.stk')"
    dates = pd.date_range(startDate, endDate, freq=frequency)
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
    methods = ['complete', 'ward', 'average', 'weighted']
    max_nc = 30
    k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    col = [['ward', 'ward', 'ward', 'ward', 'ward', 'complete', 'complete', 'complete', 'complete', 'complete',
            'average', 'average', 'average', 'average', 'average', 'single', 'single', 'single', 'single', 'single'],
           ['sse', 'sw', 'std', 'armonica', 'armonica_1', 'sse', 'sw', 'std', 'armonica', 'armonica_1', 'sse', 'sw',
            'std', 'armonica', 'armonica_1', 'sse', 'sw', 'std', 'armonica', 'armonica_1']]
    c_result = pd.DataFrame(index=k, columns=col)

    """ Paths to save/read data"""
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

    """ Configuration of stages to be executed """
    BQ = False
    AR = False
    DM = False
    CL = True

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
            most_used_pkgs.to_csv(path_most_used + 'most_used_' + period + '.csv.gz', index=True, header=True, compression='gzip')
            populars_pkgs.to_csv(path_popular + 'populars_' + period + '.csv.gz', index=True, header=True, compression='gzip')
            X.to_csv(path_most_used_data + 'most_used_data_' + period + '.csv.gz', index=True, header=True, compression='gzip')

            del X
            gc.collect()

            print "discretization ..."
            # teste = plot_discretization(matrix, populars_pkgs, most_used_pkgs, barcodes, discretization)
            summary, discratization = discretization(matrix, populars_pkgs,  most_used_pkgs, discretization_type, barcodes)
            gc.collect()
            discratization.to_csv(path_discretization + 'discretization_' + period + '.csv.gz', index=True, header=True, compression='gzip')
            summary.to_csv(path_discretization + 'summary_' + period + '.csv.gz', index=True, header=True, compression='gzip')
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
            distances.to_csv(path_distances + 'distances_' + period + '.csv.gz', index=False, header=False, compression='gzip')
            pd.DataFrame(outliers).to_csv(path_outliers + 'outliers_' + period + 'csv.gz', index=False, header=False, compression='gzip')

        if CL:
            print "clustering ..."

            distances = pd.read_csv(path_distances + 'distances_' + period + '.csv.gz', index_col=None,
                                         header=None, compression='gzip', nrows=1225)
            x = distances.values
            hac_clustering_barcodes(distances, methods, max_nc, path_labels, period, path_linkage, c_result)

            gc.collect()
            # pkgs = matrix['pkgs'].unique()
            # num_pkgs = pkgs.shape[0]
            # print 'new all pkgs ' + str(num_pkgs) + ' ...'
            # barcodes = matrix['barcodes'].unique()
            # num_barcodes = barcodes.shape[0]
            # print 'new barcodes ' + str(num_barcodes) + ' ...'


if __name__ == '__main__':
    sys.exit(main())