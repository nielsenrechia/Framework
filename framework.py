# encoding=utf8
"""Framework script"""
import sys
# from dateutil import parser
from bigquery import return_bq_query
# from dataPlot import plot_best_apps, plot_discretization
from dataPreprocess import remove_natives_pkgs, apps_matrix, select_by_min_usage, discretization, get_most_used_pkgs
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
                """
    query = query.replace('START_DATE', start_date)
    query = query.replace('END_DATE', end_date)
    query = query.replace('TABLE', table)
    query = query.replace('NATIVES', natives)

    return return_bq_query(query)


def main():
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

    """Main entry point for the script."""
    startDate = '2017-06-05'
    endDate = '2017-10-22'
    frequency = '1D'
    table = '[motorola.com:sandbox:nielsen.appusage_athene_170605_to_171022_new]'
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
    thresholds = [0.92, 0.97]
    minTime = 70.
    minPercent = 0.01
    mostUsedPercent = 0.01
    popularsPercent = 0.1
    discretization_type = ['frequency'] #['frequency', 'ip', 'clustering']

    dates = pd.date_range(startDate, endDate, freq=frequency)

    print "BigQuery sql ..."
    for d in xrange(len(dates) - 1):
        t1 = dt.now()
        start_date = dates[d]
        end_date = dates[d + 1]

        hours = pd.date_range(start_date, end_date, freq='4H')

        for h in xrange(len(hours) - 1):
            start_hour = str(hours[h])
            end_hour = str(hours[h + 1])

            if h == 0:
                X, bq_rows = query_data(table, bars, natives, start_hour, end_hour)
            else:
                Xn, bq_rows = query_data(table, bars, natives, start_hour, end_hour)
                X = pd.concat([X, Xn])
                del Xn
                gc.collect()
            print str(h)

        t2 = dt.now()
        print "BigQuery executed in " + str((t2-t1).total_seconds()/60.) + " minutes at " + str(t2) + " ..."

        # print "Removing antives pkgs ..."
        # X = remove_natives_pkgs(X, natives)

        print 'total records is ' + str(X.shape[0]) + ' ...'
        pkgs = X['pkgs'].unique()
        num_pkgs = pkgs.shape[0]
        print 'all pkgs ' + str(num_pkgs) + ' ...'
        barcodes = X['barcodes'].unique()
        num_barcodes = barcodes.shape[0]
        print 'barcodes ' + str(num_barcodes) + ' ...'

        print "groupping by ..."
        X = X.groupby(['barcodes', 'pkgs'], as_index=False).agg({'duration_sec': 'sum'})

        # print "best apps selection ..."
        # plot_best_apps(X, 'duration_sec', language, num_pkgs, thresholds)

        print "getting pkgs summary ..."
        pkgs_summary = X.groupby('pkgs').agg(
            {'duration_sec':
                 {'count': 'count',
                  'mean': 'mean',
                  'std': 'std',
                  'min': 'min',
                  '25%': lambda x: np.percentile(x, 25),
                  '50%': lambda x: np.percentile(x, 50),
                  '75%': lambda x: np.percentile(x, 75),
                  'max': 'max'
                  }})
        pkgs_summary.columns = pkgs_summary.columns.droplevel(0)
        # pkgs_summary = None

        print "most used data first..."
        most_used_pkgs, X = get_most_used_pkgs(X, barcodes, pkgs, mostUsedPercent)

        print "most important apps selection ..."
        X = select_by_min_usage(X, barcodes, pkgs_summary, minTime, minPercent)

        # print "apps to matrix ..."
        # matrix, most_used_pkgs, populars_pkgs, X = apps_matrix(X, barcodes, pkgs, mostUsedPercent, popularsPercent,
        #                                                        'duration_sec')

        gc.collect()
        # pkgs = matrix['pkgs'].unique()
        # num_pkgs = pkgs.shape[0]
        # print 'new all pkgs ' + str(num_pkgs) + ' ...'
        # barcodes = matrix['barcodes'].unique()
        # num_barcodes = barcodes.shape[0]
        # print 'new barcodes ' + str(num_barcodes) + ' ...'

        # print "discretization ..."
        # teste = plot_discretization(matrix, populars_pkgs, most_used_pkgs, barcodes, discretization)
        # summary, discratization = discretization(matrix, populars_pkgs,  most_used_pkgs, discretization_type, barcodes)
        # z = 0

if __name__ == '__main__':
    sys.exit(main())