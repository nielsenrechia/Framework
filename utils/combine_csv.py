# encoding=utf8
import numpy as np
import pandas as pd
import os
import sys
import gc
from numba import jit
from datetime import datetime as dt


def combine(raw_filenames, path_raw):
    startDate = '2017-06-17'
    endDate = '2017-08-07'
    dates = pd.date_range(startDate, endDate, freq='1D')
    for i, file in enumerate(raw_filenames):
        # if file == 'appusage_battery_athene_170605_to_171022.csv.gz':
        #     print file
        #     for i, date in enumerate(dates):
        #         chunks = pd.read_csv(path_raw + 'appusage_battery_athene_170605_to_171022.csv.gz', header=0, index_col=None, compression='gzip', nrows=None, engine='c', chunksize=20000000)
        #         print str(date - 1)
        #         X = pd.DataFrame()
        #         print 'for chunks ...'
        #         for chunk in chunks:
        #             print 'getiing data from chunk ...'
        #             data = chunk[(pd.to_datetime(chunk['end_timestamp']) >= date-1) &
        #                          (pd.to_datetime(chunk['end_timestamp']) < date)]
        #             print 'concatating ...'
        #             X = pd.concat([X, data])
        #             print 'droping ...'
        #             del data
        #             gc.collect()
        #
        #         print 'dropping from all 1 ...'
        #         gc.collect()
        #         X.drop_duplicates(keep='first', inplace=True)
        #         print 'dropping from all 2...'
        #         gc.collect()
        #         X.drop_duplicates(subset=['barcodes', 'start_timestamp', 'duration_sec', 'end_timestamp'],
        #                           keep='first', inplace=True)
        #         gc.collect()
        #         print 'saving from all ...'
        #         if i == 0:
        #             X.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished.csv.gz', header=True,
        #                      index=False, compression='gzip')
        #         else:
        #             X.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished.csv.gz', header=False,
        #                      index=False, compression='gzip', mode='a')
        #         del X
        #         gc.collect()
        #
        print file
        X = pd.read_csv(path_raw + file, header=0, index_col=False, engine='c', compression='gzip')
        gc.collect()

        print 'dropping 1 ...'
        gc.collect()
        X.drop_duplicates(keep='first', inplace=True)
        print 'dropping 2 ...'
        gc.collect()
        X.drop_duplicates(subset=['barcodes', 'start_timestamp', 'duration_sec', 'end_timestamp'], keep='first',
                          inplace=True)
        gc.collect()

        print "saving ..."
        X.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished.csv.gz', header=False, index=False,
                 mode='a', compression='gzip')
        del X
        gc.collect()


@jit
def remove_sample(path_raw, file, barcodes):
    print 'reading ...'
    chunks = pd.read_csv(path_raw + file, header=0, index_col=None, compression='gzip', engine='c',
                         skiprows=None, chunksize=30000000)
    barcodes = pd.read_csv(path_raw + barcodes, header=0, index_col=None)
    barcodes = barcodes[barcodes['is_churn'] == 'no']['a_barcodes']
    print 'for chunks ...'
    count = 0
    removed = 0
    for chunk in chunks:
        print 'doing ...'
        t1 = dt.now()
        chunk_no = chunk[chunk['a_barcodes'].isin(barcodes)]
        chunk_03 = chunk_no.sample(frac=0.03)
        chunk = chunk.loc[~chunk.index.isin(chunk_03.index)]
        l, c = chunk.shape
        removed += (30000000 - l)
        print str(removed)

        print 'saving ...'
        if count == 0:
            chunk.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished_a_p.csv.gz', header=True,
                         index=False, compression='gzip')
        else:
            chunk.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished_a_p.csv.gz', header=False,
                         index=False, compression='gzip', mode='a')
        t4 = dt.now()
        print "Chunk executed in " + str((t4 - t1).total_seconds() / 60.) + " minutes at" + str(t4) + " ..."
        gc.collect()
        count += 1


@jit
def an(path_raw, file, barcodes):
    # startDate = '2017-06-05'
    # endDate = '2017-10-23'
    # dates = pd.date_range(startDate, endDate, freq='1D')
    # for i, date in enumerate(dates):
    #     print str(date)
    print 'reading ...'
    chunks = pd.read_csv(path_raw + file, header=None, index_col=None, compression='gzip', engine='c',
                         skiprows=340000001, chunksize=20000000)
    # header = [skiprows=340000000, chunksize=2]
    barcodes = pd.read_csv(path_raw + barcodes, header=0, index_col=None)
    print 'for chunks ...'
    count = 1
    for chunk in chunks:
        chunk.columns = ['barcodes', 'duration_sec', 'end_timestamp', 'pkgs', 'start_timestamp', 'start_battery',
                         'end_battery']
        print 'doing ...'
        t1 = dt.now()
        chunk['a_barcodes'] = pd.Series(dtype=np.str)
        for i, row in chunk.iterrows():
            # t2 = dt.now()
            a_bar = barcodes[barcodes['barcodes'] == row['barcodes']]['a_barcodes'].values[0]
            chunk.set_value(i, 'a_barcodes', a_bar)
            # t3 = dt.now()
            # print "Row executed in " + str((t3 - t2).total_seconds()) + " seconds ..."

        #
        # for i, bar in barcodes.iterrows():
        #     t2 = dt.now()
        #     chunk['a_barcodes'] = pd.Series()
        #     chunk.at[chunk['barcodes'] == bar['barcodes'], 'a_barcodes'] = bar['a_barcodes']
        #     t3 = dt.now()
        #     print "Barcode executed in " + str((t3 - t2).total_seconds()) + " seconds ..."

        print 'saving ...'
        if count == 0:
            del chunk['barcodes']
            chunk.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished_a.csv.gz', header=True,
                         index=False, compression='gzip')
        else:
            del chunk['barcodes']
            chunk.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finished_a.csv.gz', header=False,
                         index=False, compression='gzip', mode='a')
        t4 = dt.now()
        print "Chunk executed in " + str((t4 - t1).total_seconds() / 60.) + " minutes at" + str(t4) + " ..."
        gc.collect()
        count += 1


def teste(path_raw, file, barcodes):
    df = pd.read_csv(path_raw + file, header=None, index_col=None, skiprows=340000000, compression='gzip', engine='c')
    print df.tail(1)
    z = 0


def main():
    path_raw = 'datasets/'
    file = 'appusage_battery_athene_170605_to_171022_finished_a.csv.gz'
    file_a = 'appusage_battery_athene_170605_to_171022_finished_a.csv.gz'
    barcodes = 'barcodes_athene_170605_to_171022_finished.csv'
    # ignored = ['appusage_battery_athene_170605_to_170605.csv.gz', 'appusage_battery_athene_170605_to_171022.csv.gz',
    #           'appusage_battery_athene_170606_to_170607.csv.gz', 'appusage_battery_athene_170608_to_170608_bar.csv.gz',
    #           'appusage_battery_athene_170608_to_170611.csv.gz', 'appusage_battery_athene_170612_to_170612.csv.gz',
    #           'appusage_battery_athene_170613_to_170615.csv.gz', 'appusage_battery_athene_170616_to_170712.csv.gz']
    ignored = ['appusage_battery_athene_170605_to_171022_finished.csv.gz']
    raw_filenames = [i for i in os.listdir(path_raw) if i not in ignored]

    # combine(raw_filenames, path_raw)
    # teste(path_raw, file_a, barcodes)
    # an(path_raw, file, barcodes)
    remove_sample(path_raw, file, barcodes)

    # X = pd.read_csv(path_raw + 'appusage_battery_athene_170605_to_171022.csv.gz', header=0, index_col=None,
    #                compression='gzip', nrows=None, engine='c')
    # print 'dropping 1 ...'
    # gc.collect()
    # X.drop_duplicates(keep='first', inplace=True)
    # print 'dropping 2 ...'
    # gc.collect()
    # X.drop_duplicates(subset=['barcodes', 'start_timestamp', 'duration_sec', 'end_timestamp'], keep='first',
    #                  inplace=True)
    # gc.collect()
    # print 'saving ...'
    # X.to_csv(path_raw + 'appusage_battery_athene_170605_to_171022_finish.csv.gz', header=False, index=False, mode='a',
    #        compression='gzip')


if __name__ == '__main__':
    sys.exit(main())
