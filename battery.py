import sys
from bigquery import return_bq_query
import pandas as pd
import numpy as np
from datetime import datetime as dt
from numba import jit


def query_data_pkgs(table, bars, start_date=None, end_date=None):
    query = """                  
            SELECT
              *
            FROM
              TABLE
            WHERE
              end_timestamp >= 'START_DATE'
              AND end_timestamp < 'END_DATE'
            ORDER BY barcodes, start_timestamp
            """
    query = query.replace('START_DATE', start_date)
    query = query.replace('END_DATE', end_date)
    query = query.replace('TABLE', table)
    # query = query.replace('BARS', bars)

    return return_bq_query(query)


def query_data_bat(table, bars, start_date=None, end_date=None):
    query = """                  
            SELECT
              *
            FROM
              TABLE
            WHERE
              record_time >= 'START_DATE'
              AND record_time < 'END_DATE'
            ORDER BY barcode, record_time
            """
    query = query.replace('START_DATE', start_date)
    query = query.replace('END_DATE', end_date)
    query = query.replace('TABLE', table)
    # query = query.replace('BARS', bars)

    return return_bq_query(query)


@jit
def get_battery(X, B, start_date, init):
    t3 = dt.now()
    barcodes = X['barcodes'].unique()
    print "barcodes - " + str(len(barcodes)) + " ..."
    for b in barcodes:
        # try:
        begin = 0
        bb = B[B['barcode'] == b].drop_duplicates(subset='record_time', keep='first')
        ab = X[X['barcodes'] == b].drop_duplicates(keep='first').sort_values(by='start_timestamp')
        if not bb.empty:
            head = pd.to_datetime(bb.head(1)['record_time'].values[0])
            for i, event in ab.iterrows():
                end_time = pd.to_datetime(event['end_timestamp'])
                start_time = pd.to_datetime(event['start_timestamp'])
                if begin <= 10:
                    if start_time <= head:
                        first = bb.iloc[bb.index.get_loc(start_time, method='backfill')]['battery_level']
                        X.set_value(i, 'start_battery', np.int(first) + 1)
                    else:
                        begin += 1
                        first = bb.iloc[bb.index.get_loc(start_time, method='pad')]['battery_level']
                        X.set_value(i, 'start_battery', np.int(first))

                    if end_time <= head:
                        last = bb.iloc[bb.index.get_loc(end_time, method='backfill')]['battery_level']
                        X.set_value(i, 'end_battery', np.int(last) + 1)
                    else:
                        last = bb.iloc[bb.index.get_loc(end_time, method='pad')]['battery_level']
                        X.set_value(i, 'end_battery', np.int(last))
                else:
                    first = bb.iloc[bb.index.get_loc(start_time, method='pad')]['battery_level']
                    X.set_value(i, 'start_battery', np.int(first))
                    last = bb.iloc[bb.index.get_loc(end_time, method='pad')]['battery_level']
                    X.set_value(i, 'end_battery', np.int(last))
        # except Exception as err:
        #     # print('Error: {}'.format(err.content))
        #     print b

    t4 = dt.now()
    print "Analytics at " + str(start_date.date()) + " executed in " + str((t4 - t3).total_seconds() / 60.) \
          + " minutes at " + str(t4) + " ..."

    if init == 1:
        X.to_csv('appusage_battery_athene_170612_to_171022.csv.gz', header=True, index=False, compression='gzip')
    else:
        X.to_csv('appusage_battery_athene_170612_to_171022.csv.gz', header=False, index=False, mode='a',
                 compression='gzip')


def main():
    """Main entry point for the script."""
    startDate = '2017-06-12'
    endDate = '2017-10-22'
    table = '[motorola.com:sandbox:nielsen.appusage_athene_170605_to_171022_new]'
    table_bat = '[motorola.com:sandbox:nielsen.batteryusage_athene_170605_to_171022]'

    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c')"
    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c', 'ied733fe5ec38ee9f11931fe38d98be89865b33751')"
    # bars = "('iea39653b70d1e391d2c8af992fc873f32104ce96c', 'ied733fe5ec38ee9f11931fe38d98be89865b33751', " \
    #        "'ie98f1a318a84ce430fff87f5b21354bbe7dc4e4e8', 'ie0a5b3c3aeff2d1cbd9274aed1816770052667115'," \
    #        "'ie7574c01a7812d2ac12043abc50a3e4c2089294ea', 'ie6edd41b6fb2e2a7feee6a02c879160e2f4b89510', " \
    #        "'iedb938860c998fdc4ffb6b72146b75c3dc688fd2f', 'ieaed3d729b4638a80c9029df3203a1fd8aa693663')"
    bars = None

    days = pd.date_range(startDate, endDate, freq='D')

    init = 0
    for d in xrange(len(days)-1):
        t1 = dt.now()
        start_date = days[d]
        end_date = days[d+1]
        init += 1

        hours = pd.date_range(start_date, end_date, freq='4H')

        print "BigQuery sql ..."
        for h in xrange(len(hours)-1):
            start_hour = str(hours[h])
            end_hour = str(hours[h+1])

            if h == 0:
                X, bq_rows = query_data_pkgs(table, bars, start_hour, end_hour)
                B, bq_rows = query_data_bat(table_bat, bars, start_hour, end_hour)
            else:
                Xn, bq_rows = query_data_pkgs(table, bars, start_hour, end_hour)
                X = pd.concat([X, Xn])
                Bn, bq_rows = query_data_bat(table_bat, bars, start_hour, end_hour)
                B = pd.concat([B, Bn])

        t2 = dt.now()
        print "Queries executed in " + str((t2-t1).total_seconds()/60.) + " minutes at " + str(t2) + " ..."

        X.reset_index(inplace=True, drop=True)
        battery = pd.DataFrame(np.zeros((X.shape[0], 2)), dtype=np.int, index=X.index.values, columns=['start_battery', 'end_battery'])
        X = pd.concat([X, battery], axis=1)
        B.set_index(pd.DatetimeIndex(B['record_time']), inplace=True, drop=True)
        get_battery(X, B, start_date, init)


if __name__ == '__main__':
    sys.exit(main())