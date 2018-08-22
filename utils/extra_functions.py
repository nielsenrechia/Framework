import pandas as pd


def add_barcodes_to_labels(path_labels, path_discretization, path_outliers, dates, clusters):

    for d in xrange(len(dates) - 11):
        start_date = dates[d]
        end_date = dates[d + 1]
        period = str(start_date.date()) + '_' + str(end_date.date())

        all_groups = pd.read_csv(path_labels + 'labels_' + period + '_' + 'ward' + '_' + str(clusters[d]) + '.csv',
                                 index_col=0, header=None)
        outliers = pd.read_csv(path_outliers + 'outliers_' + period + 'csv.gz', index_col=None, header=None)
        discratization = pd.read_csv(path_discretization + 'discretization_' + period + '.csv.gz', index_col=None,
                                     header=0, compression='gzip', usecols=[0])
        discratization.columns = ['barcodes']

        not_outliers = discratization[~discratization['barcodes'].isin(outliers[0])].reset_index(drop=True)
        all_groups['barcodes'] = not_outliers
        all_groups.columns = [['labels', 'barcodes']]
        all_groups.to_csv(path_labels + 'barcodes_labels_' + period + '_' + 'ward' + '_' + str(clusters[d]) + '.csv',
                          index=False, header=True)

    z = 0