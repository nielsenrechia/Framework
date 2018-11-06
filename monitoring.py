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
from sklearn.metrics import confusion_matrix
import itertools
from numba import jit


def get_all_groups_barcodes(path_labels, path_outliers, dates, method, clusters, barcodes):
    header = barcodes.columns.values
    for d in xrange(len(dates) - 11):
        start_date = dates[d]
        end_date = dates[d + 1]
        period = str(start_date.date()) + '_' + str(end_date.date())

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


@jit
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

def overlap(clusterX, clusterY, objectsY):
    return float(len(np.intersect1d(clusterX, clusterY))) / float(len(np.intersect1d(clusterX, objectsY)))


def makelist(absorptionsSurvivals, Y):
    new_list = []
    for list in absorptionsSurvivals:
        if list[1] == Y:
            new_list += [list[0]]
    return new_list


# @jit
def monic_external_transistions(trashold, trasholdSplit, objectsX, objectsY):
    clustersX = pd.Series(objectsX.unique())
    clustersX = clustersX[clustersX != 0.1].dropna().values
    clustersY = pd.Series(objectsY.unique())
    clustersY = clustersY[clustersY != 0.1].dropna().values
    Mcells = pd.DataFrame(np.zeros((len(clustersX), len(clustersY))), index=clustersX, columns=clustersY)
    absorptionList = []
    absorptionsSurvivals = []
    survivallist = []
    deadList = []
    splitList = []
    birthList = clustersY
    for X in clustersX:
        splitCandidates = []
        splitUnion = []
        survivalCandidate = None
        survivalCandidateMcell = 0
        for Y in clustersY:
            Mcell = overlap(objectsX[objectsX == X].index, objectsY[objectsY == Y].index,
                            objectsY[objectsY != 0.1].dropna().index)
            Mcells.at[X, Y] = Mcell
            if Mcell >= trashold and Mcell > survivalCandidateMcell:
                    survivalCandidate = Y
                    survivalCandidateMcell = Mcell
            elif Mcell >= trasholdSplit:
                splitCandidates += [Y]
                splitUnion = np.union1d(splitUnion, objectsY[objectsY == Y].index)
        if not survivalCandidate and not splitCandidates:
            deadList += [[X]]
        elif survivalCandidate:
            absorptionsSurvivals += [[X, survivalCandidate]]
        elif splitCandidates:
            if overlap(objectsX[objectsX == X].index, splitUnion, objectsY[objectsY != 0.1].dropna().index) >= trashold:
                for Y in splitCandidates:
                    splitList += [[X, Y]]
                    birthList = np.delete(birthList, np.where(birthList == Y))
            elif survivalCandidate:
                absorptionsSurvivals += [[X, survivalCandidate]]
            else:
                deadList += [[X]]
    for Y in clustersY:
        absorptionCandidates = makelist(absorptionsSurvivals, Y)
        if len(absorptionCandidates) > 1:
            for X in absorptionCandidates:
                absorptionList += [[X, Y]]
                absorptionsSurvivals.remove([X, Y])
                birthList = np.delete(birthList, np.where(birthList == Y))
        elif len(absorptionCandidates) == 1:
            survivallist += [[absorptionCandidates[0], Y]]
            absorptionsSurvivals.remove([absorptionCandidates[0], Y])
            birthList = np.delete(birthList, np.where(birthList == Y))
    return absorptionList, survivallist, deadList, splitList, birthList, len(clustersX), len(clustersY)

def put_same_behaviors_together(barcodes_behaviors):
    header = barcodes_behaviors.columns.values[:-1]
    y_true = barcodes_behaviors.values[:, -1]
    # data = barcodes_behaviors.values[:, :-1]
    data = barcodes_behaviors.values #for taxi
    # header = barcodes_behaviors.columns.values # for taxi
    # teste = barcodes_behaviors.ix[:, 0:9].drop_duplicates()
    barcodes = barcodes_behaviors.index.values
    behaviors = []
    count = []
    cds = []
    outliers = []
    loyals = []
    missings = []
    weeks = []
    churns = []
    not_churns = []
    churns_full = []
    for d in data:
        w = 0
        if list(d[:-1]) in behaviors:
            count[behaviors.index(list(d[:-1]))] += 1
            if d[-1] == 'YES':
                churns[behaviors.index(list(d[:-1]))] += 1
            else:
                not_churns[behaviors.index(list(d[:-1]))] += 1
        else:
            behaviors += [list(d[:-1])]
            if d[-1] == 'yes':
                churns.append(1)
                not_churns.append(0)
            else:
                not_churns.append(1)
                churns.append(0)
            count.append(1)
            u, c = np.unique(d, return_counts=True)
            dicionario = dict(zip(u, c))
            if dicionario.has_key('C'):
                cds.append(dicionario['C'])
                w += dicionario['C']
            else:
                cds.append(0)
            if dicionario.has_key('outlier'):
                outliers.append(dicionario['outlier'])
                w += dicionario['outlier']
            else:
                outliers.append(0)
            if dicionario.has_key('miss'):
                missings.append(dicionario['miss'])
            else:
                missings.append(0)
            if dicionario.has_key('loyal'):
                loyals.append(dicionario['loyal'])
                w += dicionario['loyal']
            else:
                loyals.append(0)
            weeks.append(w+1)

    print len(behaviors)
    print len(header)

    behaviors = pd.DataFrame(data=behaviors, columns=header)
    counters = pd.DataFrame({'total_barcodes': count, 'loyals': loyals, 'cs': cds, 'outliers': outliers,
                             'missings': missings, 'weeks': weeks, 'churns': churns, 'not_churns': not_churns})
    final_matrix = pd.concat([behaviors, counters], axis=1).sort_values('total_barcodes', ascending=False)

    final_matrix.to_csv('results/cutomers_behaviors_together_without_first_week.csv', header=True, index=True)
    return final_matrix


def get_similar_behaviors_for_less_weeks_behaviors(behaviors_together):

    less_weeks = behaviors_together[behaviors_together['weeks'] < 10]

    for i, b in less_weeks.iterrows():
        null = b[b == 'miss'].index.values
        col_bar = []
        col_conduct = []

        for c in np.array(list(itertools.product(['loyal', 'C', 'outlier'], repeat=len(null)))):
            b[null] = c
            conduct = behaviors_together[(behaviors_together.ix[:, :9] == b[:9]).all(1)]
            if not conduct.empty:
                col_bar += [conduct['total_barcodes'].values[0]]
                col_conduct += [conduct.index.values[0]]

        bar_total = float(sum(col_bar))
        for c, col in enumerate(col_bar):
            col_bar[c] = (col/bar_total)*100

        if col_bar:
            max_percent = col_bar.index(max(col_bar))
            behaviors_together.loc[i, 'prob'] = max(col_bar)
            behaviors_together.loc[i, 'likely'] = col_conduct[max_percent]

    z = 0
    behaviors_together.to_csv('results/final_behaviors_together_qtd.csv', header=True, index=True)


def churn_prediction(barcodes_behaviors, final_behaviors):
    header = barcodes_behaviors.columns.values[:-1]
    y_true = barcodes_behaviors.values[:, -1]
    # data = dataset.values[:, :-1]
    barcodes = barcodes_behaviors.index.values

    predicted = []
    y_pred = []
    already_churn = final_behaviors[final_behaviors['weeks'] < 3]
    already_churn = already_churn.ix[:, 0:9]
    already_churn = already_churn.values.tolist()
    selected_behaviors = final_behaviors

    for i in xrange(len(header)-1):
        if i == 0:
            true_false = pd.DataFrame(((selected_behaviors[header[i]] != 'loyal') | (selected_behaviors[header[i + 1]] != 'loyal')) &
                                  ((selected_behaviors[header[i]] != 'C') | (selected_behaviors[header[i + 1]] != 'loyal')) &
                                  ((selected_behaviors[header[i]] != 'outlier') | (selected_behaviors[header[i + 1]] != 'outlier')),
                                  columns=[header[i + 1]])
        else:
            true_false = pd.concat([true_false,
                                    pd.DataFrame(((selected_behaviors[header[i]] != 'loyal') | (
                                    selected_behaviors[header[i + 1]] != 'loyal')) &
                                                 ((selected_behaviors[header[i]] != 'C') | (
                                                 selected_behaviors[header[i + 1]] != 'loyal')) &
                                                 ((selected_behaviors[header[i]] != 'outlier') | (
                                                 selected_behaviors[header[i + 1]] != 'outlier')),
                                                 columns=[header[i + 1]])], axis=1)

    selected_behaviors = pd.concat([selected_behaviors, true_false.apply(pd.Series.value_counts, axis=1)], axis=1)
    header = selected_behaviors.columns.values
    header = np.delete(header, -1)
    header = np.delete(header, -1)
    header = np.append(header, ['False', 'True'])
    selected_behaviors.columns = header
    selected_behaviors = selected_behaviors.fillna(0)

    #
    selected_behaviors.to_csv('results/changes_in_behaviors_to_predict_qtd.csv', header=True, index=True,
                              index_label='comportamentos')

    df1 = selected_behaviors[(selected_behaviors['total_barcodes'] < 50) & (selected_behaviors['missings'] > 0) & (selected_behaviors['outliers'] > 0)]
    df2 = selected_behaviors[(selected_behaviors['total_barcodes'] < 50) & (selected_behaviors['cs'] >= 2) & (selected_behaviors['cs'] <= 5)]
    df3 = selected_behaviors[(selected_behaviors['total_barcodes'] < 50) & (selected_behaviors['True'] >= 5) & (selected_behaviors['True'] <= 8)]
    selected_behaviors = pd.concat([df1, df2, df3]).drop_duplicates().reset_index(drop=True)
    # selected_behaviors = selected_behaviors[(selected_behaviors['False'] >= 0) & (selected_behaviors['False'] <= 1)]
    selected_behaviors = selected_behaviors[(selected_behaviors['total_barcodes'] < 10)]

    selected_behaviors = selected_behaviors.ix[:, 0:9]
    selected_behaviors = selected_behaviors.values.tolist()

    # confusion matrix and measures for all data
    for b in barcodes:
        # if list(dataset.loc[[b]].values[0][:-1]) in already_churn:
        #     predicted.append(b)
        #     y_pred.append('YES')
        if list(barcodes_behaviors.loc[[b]].values[0][:-1]) in selected_behaviors:
            predicted.append(b)
            y_pred.append('yes')
        else:
            predicted.append(b)
            y_pred.append('no')

    # np.savetxt(path+'churn_selected_all'+filename, np.vstack((np.asarray(predicted), np.asarray(y_pred))).T, delimiter=',', fmt='%s')

    cf = confusion_matrix(y_true, y_pred)
    tn = float(cf[0][0])
    fp = float(cf[0][1])
    fn = float(cf[1][0])
    tp = float(cf[1][1])
    actual_yes = fn+tp
    actual_no = tn+fp
    predicted_yes = fp+tp
    predicted_no = tn+fn
    total = float(len(barcodes))
    accuracy = round((tp+tn)/total,3)                # Overall, how often is the classifier correct?
    misclassification_rate = round((fp+fn)/total,3)  # Overall, how often is it wrong?
    true_positive = round(tp/actual_yes,3)           # When it's actually yes, how often does it predict yes?
    false_positive = round(fp/actual_no,3)           # When it's actually no, how often does it predict yes?
    specificity = round(tn/actual_no,3)              # When it's actually no, how often does it predict no?
    precision = round(tp/predicted_yes,3)            # When it predicts yes, how often is it correct?
    prevalence = round(actual_yes/total,3)           # How often does the yes condition actually occur in our sample?
    f1 = round(2 * ((precision * true_positive) / (precision + true_positive)),3)

    # confusion matrix and measures for only barcodes with more than 3 weeks for analysis (not already churn)

    predicted = []
    y_pred = []
    new_barcodes = []
    total_barcodes = 0
    for b in barcodes:
        if list(barcodes_behaviors.loc[[b]].values[0][:-1]) not in already_churn:
            new_barcodes.append(b)
            total_barcodes += 1
            if list(barcodes_behaviors.loc[[b]].values[0][:-1]) in selected_behaviors:
                predicted.append(b)
                y_pred.append('YES')
            else:
                predicted.append(b)
                y_pred.append('NO')
    #
    # np.savetxt(path + 'churn_selected_more_than_3_weeks' + filename, np.vstack((np.asarray(predicted), np.asarray(y_pred))).T,
    #            delimiter=',', fmt='%s')

    y_true = barcodes_behaviors.loc[new_barcodes].values[:, -1]

    cf_2 = confusion_matrix(y_true, y_pred)
    tn_2 = float(cf_2[0][0])
    fp_2 = float(cf_2[0][1])
    fn_2 = float(cf_2[1][0])
    tp_2 = float(cf_2[1][1])
    actual_yes_2 = fn_2 + tp_2
    actual_no_2 = tn_2 + fp_2
    predicted_yes_2 = fp_2 + tp_2
    predicted_no_2 = tn_2 + fn_2
    total_2 = float(len(new_barcodes))
    accuracy_2 = (tp_2 + tn_2) / total_2  # Overall, how often is the classifier correct?
    misclassification_rate_2 = (fp_2 + fn_2) / total_2  # Overall, how often is it wrong?
    true_positive_2 = tp_2 / actual_yes_2  # When it's actually yes, how often does it predict yes?
    false_positive_2 = fp_2 / actual_no_2  # When it's actually no, how often does it predict yes?
    specificity_2 = tn_2 / actual_no_2  # When it's actually no, how often does it predict no?
    precision_2 = tp_2 / predicted_yes_2  # When it predicts yes, how often is it correct?
    prevalence_2 = actual_yes_2 / total_2  # How often does the yes condition actually occur in our sample?
    f1_2 = 2*((precision_2*true_positive_2)/(precision_2+true_positive_2))
    z = 0