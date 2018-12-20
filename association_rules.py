# encoding=utf8
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
import gc
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import OneHotEncoder
pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()


def association_rules(discretization):
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    base = importr('base')
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    print 'pandas to r data frame ...'
    rdata = pandas_data_frame_to_rpy2_data_frame(discretization)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    rstring = """ 
                function(discretization){
                    library(arules)
                    for(i in 1:ncol(discretization)) discretization[,i] <- factor(discretization[,i])
                    header <- colnames(discretization)
                    header <- paste(header,"=0.0",sep="")
                    trans<-as(discretization, 'transactions')
                    rm(discretization)
                    itemsets <- apriori(trans, parameter = list(supp = 0.01, target = 'closed frequent itemsets', minlen=2, maxlen=4), appearance = list(none = c(header), default = "both"))
                    quality(itemsets) <- cbind(quality(itemsets), allConfidence = interestMeasure(itemsets, measure = "allConfidence", trans))
                    quality(itemsets) <- cbind(quality(itemsets), lift = interestMeasure(itemsets, measure = "lift", trans))
                    i_1 <- subset(itemsets, subset = allConfidence > summary(itemsets@quality$allConfidence)[[4]])
                    rm(itemsets)
                    rules_from_itemsets <- ruleInduction(i_1, trans, confidence = 0.1, control = list( method = 'apriori', memopt = TRUE))
                    rm(i_1)
                    rm(trans)
                    final_rules <- subset(rules_from_itemsets, subset = lift > 1)
                    rm(rules_from_itemsets)
                    # final_rules <- as(final_rules, "data.frame")
                    # final_rules
                    final_itemsets <- unique(generatingItemsets(final_rules))
                    rm(final_rules)
                    df_final <- as(final_itemsets, "data.frame")
                    rm(fina_itemsets)
                    df_final
                }
            """
    print 'creating r function ...'
    rfunc = robjects.r(rstring)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    print 'getting itemsets ...'
    itemsets = rfunc(rdata)
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    del rdata
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    print 'r data frame to pandas ...'
    itemsets = pandas2ri.ri2py(itemsets)
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    print 'converting pandas ...'
    itemsets = convert_r_result(itemsets)
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    return itemsets


def pandas_data_frame_to_rpy2_data_frame(pDataframe):
    orderedDict = rlc.OrdDict()

    for columnName in pDataframe:
        columnValues = pDataframe[columnName].values
        filteredValues = [value if pd.notnull(value) else robjects.NA_Real
                          for value in columnValues]

        try:
            orderedDict[columnName] = robjects.FloatVector(filteredValues)
        except ValueError:
            orderedDict[columnName] = robjects.StrVector(filteredValues)

    rDataFrame = robjects.DataFrame(orderedDict)
    rDataFrame.rownames = robjects.StrVector(pDataframe.index)

    return rDataFrame


def convert_r_result(itemsets):
    data = itemsets['items'].str.lstrip('{').str.rstrip('}').str.split(',').tolist()
    itemsets = pd.DataFrame(data)
    l, c = itemsets.shape
    columns = ['x'+str(i) for i in range(1, c+1)]
    itemsets.columns = columns
    return itemsets


def findFPOF(all_behaviors):
    all_behaviors = all_behaviors.loc[
            (all_behaviors['2017-06-12_2017-06-19'] != 'miss') & (all_behaviors['2017-06-19_2017-06-26'] != 'miss') &
            (all_behaviors['2017-06-26_2017-07-03'] != 'miss') & (all_behaviors['2017-07-03_2017-07-10'] != 'miss') &
            (all_behaviors['2017-07-10_2017-07-17'] != 'miss') & (all_behaviors['2017-07-17_2017-07-24'] != 'miss') &
            (all_behaviors['2017-07-24_2017-07-31'] != 'miss') & (all_behaviors['2017-07-31_2017-08-07'] != 'miss')]

    all_behaviors = all_behaviors.loc[
        (all_behaviors['2017-06-12_2017-06-19'] != 'outlier') & (all_behaviors['2017-06-19_2017-06-26'] != 'outlier') &
        (all_behaviors['2017-06-26_2017-07-03'] != 'outlier') & (all_behaviors['2017-07-03_2017-07-10'] != 'outlier') &
        (all_behaviors['2017-07-10_2017-07-17'] != 'outlier') & (all_behaviors['2017-07-17_2017-07-24'] != 'outlier') &
        (all_behaviors['2017-07-24_2017-07-31'] != 'outlier') & (all_behaviors['2017-07-31_2017-08-07'] != 'outlier')]

    all_behaviors_dummies = all_behaviors.drop(columns=['is_churn'])
    all_behaviors_dummies = pd.get_dummies(all_behaviors_dummies)
    topn = 10
    topk = 0
    support = 0.2
    frequent_itemsets = apriori(all_behaviors_dummies, min_support=support, use_colnames=True)
    frequent_itemsets.to_csv('FPOF_itemsets_without_miss_and_outlier.csv', index=True, header=True)
    # z = 0
    for i, t in all_behaviors_dummies.iterrows():
        print i
        fpof = 0.
        for j, f in frequent_itemsets.iterrows():
            # x = t[t==1]
            # print t[t==1].index.values.tolist()
            # print "-----------------"
            # print list(f['itemsets'])
            # f = pd.Series(f['itemsets'])
            # x = f[f['itemsets'].isin(t[t == 1].index)]
            # z = 0
            # print set(f['itemsets']).issubset(set(t[t==1].index))
            if list(f['itemsets']) in t[t==1].index.values:
            #     print "yes"
            # else:
            #     print "no"
            # z = 0
                fpof += f['support']
                z = 0
        all_behaviors.loc[(i, 'FPOF')] = fpof/float(frequent_itemsets.shape[0])
    all_behaviors.sort_values(by=['FPOF'])
    all_behaviors.to_csv('FPOF_without_miss_and_outlier.csv', index=True, header=True)
    print all_behaviors.head(topn)
    z = 0
        # get top-n outliers
    # for t in top-n outliers:
    #     find top-k contract frenquent patterns