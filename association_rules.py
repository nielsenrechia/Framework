# encoding=utf8
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
import gc
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()


def association_rules(discretization):
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    r('memory.limit(size=16335)')
    base = importr('base')
    base.gc()
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    # del discretization['com.whatsapp']
    # discretization.to_csv('discretization_teste.csv')
    rdata = pandas_data_frame_to_rpy2_data_frame(discretization)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    del discretization
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    rstring = """ 
                function(discretization){
                    library(arules)
                    # library(arulesViz)
                    # library(visNetwork)
                    # library(igraph)
                    # data<- discretization[2:ncol(discretization)]
                    # discretization
                    for(i in 1:ncol(discretization)) discretization[,i] <- factor(discretization[,i])
                    header <- colnames(discretization)
                    print ('here')
                    header <- paste(header,"=0.0",sep="")
                    trans<-as(discretization, 'transactions')
                    itemsets <- apriori(trans, parameter = list(supp = 0.01, target = 'closed frequent itemsets', minlen=2, maxlen=4), appearance = list(none = c(header), default = "both"))
                    quality(itemsets) <- cbind(quality(itemsets), allConfidence = interestMeasure(itemsets, measure = "allConfidence", trans))
                    quality(itemsets) <- cbind(quality(itemsets), lift = interestMeasure(itemsets, measure = "lift", trans))
                    # quality(itemsets) <- cbind(quality(itemsets), crossSupportRatio = interestMeasure(itemsets, measure = 'crossSupportRatio', trans))
                    # itemsets <- sort(itemsets, by="allConfidence")
                    i_1 <- subset(itemsets, subset = allConfidence > summary(itemsets@quality$allConfidence)[[4]])
                    rules_from_itemsets <- ruleInduction(i_1, trans, confidence = 0.1, control = list( method = 'apriori'))
                    # quality(rules_from_itemsets) <- cbind(quality(rules_from_itemsets), cosine = interestMeasure(rules_from_itemsets, measure = "cosine", trans))
                    # quality(rules_from_itemsets) <- cbind(quality(rules_from_itemsets), chiSquared = interestMeasure(rules_from_itemsets, measure = "chiSquared", trans))
                    final_rules <- subset(rules_from_itemsets, subset = lift > 1)
                    final_itemsets <- unique(generatingItemsets(final_rules))
                    df_final <- as(final_itemsets, "data.frame")
                    df_final
                }
            """

    rfunc = robjects.r(rstring)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    print r('memory.limit()')
    itemsets = rfunc(rdata)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    # print(robjects.r['summary'](rules))
    del rdata
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    itemsets = pandas2ri.ri2py(itemsets)
    gc.collect()
    robjects.r('rm(a)')
    robjects.r('gc()')
    gc.collect()
    itemsets = convert_r_result(itemsets)
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
    itemsets = pd.DataFrame(itemsets['items'].str.lstrip('{').str.rstrip('}').str.split(',').tolist(), columns = ['x1', 'x2', 'x3', 'x4'])
    return itemsets
