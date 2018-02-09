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

