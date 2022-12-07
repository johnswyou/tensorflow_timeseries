import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
# from rpy2.robjects.vectors import StrVector
# from rpy2.robjects import FloatVector
# from rpy2.robjects import IntVector
# import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

base = importr('base')
utils = importr('utils')

def get_data(dataset="HUC_03_GAGEID_03488000"):
    
    # Parameters to vary for each experiment
    robjects.globalenv['.dataset'] = dataset

    robjects.r('''
    if (.dataset == "ourthe") {
    
    .x <- wddff::ourthe[, 6:8]
    df <- cbind(Q=wddff::ourthe$Q, .x)

    } else if (.dataset == "montreal") {
    
    .x <- wddff::montreal[, 6:8]
    df <- cbind(UWD=wddff::montreal$UWD, .x)

    } else if (.dataset == "HUC_03_GAGEID_03488000") {
    
    .x <- wddff::HUC_03_GAGEID_03488000[, 7:12]
    df <- cbind(Q.ft3.s.=wddff::HUC_03_GAGEID_03488000$Q.ft3.s., .x)

    } else {
    stop(".dataset should be either ourthe or motnreal")
    }

    ''')

    r_df = robjects.globalenv['df']

    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_from_r_df = robjects.conversion.rpy2py(r_df)

    return(pd_from_r_df)
