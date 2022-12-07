import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import FloatVector
from rpy2.robjects import IntVector
import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

base = importr('base')
utils = importr('utils')

# Fixed parameters for JY 2022 paper
# robjects.globalenv['.evaluation_metric'] = "NSE"
# robjects.globalenv['.leadtime'] = 0
# robjects.globalenv['.lag_Y'] = 0
# robjects.globalenv['.lag_X'] = 0
robjects.globalenv['.wt'] = "modwt"
robjects.globalenv['.max_decomp_level'] = 6
robjects.globalenv['.max_wavelet_length'] = 14
# robjects.globalenv['.ivsm'] = "pcis_bic"
# robjects.globalenv['.ddm'] = "spov"

def make_preprocessor_function(dataset="ourthe"):
    
    # Parameters to vary for each experiment
    robjects.globalenv['.dataset'] = dataset
    # robjects.globalenv['.wfm'] = wfm

    # if dataset == "ourthe":

        # robjects.globalenv['.nval'] = 1096
        # robjects.globalenv['.ntst'] = 730
        # robjects.globalenv['.fixed_ddm_param'] = 2

    # elif dataset == "montreal":

        # robjects.globalenv['.nval'] = 583
        # robjects.globalenv['.ntst'] = 366
        # robjects.globalenv['.fixed_ddm_param'] = 1

    # else:

        # raise ValueError

    robjects.r('''
    if (.dataset == "ourthe") {
    .y <- as.matrix(wddff::ourthe$Q)
    .x <- as.matrix(wddff::ourthe[, 6:8])
    } else if (.dataset == "montreal") {
    .y <- as.matrix(wddff::montreal$UWD)
    .x <- as.matrix(wddff::montreal[, 6:8])
    } else if (.dataset == "HUC_03_GAGEID_03488000") {
    .y <- as.matrix(wddff::HUC_03_GAGEID_03488000$Q.ft3.s.)
    .x <- as.matrix(wddff::HUC_03_GAGEID_03488000[, 7:12])
    } else {
    stop(".dataset should be either ourthe or motnreal")
    }

    obj_func <- wddff:::makePreprocessor(.y,
                                         .x,
                                         .wt,
                                         .max_decomp_level, .max_wavelet_length,
                                         .temp_param=FALSE)
    ''')

    obj_func = robjects.globalenv['obj_func']

    def preprocessor_function(input_vector):

        '''
        input_vector: a list
        '''

        input_vector = [round(num) for num in input_vector]
        input_vector = IntVector(input_vector)
        
        # r_df = obj_func(input_vector)[0]
        r_df = obj_func(input_vector)
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            pd_from_r_df = robjects.conversion.rpy2py(r_df)

        return(pd_from_r_df)

        # return np.array(obj_func(input_vector)[0])
    
    return preprocessor_function