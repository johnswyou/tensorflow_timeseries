from tsutils import WindowGenerator, compile_and_fit, split_data, split_data_2
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from tensorflow.keras import backend as K
from preprocessor import make_preprocessor_function
import pandas as pd

# *************
# Search Space
# *************

# Wavelet scaling filter and decomposition level
dim_w1= Integer(low=1, high=6, name='w1')
dim_w2= Integer(low=1, high=6, name='w2')
dim_w3= Integer(low=1, high=6, name='w3')
dim_w4= Integer(low=1, high=6, name='w4')
dim_w5= Integer(low=1, high=6, name='w5')
dim_w6= Integer(low=1, high=6, name='w6')

dim_w7= Integer(low=1, high=20, name='w7')
dim_w8= Integer(low=1, high=20, name='w8')
dim_w9= Integer(low=1, high=20, name='w9')
dim_w10= Integer(low=1, high=20, name='w10')
dim_w11= Integer(low=1, high=20, name='w11')
dim_w12= Integer(low=1, high=20, name='w12')

# LSTM architecture
dim_w13= Integer(low=2, high=8, name='LSTM_units')
dim_w14= Integer(low=1, high=9, name='recurrent_dropout')
dim_w15= Integer(low=1, high=9, name='dropout')

dimensions = [dim_w1,dim_w2,dim_w3,dim_w4,dim_w5,dim_w6,dim_w7,dim_w8, dim_w9, dim_w10, dim_w11, dim_w12, dim_w13, dim_w14, dim_w15]

# *******************
# Objective function
# *******************

@use_named_args(dimensions=dimensions)
def fitness_LSTM(w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,LSTM_units,recurrent_dropout,dropout):
    
    print('Training a new configuration ...')

    tempfun = make_preprocessor_function("HUC_03_GAGEID_03488000")
    basin_3 = tempfun([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12])
    basin_3.drop("y_0", axis=1, inplace=True)
    train_df, val_df, test_df = split_data_2(basin_3, 0.7, 0.15, "y_target")

    ##########################################################################################################
    # CHANGE THE FIRST AND THIRD ARGUMENTS IN WindowGenerator BELOW TO CHANGE LOOKBACK AND FORECAST LEAD TIME
    ##########################################################################################################

    w = WindowGenerator(15, 1, 0, train_df, val_df, test_df, ['y_target'])

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(2**LSTM_units, recurrent_dropout = 0.1*recurrent_dropout,return_sequences=False),
        tf.keras.layers.Dropout(0.1*dropout),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1, activation = 'linear')
    ])

    patience = 10
    maximum_epochs = 100

    history = compile_and_fit(lstm_model, w, maximum_epochs, patience)
    val_perf = lstm_model.evaluate(w.val)
    accuracy = val_perf[0]

    # We use the global keyword so we update the variable outside of this function.
    global best_accuracy

    # If the prediction accuracy of the saved model is improved ...
    if accuracy < best_accuracy:
            
        print('Accuracy improved!')
        print(f'Old accuracy: {best_accuracy}')
        print(f'New accuracy: {accuracy}')

        # Update the prediction accuracy.
        best_accuracy = accuracy

        print('Saving new best model ...')
        lstm_model.save('model_LSTM.h5')
            
    else:
        print('Accuracy did not improve from current best model.')

    # Delete the Keras model with these hyper-parameters from memory.
    del lstm_model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    return accuracy

if __name__ == "__main__":
    # Bayesian optimization of the hyperparameters
    best_accuracy=10
    search_result = gp_minimize(func=fitness_LSTM,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=75)

    # Saving the results to csv
    nse_values=search_result.func_vals
    df_nse=pd.DataFrame(nse_values)
    df_nse.to_csv('NSE.csv',index=False)
    list_hp=list(search_result.x)
    list_hp.append(-best_accuracy)

    # Change the names if you want
    pd_best=pd.DataFrame(list_hp, index=['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','w11','w12','LSTM_units', 'recurrent_dropout', 'dropout','NSE'],columns=['parameter'])
    name_best='best_config_LSTM_it.csv'
    pd_best.to_csv(name_best)