from tsutils import WindowGenerator, compile_and_fit, split_data_2
from get_data import get_data
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from tensorflow.keras import backend as K
import pandas as pd

# *************
# Search Space
# *************

# LSTM architecture
dim_w1 = Integer(low=2, high=8, name='LSTM_units')
dim_w2 = Integer(low=1, high=9, name='recurrent_dropout')
dim_w3 = Integer(low=1, high=9, name='dropout')

dimensions = [dim_w1, dim_w2, dim_w3]

basin_3 = get_data("HUC_03_GAGEID_03488000")

# IF FORECASTING
basin_3['y_target'] = basin_3.loc[:, 'Q.ft3.s.']

# ELSE
# basin_3.rename(columns={'Q.ft3.s.': 'y_target'}, inplace=True)

# *******************
# Objective function
# *******************

@use_named_args(dimensions=dimensions)
def fitness_LSTM(LSTM_units, recurrent_dropout, dropout):
    
    print('Training a new configuration ...')

    train_df, val_df, test_df = split_data_2(basin_3, 0.7, 0.15, "y_target")

    ##########################################################################################################
    # CHANGE THE FIRST AND THIRD ARGUMENTS IN WindowGenerator BELOW TO CHANGE LOOKBACK AND FORECAST LEAD TIME
    ##########################################################################################################

    w = WindowGenerator(14, 1, 1, train_df, val_df, test_df, ['y_target'])

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(2**LSTM_units, recurrent_dropout = 0.1*recurrent_dropout,return_sequences=False),
        tf.keras.layers.Dropout(0.1*dropout),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1, activation = 'linear')
    ])

    patience = 10
    maximum_epochs = 1000

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
    pd_best=pd.DataFrame(list_hp, index=['LSTM_units', 'recurrent_dropout', 'dropout','NSE'],columns=['parameter'])
    name_best='best_config_LSTM_it.csv'
    pd_best.to_csv(name_best)
