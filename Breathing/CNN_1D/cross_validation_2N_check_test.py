import gc

import scipy

from Compare2020.CNN_1D.cross_validation_2N_train import divide_data_on_parts, extract_list_of_parts, \
    reshaping_data_for_model, choose_real_labs_only_with_filenames, concatenate_prediction
from Compare2020.CNN_1D.utils import create_model, load_data, prepare_data, correlation_coefficient_loss
import pandas as pd
import numpy as np
from keras import backend as K

if __name__ == "__main__":
    # data params
    batch_size=45
    length_sequence=256000
    step_sequence=102400
    # paths to models of ensemble
    path_to_model_0='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_0.h5'
    path_to_model_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_1.h5'
    path_to_model_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_2.h5'
    path_to_model_3='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_3.h5'
    paths_to_models=[path_to_model_0, path_to_model_1, path_to_model_2, path_to_model_3]
    # test data
    path_to_test_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
    path_to_test_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
    test_data, test_labels, test_dict, frame_rate=load_data(path_to_test_data, path_to_test_labels, 'test')
    prepared_test_data, prepared_test_labels,prepared_test_labels_timesteps=prepare_data(test_data, test_labels, test_dict, frame_rate, length_sequence, step_sequence)
    prepared_test_data, _=reshaping_data_for_model(prepared_test_data, prepared_test_labels)

    predicted=[]
    for path_to_model in paths_to_models:
        # crate and load weights of single model of ensemble
        model=create_model(input_shape=(prepared_test_data.shape[-2], prepared_test_data.shape[-1]))
        model.load_weights(path_to_model)
        model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
        # predict test data bz loaded model
        predicted_labels = model.predict(prepared_test_data, batch_size=batch_size)
        # average predictions of model. Data was cutted on windows with overlapping.
        # That is why we need to average predictions in overlapping points
        concatenated_predicted_labels = concatenate_prediction(predicted_labels, prepared_test_labels_timesteps, test_dict)
        predicted.append(concatenated_predicted_labels)
        # clear RAM
        del model
        K.clear_session()
        gc.collect()
    # Since we have several models and, accordingly, several predictions on the same data, we need to average it
    # It is done via typical unweighted mean of predictions
    # total_prediction is an array to accumulate predictions
    total_prediction=predicted[0].copy(deep=True)
    for i in range(1,len(predicted)):
        # sum all predictions
        total_prediction['upper_belt']=total_prediction['upper_belt']+predicted[i]['upper_belt']
    # divide summered predictions by number of models (in this case it is 4, while there are 4 models in ensemble)
    total_prediction['upper_belt']=total_prediction['upper_belt']/4.
    # save final predictions in file
    total_prediction.to_csv('result.csv', index=False)
