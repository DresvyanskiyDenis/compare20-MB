"""
This file calculates ensemble performance (Pearson correlation) as separately for each model in ensemble as
performance of ensemble over all data (train+development)
To do it, we load each model and then evaluate performence of loaded model exactly on such part of data, which
was validation part in process of training of this model.
For example, for dividing data on 2 parts, we have part_1 and part_2 of training_data as well as development data
    training_data                       development_data
part_1          part_2               part_1            part_2
Also, we have 4 models of ensemble:
model_for_part_0 was trained on [training_data_part_2, development_data_part_1, development_data_part_2], validation on: [training_data_part_1]
model_for_part_1 was trained on [training_data_part_1, development_data_part_1, development_data_part_2], validation on: [training_data_part_2]
model_for_part_2 was trained on [training_data_part_1, training_data_part_2, development_data_part_2], validation on: [development_data_part_1]
model_for_part_3 was trained on [training_data_part_1, training_data_part_2, development_data_part_1], validation on: [development_data_part_2]

So, we evaluate performance of model_for_part_0 only and exactly on training_data_part_1
                               model_for_part_1 only and exactly on training_data_part_2
                               and so on
At the end, we concatenate predicted training_data_part_1, training_data_part_2, development_data_part_1, development_data_part_2
                                  by   model_for_part_0  ,    model_for_part_1 ,    model_for_part_2    ,      model_for_part_3
in one array and test its performance (Pearson correlation).
That is a final metric to measure performance of ensemble without test data
"""



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
    data_parts=2
    batch_size=45
    length_sequence=256000
    step_sequence=102400
    # paths to models
    path_to_model_for_part_0='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_0.h5'
    path_to_model_for_part_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_1.h5'
    path_to_model_for_part_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_2.h5'
    path_to_model_for_part_3='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Compare_2020\\models\\best_model_weights_idx_of_part_3.h5'
    paths_to_models=[path_to_model_for_part_0, path_to_model_for_part_1, path_to_model_for_part_2, path_to_model_for_part_3]

    # train data
    path_to_train_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
    path_to_train_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
    train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
    prepared_train_data, prepared_train_labels,prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, length_sequence, step_sequence)
    train_parts=divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps, parts=data_parts, filenames_dict=train_dict)

    # devel data
    path_to_devel_data='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/wav/'
    path_to_devel_labels='C:/Users/Dresvyanskiy/Desktop/ComParE2020_Breathing/lab/'
    devel_data, devel_labels, devel_dict, frame_rate=load_data(path_to_devel_data, path_to_devel_labels, 'devel')
    prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, frame_rate, length_sequence, step_sequence)
    devel_parts=divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps, parts=data_parts, filenames_dict=devel_dict)

    ground_truth_labels=pd.DataFrame(columns=train_labels.columns)
    total_predicted_labels=pd.DataFrame(columns=train_labels.columns)
    total_parts=train_parts+devel_parts


    for i in range(len(paths_to_models)):
        # choose part to evaluate performance of model of ensemble
        part=[total_parts[i]]
        # unpack it
        part_d, part_lbs, part_timesteps, part_filenames_dict = extract_list_of_parts(list_of_parts=part)
        part_filenames_dict = part_filenames_dict[0]
        # reshape data to do it readable for keras
        part_d, _ = reshaping_data_for_model(part_d, part_lbs)
        # create and load model
        model = create_model(input_shape=(part_d.shape[-2], part_d.shape[-1]))
        model.load_weights(paths_to_models[i])
        model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])

        predicted_labels = model.predict(part_d, batch_size=batch_size)
        # average predictions. Data was cutted on windows with overlapping.
        # That is why we need to average predictions in overlapping points
        concatenated_predicted_labels = concatenate_prediction(predicted_labels, part_timesteps, part_filenames_dict)
        # save predicted labels in total_predicted_labels to evaluate then performances over all models in ensemble
        total_predicted_labels = pd.concat((total_predicted_labels, concatenated_predicted_labels), axis=0)
        # load ground truth labels. First half is always comes from train data and the second half - from development part
        if i<(len(train_parts)+len(devel_parts))/2:
            ground_truth_labels_part = choose_real_labs_only_with_filenames(train_labels, list(part_filenames_dict.values()))
        else:
            ground_truth_labels_part = choose_real_labs_only_with_filenames(devel_labels, list(part_filenames_dict.values()))
        r = scipy.stats.pearsonr(ground_truth_labels_part.iloc[:, 2].values, concatenated_predicted_labels.iloc[:, 2].values)
        # save ground truth labels for exactly this part of data to calculate then performance of ensemble
        # over all parts of data
        ground_truth_labels = pd.concat((ground_truth_labels, ground_truth_labels_part), axis=0)
        print('r on part ', str(i)+':', r)
        # clear RAM
        del model
        K.clear_session()
        gc.collect()

    # evaluate performance of ensemble over all parts of data
    r=scipy.stats.pearsonr(ground_truth_labels.iloc[:,2].values,total_predicted_labels.iloc[:,2].values)
    print('correlation, total:',r)
