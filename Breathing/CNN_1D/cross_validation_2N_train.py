import os

import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
import gc
from keras import backend as K

from Breathing.CNN_1D.utils import create_model, load_data, prepare_data, correlation_coefficient_loss, \
    create_complex_model


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def divide_data_on_parts(data, labels, timesteps, filenames_dict, parts=2):
    list_parts=[]
    length_part=int(data.shape[0]/parts)
    start_point=0
    for i in range(parts-1):
        tmp_data=data[start_point:(start_point+length_part)]
        tmp_labels=labels[start_point:(start_point+length_part)]
        tmp_timesteps = timesteps[start_point:(start_point + length_part)]
        tmp_filenames_dict={}
        idx=0
        for j in range(start_point,start_point+length_part):
            tmp_filenames_dict[idx]=list(filenames_dict.values())[j]
            idx+=1
        list_parts.append((tmp_data, tmp_labels, tmp_timesteps, tmp_filenames_dict))
        start_point+=length_part
    tmp_data = data[start_point:]
    tmp_labels = labels[start_point:]
    tmp_timesteps = timesteps[start_point:]
    tmp_filenames_dict = {}
    idx = 0
    for j in range(start_point, data.shape[0]):
        tmp_filenames_dict[idx] = list(filenames_dict.values())[j]
        idx += 1
    list_parts.append((tmp_data, tmp_labels, tmp_timesteps,tmp_filenames_dict))
    return list_parts


def form_train_and_val_datasets(train_parts, dev_parts, index_for_validation_part):
    total=[]
    for i in range(len(train_parts)):
        total.append(train_parts[i])
    for i in range(len(dev_parts)):
        total.append((dev_parts[i]))
    val_dataset=[total.pop(index_for_validation_part)]
    train_dataset=total
    return train_dataset, val_dataset

def extract_list_of_parts(list_of_parts):
    data=list_of_parts[0][0]
    labels=list_of_parts[0][1]
    timesteps=list_of_parts[0][2]
    dicts=[list_of_parts[0][3]]
    for i in range(1,len(list_of_parts)):
        data=np.append(data,list_of_parts[i][0], axis=0)
        labels = np.append(labels, list_of_parts[i][1], axis=0)
        timesteps = np.append(timesteps, list_of_parts[i][2], axis=0)
        dicts.append(list_of_parts[i][3])
    return data, labels, timesteps, dicts

def reshaping_data_for_model(data, labels):
    result_data=data.reshape((-1,data.shape[2])+(1,))
    result_labels=labels.reshape((-1,labels.shape[2]))
    return result_data, result_labels

def concatenate_prediction(predicted_values, labels_timesteps, filenames_dict, columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
    predicted_values = predicted_values.reshape(labels_timesteps.shape)
    result_predicted_values = pd.DataFrame(columns=columns_for_real_labels, dtype='float32')
    result_predicted_values['filename'] = result_predicted_values['filename'].astype('str')
    for instance_idx in range(predicted_values.shape[0]):
        predicted_values_tmp = predicted_values[instance_idx].reshape((-1, 1))
        timesteps_labels_tmp = labels_timesteps[instance_idx].reshape((-1, 1))
        tmp = pd.DataFrame(columns=['timeFrame', 'upper_belt'],
                           data=np.concatenate((timesteps_labels_tmp, predicted_values_tmp), axis=1))
        tmp = tmp.groupby(by=['timeFrame']).mean().reset_index()
        tmp['filename'] = filenames_dict[instance_idx]
        result_predicted_values = result_predicted_values.append(tmp.copy(deep=True))
    result_predicted_values['timeFrame'] = result_predicted_values['timeFrame'].astype('float32')
    result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
    return result_predicted_values[columns_for_real_labels]

def choose_real_labs_only_with_filenames(labels, filenames):
    return labels[labels['filename'].isin(filenames)]


def main(window_size=256000, data_parts=2,
         model_type='default', # can be 'complex'
         path_to_save_models='best_models/',
         path_to_save_tmp_models='tmp_model/',
         path_to_train_data='D:/Challenges/Compare2020/ComParE2020_Breathing/wav/',
         path_to_train_labels = 'D:/Challenges/Compare2020/ComParE2020_Breathing/lab/',
         path_to_devel_data='D:/Challenges/Compare2020/ComParE2020_Breathing/wav/',
         path_to_devel_labels = 'D:/Challenges/Compare2020/ComParE2020_Breathing/lab/'):
    # train params
    length_sequence = window_size
    step_sequence = 102400
    batch_size = 45
    epochs = 200
    data_parts = data_parts
    path_to_save_best_model = path_to_save_models
    if not os.path.exists(path_to_save_best_model):
        os.mkdir(path_to_save_best_model)
    path_to_tmp_model = path_to_save_tmp_models
    if not os.path.exists(path_to_tmp_model):
        os.mkdir(path_to_tmp_model)

    # train data

    train_data, train_labels, train_dict, frame_rate = load_data(path_to_train_data, path_to_train_labels, 'train')
    prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps = prepare_data(train_data, train_labels,
                                                                                               train_dict, frame_rate,
                                                                                               length_sequence,
                                                                                               step_sequence)
    # divide train data on parts
    train_parts = divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps,
                                       parts=data_parts, filenames_dict=train_dict)

    # development data
    devel_data, devel_labels, devel_dict, frame_rate = load_data(path_to_devel_data, path_to_devel_labels, 'devel')
    prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps = prepare_data(devel_data, devel_labels,
                                                                                               devel_dict, frame_rate,
                                                                                               length_sequence,
                                                                                               step_sequence)
    # divide development data on parts
    devel_parts = divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps,
                                       parts=data_parts, filenames_dict=devel_dict)

    for index_of_part in range(0, len(train_parts) + len(devel_parts)):
        best_result = 0
        coefs = []
        # form train and validation datasets from rain and development parts of data
        train_dataset, val_dataset = form_train_and_val_datasets(train_parts, devel_parts,
                                                                 index_for_validation_part=index_of_part)
        # unpacking data from train_dataset to make it readeble for keras
        train_d, train_lbs, train_timesteps, _ = extract_list_of_parts(list_of_parts=train_dataset)
        # unpacking data from val_dataset to make it readable for keras
        val_d, val_lbs, val_timesteps, val_filenames_dict = extract_list_of_parts(list_of_parts=val_dataset)
        val_filenames_dict = val_filenames_dict[0]
        # reshaping data to make it readable for keras
        train_d, train_lbs = reshaping_data_for_model(train_d, train_lbs)
        val_d, _val_lbs = reshaping_data_for_model(val_d, val_lbs)
        # load ground truth labels. First half is always comes from train data and the second half - from development part
        if index_of_part < (len(train_parts) + len(devel_parts)) / 2:
            ground_truth_labels = choose_real_labs_only_with_filenames(train_labels, list(val_filenames_dict.values()))
        else:
            ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(val_filenames_dict.values()))
        # create and compile model
        if model_type=='default':
            model = create_model(input_shape=(train_d.shape[-2], train_d.shape[-1]))
        elif model_type=='complex':
            model = create_complex_model(input_shape=(train_d.shape[-2], train_d.shape[-1]))
        model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
        # training process
        for epoch in range(epochs):
            # shuffle train data
            permutations = np.random.permutation(train_d.shape[0])
            train_d, train_lbs = train_d[permutations], train_lbs[permutations]
            model.fit(train_d, train_lbs, batch_size=batch_size, epochs=1,
                      shuffle=True, verbose=1, use_multiprocessing=True,
                      validation_data=(val_d, _val_lbs), callbacks=[MyCustomCallback()])
            # save tmp weights for each training epoch in case we need it in future
            model.save_weights(path_to_tmp_model + 'tmp_model_weights_idx_of_part_' + str(index_of_part)
                               + '_epoch_' + str(epoch) + '.h5')
            # every second epoch check the performance of model on validation dataset
            if epoch % 2 == 0:
                predicted_labels = model.predict(val_d, batch_size=batch_size)
                # average predictions. Data was cutted on windows with overlapping.
                # That is why we need to average predictions in overlapping points
                concatenated_predicted_labels = concatenate_prediction(predicted_labels, val_timesteps,
                                                                       val_filenames_dict)
                prc_coef = scipy.stats.pearsonr(ground_truth_labels.iloc[:, 2].values,
                                                concatenated_predicted_labels.iloc[:, 2].values)
                print('epoch:%i,   Pearson coefficient:%f' % (epoch, prc_coef))
                coefs.append(np.abs(prc_coef[0]))
                # if Pearson coefficient becomes better, we will save model with corresponding weights
                if prc_coef[0] > best_result:
                    best_result = prc_coef[0]
                    model.save_weights(
                        path_to_save_best_model + 'best_model_weights_idx_of_part_' + str(index_of_part) + '.h5')
        # clear RAM
        del model
        K.clear_session()

if __name__ == "__main__":
    main()
