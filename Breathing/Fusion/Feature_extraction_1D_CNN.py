import gc
import os

from tensorflow import keras
from Compare2020.CNN_1D.utils import create_model, load_data, prepare_data, correlation_coefficient_loss
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from Compare2020.Fusion.fusion_utils import transform_deep_features_to_filetype

if __name__ == "__main__":
    # data params for feature extraction
    data_parts = 2
    batch_size = 32
    length_sequence = 256000
    step_sequence = 256000
    path_to_save_extracted_features='extracted_features/'
    path_to_model_for_part_0 = '/content/drive/My Drive/Compare_2020/Breathing_1/trial_2/best_models/best_model_weights_idx_of_part_0.h5'
    path_to_model_for_part_1 = '/content/drive/My Drive/Compare_2020/Breathing_1/trial_2/best_models/best_model_weights_idx_of_part_1.h5'
    path_to_model_for_part_2 = '/content/drive/My Drive/Compare_2020/Breathing_1/trial_2/best_models/best_model_weights_idx_of_part_2.h5'
    path_to_model_for_part_3 = '/content/drive/My Drive/Compare_2020/Breathing_1/trial_2/best_models/best_model_weights_idx_of_part_3.h5'
    paths_to_models = [path_to_model_for_part_0, path_to_model_for_part_1, path_to_model_for_part_2,
                       path_to_model_for_part_3]
    if not os.path.exists(path_to_save_extracted_features):
        os.mkdir(path_to_save_extracted_features)

    # train data
    path_to_train_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    train_data, train_labels, train_dict, frame_rate=load_data(path_to_train_data, path_to_train_labels, 'train')
    prepared_train_data, prepared_train_labels,prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, frame_rate, length_sequence, step_sequence)


    # devel data
    path_to_devel_data='/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_devel_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    devel_data, devel_labels, devel_dict, frame_rate=load_data(path_to_devel_data, path_to_devel_labels, 'devel')
    prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, frame_rate, length_sequence, step_sequence)

    # test data
    path_to_test_data = '/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_test_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'
    test_data, test_labels, test_dict, frame_rate = load_data(path_to_test_data, path_to_test_labels, 'test')
    prepared_test_data, prepared_test_labels, prepared_test_labels_timesteps = prepare_data(test_data, test_labels,
                                                                                            test_dict, frame_rate,
                                                                                            length_sequence,
                                                                                            step_sequence)

    # reshaping for extracting process
    prepared_train_data = prepared_train_data.reshape((prepared_train_data.shape + (1,)))
    prepared_train_data = prepared_train_data.reshape(((-1,) + prepared_train_data.shape[2:]))
    prepared_train_data = prepared_train_data.astype('float32')

    # reshaping for extracting process
    prepared_devel_data = prepared_devel_data.reshape((prepared_devel_data.shape + (1,)))
    prepared_devel_data = prepared_devel_data.reshape(((-1,) + prepared_devel_data.shape[2:]))
    prepared_devel_data = prepared_devel_data.astype('float32')

    # reshaping for extracting process
    print(prepared_test_data.shape)
    prepared_test_data = prepared_test_data.reshape((prepared_test_data.shape + (1,)))
    prepared_test_data = prepared_test_data.reshape(((-1,) + prepared_test_data.shape[2:]))
    prepared_test_data = prepared_test_data.astype('float32')

    # model parameters
    input_shape = (prepared_train_data.shape[-2], prepared_train_data.shape[-1])


    for num_model in range(len(paths_to_models)):
        # create and load model
        model = create_model(input_shape=input_shape)
        model.load_weights(paths_to_models[num_model])
        model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
        # stack lambda_layer above 1D_CNN model to normalize deep features
        lambda_layer = tf.keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))
        # create new model with lambda layer
        extractor = tf.keras.Model(inputs=[model.inputs], outputs=[model.get_layer('average_pooling1d').output])
        extractor.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])
        # extract deep features from train and development data
        extracted_features_train = extractor.predict(prepared_train_data, batch_size=batch_size)
        extracted_features_development = extractor.predict(prepared_devel_data, batch_size=batch_size)
        extracted_features_test = extractor.predict(prepared_test_data, batch_size=batch_size)
        # transform extracted features in convenient format for saving as csv file
        transformed_extracted_features_train=transform_deep_features_to_filetype(deep_features=extracted_features_train,
                                                                                 timesteps_labels=prepared_train_labels_timesteps,
                                                                                 class_dict=train_dict)
        transformed_extracted_features_development=transform_deep_features_to_filetype(deep_features=extracted_features_development,
                                                                                 timesteps_labels=prepared_devel_labels_timesteps,
                                                                                 class_dict=devel_dict)
        transformed_extracted_features_test=transform_deep_features_to_filetype(deep_features=extracted_features_test,
                                                                                 timesteps_labels=prepared_test_labels_timesteps,
                                                                                 class_dict=test_dict)
        # save transformed deep features as csv file
        transformed_extracted_features_train.to_csv('deep_features_train_model_%i.csv'%(num_model), index=False)
        transformed_extracted_features_development.to_csv('deep_features_devel_model_%i.csv' % (num_model), index=False)
        transformed_extracted_features_test.to_csv('deep_features_test_model_%i.csv' % (num_model), index=False)
        # clear RAM
        del extractor
        del model
        K.clear_session()
        gc.collect()