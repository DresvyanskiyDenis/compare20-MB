from scipy.stats import pearsonr
import os
import pandas as pd

from Breathing.Fusion_system.fusion_utils import load_deep_features_1d_CNN, normalize_features, vector_normalization, \
    prepare_data, divide_data_on_parts, form_train_and_val_datasets, extract_and_reshape_list_of_parts, \
    reshaping_data_for_model, choose_real_labs_only_with_filenames, create_LSTM_model, correlation_coefficient_loss, \
    MyCustomCallback, concatenate_prediction

if __name__ == "__main__":
    # to run script, please, specify all paths to data and prefixes
    window=1600
    step=int(window*2/5.)
    batch_size=15
    num_parts=4
    epochs=100
    path_to_model_for_part_0='/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_0.h5'
    path_to_model_for_part_1='/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_1.h5'
    path_to_model_for_part_2='/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_2.h5'
    path_to_model_for_part_3='/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_3.h5'

    # 1D CNN deep features
    # train data
    path_to_train_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    # devel data
    path_to_devel_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_devel_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    # test data
    path_to_test_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_test_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'

    # 2D CNN deep features
    # train data
    path_to_train_data_2D_CNN=''
    # devel data
    path_to_devel_data_2D_CNN=''
    # test data
    path_to_test_data_2D_CNN=''

    deep_features_1D_prefix_train = 'deep_features_train_model_'
    deep_features_1D_prefix_dev = 'deep_features_devel_model_'
    deep_features_1D_prefix_test = 'deep_features_test_model_'
    deep_features_2D_prefix_train = 'deep_features_train_fold_'
    deep_features_2D_prefix_dev = 'deep_features_dev_fold_'
    deep_features_2D_prefix_test = 'deep_features_test_fold_'

    paths_to_models=[path_to_model_for_part_0, path_to_model_for_part_1, path_to_model_for_part_2, path_to_model_for_part_3]


    bests=[]
    total_predicted_labels=pd.DataFrame(columns=['filename', 'timeFrame','upper_belt'])
    for num_part in range(num_parts):
        best_result=0



        train_data, train_labels, train_dict=load_deep_features_1d_CNN(path_to_deep_features_1D_CNN=path_to_train_data+deep_features_1D_prefix_train+str(num_part)+'.csv',
                                                                  path_to_labels=path_to_train_labels,
                                                                  path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN+deep_features_2D_prefix_train+str(num_part)+'.npy',
                                                                  path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN+deep_features_2D_prefix_dev+str(num_part)+'.npy',
                                                                  prefix='train')
        # devel data
        devel_data, devel_labels, devel_dict=load_deep_features_1d_CNN(path_to_deep_features_1D_CNN=path_to_devel_data+deep_features_1D_prefix_dev+str(num_part)+'.csv',
                                                                  path_to_labels=path_to_devel_labels,
                                                                  path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN+deep_features_2D_prefix_train+str(num_part)+'.npy',
                                                                  path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN+deep_features_2D_prefix_dev+str(num_part)+'.npy',
                                                                  prefix='devel',
                                                                  exceptions_filenames=['devel_00.wav'])
        test_data, test_labels, test_dict=load_deep_features_1d_CNN(path_to_deep_features_1D_CNN=path_to_test_data+deep_features_1D_prefix_test+str(num_part)+'.csv',
                                                                  path_to_labels=path_to_test_labels,
                                                                  path_to_deep_features_train_2D_CNN=path_to_test_data_2D_CNN+deep_features_2D_prefix_test+str(num_part)+'.npy',
                                                                  path_to_deep_features_dev_2D_CNN='',
                                                                  prefix='test',
                                                                  exceptions_filenames=['devel_00.wav'])
        train_dev_data=pd.concat([train_data, devel_data], axis=0)
        train_dev_data, scaler=normalize_features(train_dev_data)
        test_data, scaler=normalize_features(test_data, scaler)
        test_data=vector_normalization(test_data)
        prepared_test_data, prepared_test_labels,prepared_test_labels_timesteps=prepare_data(test_data, test_labels, test_dict, window, step)
        print(prepared_test_data.shape)
        prepared_test_data=prepared_test_data.reshape((-1,prepared_test_data.shape[2], prepared_test_data.shape[3]))
        print(prepared_test_data.shape)

        model=create_LSTM_model(input_shape=(prepared_test_data.shape[-2], prepared_test_data.shape[-1]))
        model.load_weights(paths_to_models[num_part])
        model.compile(optimizer='Nadam', loss=correlation_coefficient_loss, metrics=['mse'])
        predicted_labels = model.predict(prepared_test_data, batch_size=batch_size)
        concatenated_predicted_labels=concatenate_prediction(predicted_labels, prepared_test_labels_timesteps, test_dict)
        print('part number', num_part,'done...')
        concatenated_predicted_labels.to_csv('test_prediction_deep_features_fold_'+str(num_part)+'.csv')