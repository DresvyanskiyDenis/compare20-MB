from scipy.stats import pearsonr
import os
import pandas as pd

from Breathing.Fusion_system.fusion_utils import load_deep_features_1d_CNN, normalize_features, vector_normalization, \
    prepare_data, divide_data_on_parts, form_train_and_val_datasets, extract_and_reshape_list_of_parts, \
    reshaping_data_for_model, choose_real_labs_only_with_filenames, create_LSTM_model, correlation_coefficient_loss, \
    MyCustomCallback, concatenate_prediction

if __name__ == "__main__":
    window = 1600
    step = int(window * 2 / 5.)
    batch_size = 15
    num_parts = 4
    epochs = 100
    path_to_model_for_part_0 = '/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_0.h5'
    path_to_model_for_part_1 = '/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_1.h5'
    path_to_model_for_part_2 = '/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_2.h5'
    path_to_model_for_part_3 = '/content/drive/My Drive/Compare_2020/Breathing_1/end_to_end/best_models/best_model_weights_idx_of_part_3.h5'
    paths_to_models = [path_to_model_for_part_0, path_to_model_for_part_1, path_to_model_for_part_2,
                       path_to_model_for_part_3]
    # 1D CNN deep features
    # train data
    path_to_train_data = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_train_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'
    # devel data
    path_to_devel_data = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_devel_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'

    # 2D CNN deep features
    # train data
    path_to_train_data_2D_CNN = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/'
    # devel data
    path_to_devel_data_2D_CNN = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/'
    bests = []
    total_predicted_labels = pd.DataFrame(columns=['filename', 'timeFrame', 'upper_belt'])
    for num_part in range(num_parts):
        best_result = 0
        deep_features_1D_prefix_train = 'deep_features_train_model_'
        deep_features_1D_prefix_dev = 'deep_features_devel_model_'
        deep_features_2D_prefix_train = 'deep_features_train_fold_'
        deep_features_2D_prefix_dev = 'deep_features_dev_fold_'
        train_data, train_labels, train_dict = load_deep_features_1d_CNN(
            path_to_deep_features_1D_CNN=path_to_train_data + deep_features_1D_prefix_train + str(num_part) + '.csv',
            path_to_labels=path_to_train_labels,
            path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN + deep_features_2D_prefix_train + str(num_part) + '.npy',
            path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN + deep_features_2D_prefix_dev + str(num_part) + '.npy',
            prefix='train')
        # devel data
        devel_data, devel_labels, devel_dict = load_deep_features_1d_CNN(
            path_to_deep_features_1D_CNN=path_to_devel_data + deep_features_1D_prefix_dev + str(num_part) + '.csv',
            path_to_labels=path_to_devel_labels,
            path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN + deep_features_2D_prefix_train + str(num_part) + '.npy',
            path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN + deep_features_2D_prefix_dev + str(num_part) + '.npy',
            prefix='devel',
            exceptions_filenames=['devel_00.wav'])

        total_data = pd.concat([train_data, devel_data], axis=0)
        total_data, scaler = normalize_features(total_data)
        total_data = vector_normalization(total_data)
        train_data = total_data.loc[total_data['filename'].str.contains('train')]
        devel_data = total_data.loc[total_data['filename'].str.contains('dev')]

        prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps = prepare_data(train_data, train_labels,
                                                                                                   train_dict, window, step)
        train_parts = divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps,
                                           filenames_dict=train_dict)
        prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps = prepare_data(devel_data, devel_labels,
                                                                                                   devel_dict, window, step)
        devel_parts = divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps,
                                           filenames_dict=devel_dict)
        val_loss = []
        train_dataset, val_dataset = form_train_and_val_datasets(train_parts, devel_parts,
                                                                 index_for_validation_part=num_part)
        train_d, train_lbs, train_timesteps, _ = extract_and_reshape_list_of_parts(list_of_parts=train_dataset)
        val_d, val_lbs, val_timesteps, val_filenames_dict = extract_and_reshape_list_of_parts(list_of_parts=val_dataset)
        val_filenames_dict = val_filenames_dict[0]
        train_d, train_lbs = reshaping_data_for_model(train_d, train_lbs)
        val_d, _val_lbs = reshaping_data_for_model(val_d, val_lbs)
        if num_part < (len(train_parts) + len(devel_parts)) / 2:
            ground_truth_labels = choose_real_labs_only_with_filenames(train_labels, list(val_filenames_dict.values()))
        else:
            ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(val_filenames_dict.values()))
        model = create_LSTM_model(input_shape=(train_d.shape[-2], train_d.shape[-1]))
        model.load_weights(paths_to_models[num_part])
        model.compile(optimizer='Nadam', loss=correlation_coefficient_loss, metrics=['mse'])
        predicted_labels = model.predict(val_d, batch_size=batch_size)
        concatenated_predicted_labels = concatenate_prediction(predicted_labels, val_timesteps, val_filenames_dict)
        prc_coef = pearsonr(ground_truth_labels.iloc[:, 2].values, concatenated_predicted_labels.iloc[:, 2].values)
        print('part number', num_part, 'corr:', prc_coef)
        total_predicted_labels = total_predicted_labels.append(concatenated_predicted_labels)

    total_predicted_labels.to_csv('predictions_5_trial.csv', index=False)