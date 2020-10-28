from Breathing.Fusion_system.Feature_extraction_1D_CNN import main

if __name__ == "__main__":
    # please, to run script, specify paths to data and models
    window_size = 256000
    data_parts = 2
    batch_size = 32
    paths_to_models = '/content/drive/My Drive/Compare_2020/Breathing_1/trial_2/best_models/'
    path_to_train_data = '/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_train_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'
    path_to_devel_data = '/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_devel_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'
    path_to_test_data = '/content/drive/My Drive/ComParE2020_Breathing/wav/'
    path_to_test_labels = '/content/drive/My Drive/ComParE2020_Breathing/lab/'
    path_to_output_features = '../../logs/1D_CNN_predictions/'
    main(window_size=window_size, data_parts=data_parts, batch_size=batch_size,
         paths_to_models=paths_to_models,
         path_to_train_data=path_to_train_data,
         path_to_train_labels = path_to_train_labels,
         path_to_devel_data=path_to_devel_data,
         path_to_devel_labels = path_to_devel_labels,
         path_to_test_data=path_to_test_data,
         path_to_test_labels = path_to_test_labels,
         path_to_output_features=path_to_output_features)