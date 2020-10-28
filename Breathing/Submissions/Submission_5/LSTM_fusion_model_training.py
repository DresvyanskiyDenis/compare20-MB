from Breathing.Fusion_system.feature_level_fusion_model_training import main

if __name__ == "__main__":
    # please, to run script, specify paths to data and prefixes of deep embeddings for 1D and 2D CNNs
    main(window_size=1600, batch_size=15, num_parts=4, epochs=100,
             path_to_save_models='../../logs/LSTM_models/',
             path_to_train_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/',
             path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/',
             path_to_devel_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/',
             path_to_devel_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/',
             path_to_train_data_2D_CNN='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/',
             path_to_devel_data_2D_CNN='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/',
             deep_features_1D_prefix_train='deep_features_train_model_',
             deep_features_1D_prefix_dev='deep_features_devel_model_',
             deep_features_2D_prefix_train='deep_features_train_fold_',
             deep_features_2D_prefix_dev='deep_features_dev_fold_')
