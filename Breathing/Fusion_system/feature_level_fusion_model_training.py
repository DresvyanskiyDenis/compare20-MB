from scipy.stats import pearsonr
import os
import pandas as pd
from tensorflow.keras import backend as K
import numpy as np


from Compare2020.Fusion.fusion_utils import load_deep_features_1d_CNN, normalize_features, vector_normalization, \
    prepare_data, divide_data_on_parts, form_train_and_val_datasets, extract_and_reshape_list_of_parts, \
    reshaping_data_for_model, choose_real_labs_only_with_filenames, create_LSTM_model, correlation_coefficient_loss, \
    MyCustomCallback, concatenate_prediction

if __name__ == "__main__":
    window=1600
    step=int(window*2/5.)
    batch_size=15
    num_parts=4
    epochs=100
    path_to_save_model='best_models/'
    if not os.path.exists(path_to_save_model):
      os.mkdir(path_to_save_model)

    # 1D CNN deep features
    # train data
    path_to_train_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_train_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'
    # devel data
    path_to_devel_data='/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features/'
    path_to_devel_labels='/content/drive/My Drive/ComParE2020_Breathing/lab/'

    # 2D CNN deep features
    # train data
    path_to_train_data_2D_CNN = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/'
    # devel data
    path_to_devel_data_2D_CNN = '/content/drive/My Drive/Compare_2020/Breathing_1/Deep_features_Danila/'
    bests=[]
    for num_part in range(0,num_parts):
          best_result=0
          deep_features_1D_prefix_train = 'deep_features_train_model_'
          deep_features_1D_prefix_dev = 'deep_features_devel_model_'
          deep_features_2D_prefix_train = 'deep_features_train_fold_'
          deep_features_2D_prefix_dev = 'deep_features_dev_fold_'
          train_data, train_labels, train_dict=load_deep_features_1d_CNN(path_to_deep_features_1D_CNN=path_to_train_data+deep_features_1D_prefix_train+str(num_part)+'.csv',
                                                                      path_to_labels=path_to_train_labels,
                                                                      path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN+deep_features_2D_prefix_train+str(num_part)+'.npy',
                                                                      path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN+deep_features_2D_prefix_dev+str(num_part)+'.npy',
                                                                      prefix='train')
          print('train_data before:', train_data)
          devel_data, devel_labels, devel_dict=load_deep_features_1d_CNN(path_to_deep_features_1D_CNN=path_to_devel_data+deep_features_1D_prefix_dev+str(num_part)+'.csv',
                                                                      path_to_labels=path_to_devel_labels,
                                                                      path_to_deep_features_train_2D_CNN=path_to_train_data_2D_CNN+deep_features_2D_prefix_train+str(num_part)+'.npy',
                                                                      path_to_deep_features_dev_2D_CNN=path_to_devel_data_2D_CNN+deep_features_2D_prefix_dev+str(num_part)+'.npy',
                                                                      prefix='devel',
                                                                      exceptions_filenames=['devel_00.wav'])
          total_data=pd.concat([train_data, devel_data], axis=0)
          total_data, scaler=normalize_features(total_data)
          total_data=vector_normalization(total_data)
          print('train_data after:', train_data)
          train_data=total_data.loc[total_data['filename'].str.contains('train')]
          devel_data=total_data.loc[total_data['filename'].str.contains('dev')]
          print('train_data after1:', train_data)
          print('devel_data after1:', devel_data)
          prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps=prepare_data(train_data, train_labels, train_dict, window, step)
          train_parts=divide_data_on_parts(prepared_train_data, prepared_train_labels, prepared_train_labels_timesteps, filenames_dict=train_dict)

          # devel data
          prepared_devel_data, prepared_devel_labels,prepared_devel_labels_timesteps=prepare_data(devel_data, devel_labels, devel_dict, window, step)
          devel_parts=divide_data_on_parts(prepared_devel_data, prepared_devel_labels, prepared_devel_labels_timesteps, filenames_dict=devel_dict)
          val_loss=[]
          train_dataset, val_dataset=form_train_and_val_datasets(train_parts, devel_parts, index_for_validation_part=num_part)
          train_d, train_lbs, train_timesteps, _ = extract_and_reshape_list_of_parts(list_of_parts=train_dataset)
          val_d, val_lbs, val_timesteps, val_filenames_dict=extract_and_reshape_list_of_parts(list_of_parts=val_dataset)
          val_filenames_dict=val_filenames_dict[0]
          train_d, train_lbs=reshaping_data_for_model(train_d, train_lbs)
          val_d, _val_lbs=reshaping_data_for_model(val_d, val_lbs)

          if num_part<(len(train_parts)+len(devel_parts))/2:
              ground_truth_labels=choose_real_labs_only_with_filenames(train_labels,list(val_filenames_dict.values()))
          else:
              ground_truth_labels = choose_real_labs_only_with_filenames(devel_labels, list(val_filenames_dict.values()))

          model=create_LSTM_model(input_shape=(train_d.shape[-2], train_d.shape[-1]))
          model.compile(optimizer='Nadam', loss=correlation_coefficient_loss, metrics=['mse'])

          for epoch in range(epochs):
            permutations=np.random.permutation(train_d.shape[0])
            train_d, train_lbs=train_d[permutations], train_lbs[permutations]
            history=model.fit(train_d, train_lbs, batch_size=batch_size, epochs=1, verbose=1, use_multiprocessing=True,
                    validation_data=(val_d, _val_lbs), callbacks=[MyCustomCallback()])
            predicted_labels = model.predict(val_d, batch_size=batch_size)
            concatenated_predicted_labels=concatenate_prediction(predicted_labels, val_timesteps, val_filenames_dict)
            prc_coef=pearsonr(ground_truth_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
            print('epoch:', epoch, '     pearson:', prc_coef)
            val_loss.append(prc_coef[0])
            pd.DataFrame(columns=['prc_coef'], data=val_loss).to_csv(
                        path_to_save_model+'val_prc_coefs_part_'+str(num_part)+'.csv', index=False)
            if prc_coef[0] > best_result:
              best_result = prc_coef[0]
              model.save_weights(path_to_save_model+'best_model_weights_idx_of_part_'+str(num_part)+'.h5')
          bests.append(best_result)
          del model
          K.clear_session()

    print('bests:', bests)
