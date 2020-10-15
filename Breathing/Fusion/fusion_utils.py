import re
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import os
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler


def transform_deep_features_to_filetype(deep_features,timesteps_labels,class_dict,columns=['filename', 'timeFrame', 'deep_features']):
    reshaped_timesteps=timesteps_labels.reshape((timesteps_labels.shape[0],-1))
    new_shape=reshaped_timesteps.shape+(-1,)
    reshaped_deep_features=deep_features.reshape(new_shape).astype('float32')
    print(reshaped_deep_features.shape)
    result=pd.DataFrame(columns=columns, data=np.zeros((timesteps_labels.shape[0]*timesteps_labels.shape[1]*timesteps_labels.shape[2],3)))
    result['filename']=result['filename'].astype('str')
    result['timeFrame']=result['timeFrame'].astype('float32')
    result['deep_features']=result['deep_features'].astype('object')
    print(result.shape)
    idx_result=0
    for instance_idx in range(reshaped_deep_features.shape[0]):
      for timesteps_idx in range(reshaped_timesteps.shape[1]):
        result['filename'].iloc[idx_result]=class_dict[instance_idx]
        result['timeFrame'].iloc[idx_result]=reshaped_timesteps[instance_idx,timesteps_idx]
        result['deep_features'].iloc[idx_result]=[reshaped_deep_features[instance_idx,timesteps_idx]]
        idx_result=idx_result+1
    return result


def load_deep_features_1d_CNN(path_to_deep_features_1D_CNN, path_to_labels, path_to_deep_features_train_2D_CNN,
                              path_to_deep_features_dev_2D_CNN, prefix, exceptions_filenames=[]):
    # labels
    labels = pd.read_csv(path_to_labels + 'labels.csv', sep=',')
    labels = labels.loc[labels['filename'].str.contains(prefix)]
    if len(exceptions_filenames) > 0: labels = del_some_filenames(labels, exceptions_filenames)
    if not prefix == 'test':
        labels['upper_belt'] = labels['upper_belt'].astype('float32')
    else:
        labels['upper_belt'] = 0
        labels['upper_belt'] = labels['upper_belt'].astype('float32')

    # 2D CNN deep features
    train = pd.DataFrame(data=np.load(path_to_deep_features_train_2D_CNN, allow_pickle=True))
    if not path_to_deep_features_dev_2D_CNN == '':
        dev = pd.DataFrame(np.load(path_to_deep_features_dev_2D_CNN, allow_pickle=True))
        train_dev = pd.concat([train, dev])
    else:
        train_dev = train
    print('train_dev columns', train_dev.columns)
    train_dev = train_dev.loc[train_dev[0].str.contains(prefix)]
    train_dev = train_dev.sort_values(by=[0, 1], axis=0)

    data = pd.read_csv(path_to_deep_features_1D_CNN)
    tmp = data.drop(columns=['filename', 'timeFrame']).applymap(convert_string_to_ndarray)
    data['deep_features'] = tmp['deep_features']
    data['deep_features'] = data['deep_features'].astype('object')
    if len(exceptions_filenames) > 0: data = del_some_filenames(data, exceptions_filenames)

    for i in range(data.shape[0]):
        arr = data.iloc[i, 2][0]
        add_arr = np.array(train_dev.iloc[i, 3:].values).reshape((-1,))
        result_arr = np.concatenate((arr, add_arr))
        data.iloc[i, 2][0] = result_arr.astype('float32')

    files = np.unique(labels['filename'])
    filename_dict = {}
    for i in range(len(files)):
        filename_dict[i] = files[i]
    return data, labels, filename_dict


def normalize_features(data, scaler=None):
    result = np.zeros(shape=(data.shape[0], data.iloc[0, 2][0][:256].shape[0]))
    for i in range(data.shape[0]):
        result[i] = data.iloc[i, 2][0][:256]
    if scaler == None:
        scaler = StandardScaler()
        scaler = scaler.fit(result)
    result = scaler.transform(result)
    for i in range(data.shape[0]):
        data.iloc[i, 2][0][:256] = result[i]
    return data, scaler


def vector_normalization(data):
    for i in range(data.shape[0]):
        tmp = data.iloc[i, 2][0]
        sum = np.sqrt(np.square(tmp).sum())
        tmp = tmp / sum
        data.iloc[i, 2][0] = tmp
    return data


def how_many_windows_do_i_need(length_sequence, window_size, step):
    start_idx = 0
    counter = 0
    while True:
        if start_idx + window_size > length_sequence:
            break
        start_idx += step
        counter += 1
    if start_idx != length_sequence:
        counter += 1
    return counter


def unpack_row_dataFrame_to_2D_array(row):
    result = np.zeros(shape=(row.shape[0], row.iloc[0][0].shape[0]))
    for i in range(row.shape[0]):
        result[i] = row.iloc[i][0]
    return result


def convert_string_to_ndarray(string_to_arr):
    # print(string_to_arr)
    res = re.sub("[^0-9e,\n+-.]", "", string_to_arr)
    res = res[:res.rindex(',')]
    res = np.fromstring(res, dtype='float32', sep=",")
    return [res]


def prepare_data(data, labels, filename_dict, size_window, step_window):

    files = np.unique(labels['filename'])
    num_features = data['deep_features'].iloc[0][0].shape[0]
    num_instances = files.shape[0]
    num_windows = int(how_many_windows_do_i_need(data[data['filename'] == files[0]].shape[0], size_window, step_window))
    print('data.shape[1]', data.shape[1])
    print('num_windows:', num_windows)
    new_data = np.zeros(shape=(num_instances, num_windows, size_window, num_features))
    new_labels = np.zeros(shape=(num_instances, num_windows, size_window))
    new_labels_timesteps = np.zeros(shape=new_labels.shape)
    for instance_idx in range(num_instances):
        start_idx_data = 0
        start_idx_label = 0
        temp_labels = labels[labels['filename'] == filename_dict[instance_idx]]
        temp_labels = temp_labels.drop(columns=['filename'])
        temp_labels = temp_labels.values
        temp_data = data[data['filename'] == filename_dict[instance_idx]]
        temp_data = temp_data['deep_features']
        # print('shape temp data:', temp_data.shape)
        for num_window_idx in range(num_windows - 1):
            row = temp_data.iloc[start_idx_data:start_idx_data + size_window]
            # print(row.shape)
            unpacked_row = unpack_row_dataFrame_to_2D_array(row)
            new_data[instance_idx, num_window_idx] = unpacked_row
            new_labels[instance_idx, num_window_idx] = temp_labels[start_idx_label:start_idx_label + size_window, 1]
            new_labels_timesteps[instance_idx, num_window_idx] = temp_labels[
                                                                 start_idx_label:start_idx_label + size_window, 0]
            start_idx_data += step_window
            start_idx_label += step_window
        if start_idx_data + size_window >= temp_data.shape[0]:
            row = temp_data.iloc[(temp_data.shape[0] - size_window):temp_data.shape[0]]
            unpacked_row = unpack_row_dataFrame_to_2D_array(row)
            new_data[instance_idx, num_windows - 1] = unpacked_row
            new_labels[instance_idx, num_windows - 1] = temp_labels[
                                                        temp_labels.shape[0] - size_window:temp_labels.shape[0], 1]
            new_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[
                                                                  temp_labels.shape[0] - size_window:temp_labels.shape[
                                                                      0], 0]
        else:
            row = temp_data.iloc[start_idx_data:start_idx_data + size_window]
            unpacked_row = unpack_row_dataFrame_to_2D_array(row)
            new_data[instance_idx, num_windows - 1] = unpacked_row
            new_labels[instance_idx, num_windows - 1] = temp_labels[start_idx_label:start_idx_label + size_window, 1]
            new_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[
                                                                  start_idx_label:start_idx_label + size_window, 0]
    return new_data, new_labels, new_labels_timesteps


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


def extract_and_reshape_list_of_parts(list_of_parts):
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

def concatenate_prediction(predicted_values, timesteps_labels, class_dict, columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
  predicted_values=predicted_values.reshape(timesteps_labels.shape)
  result_predicted_values=pd.DataFrame(columns=columns_for_real_labels, dtype='float32')
  result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
  for instance_idx in range(predicted_values.shape[0]):
      predicted_values_tmp=predicted_values[instance_idx].reshape((-1,1))
      timesteps_labels_tmp=timesteps_labels[instance_idx].reshape((-1,1))
      tmp=pd.DataFrame(columns=['timeFrame', 'upper_belt'], data=np.concatenate((timesteps_labels_tmp, predicted_values_tmp), axis=1))
      tmp=tmp.groupby(by=['timeFrame']).mean().reset_index()
      tmp['filename']=class_dict[instance_idx]
      result_predicted_values=result_predicted_values.append(tmp.copy(deep=True))
  result_predicted_values['timeFrame']=result_predicted_values['timeFrame'].astype('float32')
  result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
  return result_predicted_values[columns_for_real_labels]

def create_LSTM_model(input_shape):
  model=tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(input_shape=input_shape, units=512, return_sequences=True))
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
  model.add(tf.keras.layers.Dense(1, activation='tanh'))
  model.add(tf.keras.layers.Flatten())
  model.summary()
  return model

class MyCustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
      gc.collect()

def del_some_filenames(labels, filenames):
  for filename in filenames:
      labels.drop(labels[labels['filename']==filename].index, inplace=True)
  return labels

def correlation_coefficient_loss(y_true, y_pred):
  x=y_true
  y=y_pred
  mx=K.mean(x, axis=1, keepdims=True)
  my=K.mean(y, axis=1, keepdims=True)
  xm,ym=x-mx,y-my
  r_num=K.sum(tf.multiply(xm, ym), axis=1)
  sum_square_x=K.sum(K.square(xm), axis=1)
  sum_square_y = K.sum(K.square(ym), axis=1)
  sqrt_x = tf.sqrt(sum_square_x)
  sqrt_y = tf.sqrt(sum_square_y)
  r_den=tf.multiply(sqrt_x, sqrt_y)
  result=tf.divide(r_num, r_den)
  result=K.mean(result)
  return 1 - result


def choose_real_labs_only_with_filenames(labels, filenames):
  return labels[labels['filename'].isin(filenames)]