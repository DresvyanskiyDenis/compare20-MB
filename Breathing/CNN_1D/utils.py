import os
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
from keras import Sequential
from keras.layers import Conv1D, MaxPool1D, LSTM, Dense, Dropout, Flatten, TimeDistributed
from keras import backend as K
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

def load_data(path_to_data, path_to_labels, prefix):
    # labels
    labels=pd.read_csv(path_to_labels+'labels.csv', sep=',')
    labels = labels.loc[labels['filename'].str.contains(prefix)]
    if not prefix=='test':
        labels['upper_belt']=labels['upper_belt'].astype('float32')
    else:
        labels['upper_belt']=0
        labels['upper_belt']=labels['upper_belt'].astype('float32')

    # data
    fs, example = wavfile.read(path_to_data + labels.iloc[0, 0])
    result_data = np.zeros(shape=(np.unique(labels['filename']).shape[0], example.shape[0]))

    files=np.unique(labels['filename'])
    filename_dict={}
    for i in range(len(files)):
        frame_rate, data = wavfile.read(path_to_data+files[i])
        result_data[i]=data
        filename_dict[i]=files[i]
    return result_data, labels, filename_dict, frame_rate


def how_many_windows_do_i_need(length_sequence, window_size, step):
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter

def prepare_data(data, labels, class_to_filename_dict, frame_rate, size_window, step_for_window):
    label_rate=25 # 25 Hz label rate
    num_windows=how_many_windows_do_i_need(data.shape[1],size_window, step_for_window)
    new_data=np.zeros(shape=(data.shape[0],int(num_windows),size_window))
    length_of_label_window=int(size_window/frame_rate*label_rate)
    step_of_label_window=int(length_of_label_window*(step_for_window/size_window))
    new_labels=np.zeros(shape=(np.unique(labels['filename']).shape[0], int(num_windows),length_of_label_window ))
    new_labels_timesteps=np.zeros(shape=new_labels.shape)
    for instance_idx in range(data.shape[0]):
        start_idx_data=0
        start_idx_label=0
        temp_labels=labels[labels['filename']==class_to_filename_dict[instance_idx]]
        temp_labels=temp_labels.drop(columns=['filename'])
        temp_labels=temp_labels.values
        for windows_idx in range(num_windows-1):
            new_data[instance_idx,windows_idx]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels[instance_idx,windows_idx]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 1]
            new_labels_timesteps[instance_idx, windows_idx]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
        if start_idx_data+size_window>=data.shape[1]:
            new_data[instance_idx,num_windows-1]=data[instance_idx, data.shape[1]-size_window:data.shape[1]]
            new_labels[instance_idx, num_windows-1]=temp_labels[temp_labels.shape[0]-length_of_label_window:temp_labels.shape[0],1]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[temp_labels.shape[0]-length_of_label_window:temp_labels.shape[0],0]
        else:
            new_data[instance_idx,num_windows-1]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels[instance_idx,num_windows-1]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 1]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
    return new_data, new_labels, new_labels_timesteps


def instance_normalization(data):
    for instance_idx in range(data.shape[0]):
        scaler=StandardScaler()
        temp_data=data[instance_idx].reshape((-1,1))
        temp_data=scaler.fit_transform(temp_data)
        temp_data=temp_data.reshape((data.shape[1:]))
        data[instance_idx]=temp_data
    return data

def sample_standart_normalization(data, scaler=None):
    tmp_shape=data.shape
    tmp_data=data.reshape((-1,1))
    if scaler==None:
        scaler=StandardScaler()
        tmp_data=scaler.fit_transform(tmp_data)
    else:
        tmp_data=scaler.transform(tmp_data)
    data=tmp_data.reshape(tmp_shape)
    return data

def sample_minmax_normalization(data, min=None, max=None):
    result_shape=data.shape
    tmp_data=data.reshape((-1))
    if max==None:
        max=np.max(tmp_data)
    if min == None:
        min=np.min(tmp_data)
    tmp_data=2*(tmp_data-min)/(max-min)-1
    data=tmp_data.reshape(result_shape)
    return data, min, max

def create_model(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())
    return model

def identity_block(input_tensor, filters, block_number):
    filter1, filter2, filter3 = filters

    x = tf.keras.layers.Conv1D(filters=filter1, kernel_size=1, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)

    x = tf.keras.layers.Conv1D(filters=filter2, kernel_size=5, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)

    x = tf.keras.layers.Conv1D(filters=filter3, kernel_size=1, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization(name='last_identity_bn_block_' + str(block_number))(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, filters, block_number):
    filter1, filter2, filter3 = filters
    x = tf.keras.layers.Conv1D(filters=filter1, kernel_size=1, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=filter2, kernel_size=5, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=filter3, kernel_size=1, strides=1, activation=None, padding='same',
                               use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization(name='last_conv_bn_block_' + str(block_number))(x)
    shortcut = tf.keras.layers.Conv1D(filters=filter3, kernel_size=1, strides=1, activation=None,
                                      use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(name='shortcut_bn_block_' + str(block_number))(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_complex_model(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation=None, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
    x = tf.keras.layers.BatchNormalization(name='last_conv_bn_block_1')(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    output_block1 = tf.keras.layers.MaxPool1D(pool_size=10)(x)

    x = conv_block(output_block1, [64, 64, 256], 2)
    x = identity_block(x, [64, 64, 256], 'identity_1')
    x = identity_block(x, [64, 64, 256], 'identity_2')
    output_block2 = tf.keras.layers.AvgPool1D(pool_size=8)(x)

    x = conv_block(output_block2, [128, 128, 512], 3)
    x = identity_block(x, [128, 128, 512], 'identity_3')
    x = identity_block(x, [128, 128, 512], 'identity_4')
    output_block3 = tf.keras.layers.AvgPool1D(pool_size=8)(x)

    x = tf.keras.layers.LSTM(512, return_sequences=True)(output_block3)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=[input], outputs=[x])
    print(model.summary())
    return model

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
    #tf.print('result:', result)
    result=K.mean(result)
    #tf.print('mean result:', result)
    return 1 - result

def pearson_coef(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)

def concatenate_prediction(true_values, predicted_values, timesteps_labels, class_dict):
    predicted_values=predicted_values.reshape(timesteps_labels.shape)
    tmp=np.zeros(shape=(true_values.shape[0],3))
    result_predicted_values=pd.DataFrame(data=tmp, columns=true_values.columns, dtype='float32')
    result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
    index_temp=0
    for instance_idx in range(predicted_values.shape[0]):
        timesteps=np.unique(timesteps_labels[instance_idx])
        for timestep in timesteps:
            # assignment for filename and timestep
            result_predicted_values.iloc[index_temp,0]=class_dict[instance_idx]
            result_predicted_values.iloc[index_temp,1]=timestep
            # calculate mean of windows
            result_predicted_values.iloc[index_temp,2]=np.mean(predicted_values[instance_idx,timesteps_labels[instance_idx]==timestep])
            index_temp+=1
        #print('concatenation...instance:', instance_idx, '  done')

    return result_predicted_values

def load_test_data(path_to_data, path_to_labels, prefix):
    # labels
    labels = pd.read_csv(path_to_labels + 'labels.csv', sep=',')
    labels = labels.loc[labels['filename'].str.contains(prefix)]
    #labels.drop(columns=['upper_belt'], inplace=True)
    # data
    fs, example = wavfile.read(path_to_data + labels.iloc[0, 0])
    result_data = np.zeros(shape=(np.unique(labels['filename']).shape[0], example.shape[0]))
    files = np.unique(labels['filename'])
    filename_dict = {}
    for i in range(len(files)):
        frame_rate, data = wavfile.read(path_to_data + files[i])
        result_data[i] = data
        filename_dict[i] = files[i]
    return result_data, labels, filename_dict, frame_rate

def prepare_test_data(data, labels, class_to_filename_dict, frame_rate, size_window, step_for_window):
    label_rate=25 # 25 Hz label rate
    num_windows=how_many_windows_do_i_need(data.shape[1],size_window, step_for_window)
    new_data=np.zeros(shape=(data.shape[0],int(num_windows),size_window))
    length_of_label_window=int(size_window/frame_rate*label_rate)
    step_of_label_window=int(length_of_label_window*(step_for_window/size_window))
    new_labels_timesteps=np.zeros(shape=(new_data.shape[0], int(num_windows),length_of_label_window ))
    for instance_idx in range(data.shape[0]):
        start_idx_data=0
        start_idx_label=0
        temp_labels=labels[labels['filename']==class_to_filename_dict[instance_idx]]
        temp_labels=temp_labels.drop(columns=['filename','upper_belt'])
        temp_labels=temp_labels.values.reshape((-1,1))
        for windows_idx in range(num_windows-1):
            new_data[instance_idx,windows_idx]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels_timesteps[instance_idx, windows_idx]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
        if start_idx_data+size_window>=data.shape[1]:
            new_data[instance_idx,num_windows-1]=data[instance_idx, data.shape[1]-size_window:data.shape[1]]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[temp_labels.shape[0]-length_of_label_window:temp_labels.shape[0],0]
        else:
            new_data[instance_idx,num_windows-1]=data[instance_idx,start_idx_data:start_idx_data+size_window]
            new_labels_timesteps[instance_idx, num_windows-1]=temp_labels[start_idx_label:start_idx_label+length_of_label_window, 0]
            start_idx_data+=step_for_window
            start_idx_label+=step_of_label_window
    return new_data, new_labels_timesteps

def concatenate_prediction_test(true_values, predicted_values, timesteps_labels, class_dict):
    predicted_values=predicted_values.reshape(timesteps_labels.shape)
    tmp=np.zeros(shape=(true_values.shape[0],3))
    result_predicted_values=pd.DataFrame(data=tmp, columns=true_values.columns, dtype='float32')
    result_predicted_values['filename']=result_predicted_values['filename'].astype('str')
    index_temp=0
    for instance_idx in range(predicted_values.shape[0]):
        timesteps=np.unique(timesteps_labels[instance_idx])
        for timestep in timesteps:
            # assignment for filename and timestep
            result_predicted_values.iloc[index_temp,0]=class_dict[instance_idx]
            result_predicted_values.iloc[index_temp,1]=timestep
            # calculate mean of windows
            result_predicted_values.iloc[index_temp,2]=np.mean(predicted_values[instance_idx,timesteps_labels[instance_idx]==timestep])
            index_temp+=1
        #print('concatenation...instance:', instance_idx, '  done')

    return result_predicted_values

