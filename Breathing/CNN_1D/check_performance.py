import os
# test
from Compare2020.CNN_1D.utils import load_data, prepare_data, create_model, concatenate_prediction, pearson_coef, \
    correlation_coefficient_loss, load_test_data, prepare_test_data, concatenate_prediction_test
import numpy as np

# params
size_window=256000
step=102400
batch_size=20
path_to_weights='weights/model_weights.h5'

'''# validation data
path_to_validation_data='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Breathing\\wav\\'
path_to_validation_labels='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Breathing\\lab\\'
val_data, val_labels, val_dict, frame_rate=load_data(path_to_validation_data, path_to_validation_labels, 'devel')
prepared_val_data, prepared_val_labels,prepared_val_labels_timesteps=prepare_data(val_data, val_labels, val_dict, frame_rate, size_window, step)

# reshaping for prediction
prepared_val_data=prepared_val_data.reshape((prepared_val_data.shape+(1,)))
prepared_val_data=prepared_val_data.reshape(((-1,)+prepared_val_data.shape[2:]))
prepared_val_data=prepared_val_data.astype('float32')
prepared_val_labels=prepared_val_labels.reshape(((-1,)+prepared_val_labels.shape[2:]))'''

# test data
path_to_test_data='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Breathing\\wav\\'
path_to_test_labels='C:\\Users\\Dresvyanskiy\\Desktop\\ComParE2020_Breathing\\lab\\'
test_data, test_labels, test_dict, frame_rate=load_test_data(path_to_test_data, path_to_test_labels, 'test')
prepared_test_data, prepared_test_labels_timesteps=prepare_test_data(test_data, test_labels, test_dict, frame_rate, size_window, step)

# reshaping for test prediction
prepared_test_data=prepared_test_data.reshape((prepared_test_data.shape+(1,)))
prepared_test_data=prepared_test_data.reshape(((-1,)+prepared_test_data.shape[2:]))
prepared_test_data=prepared_test_data.astype('float32')


input_shape=(prepared_test_data.shape[-2],prepared_test_data.shape[-1])
output_shape=(prepared_test_data.shape[-1])
model=create_model(input_shape=input_shape, output_shape=output_shape)
model.load_weights(path_to_weights)
model.compile(optimizer='Adam', loss=correlation_coefficient_loss, metrics=['mse', 'mae'])

'''predicted_labels=model.predict(prepared_val_data, batch_size=batch_size)
concatenated_predicted_labels=concatenate_prediction(true_values=val_labels, predicted_values=predicted_labels,
                                                     timesteps_labels=prepared_val_labels_timesteps, class_dict=val_dict)
prc_coef=pearson_coef(val_labels.iloc[:,2].values,concatenated_predicted_labels.iloc[:,2].values)
print(prc_coef)'''


# prediction
predicted_labels=model.predict(prepared_test_data, batch_size=batch_size)
concatenated_test_labels=concatenate_prediction_test(true_values=test_labels, predicted_values=predicted_labels,
                                                     timesteps_labels=prepared_test_labels_timesteps, class_dict=test_dict)
a=1+2
path_to_save_result='weights/result.csv'
concatenated_test_labels.to_csv(path_to_save_result, index=False)
