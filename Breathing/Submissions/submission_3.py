import os

from Breathing.CNN_1D.cross_validation_2N_train import main


if __name__ == "__main__":
    # please, specify paths to data and other parameters
    path_to_save_model='..\\logs\\best_models_CV\\'
    if not os.path.exists(path_to_save_model):
        os.mkdir(path_to_save_model)
    path_to_save_tmp_model='..\\logs\\tmp_models_CV\\'
    if not os.path.exists(path_to_save_tmp_model):
        os.mkdir(path_to_save_tmp_model)
    main(window_size=256000, data_parts=2,
         model_type='complex',
         path_to_save_models=path_to_save_model,
         path_to_save_tmp_models=path_to_save_tmp_model,
         path_to_train_data='',
         path_to_train_labels='',
         path_to_devel_data='',
         path_to_devel_labels='')