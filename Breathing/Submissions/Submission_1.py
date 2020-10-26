import os

from Breathing.CNN_1D.train import main

if __name__ == "__main__":
    window_size=256000
    path_to_save_model='..\\logs\\best_model\\'
    if not os.path.exists(path_to_save_model):
        os.mkdir(path_to_save_model)
    main(window_size=window_size, path_to_save_model=path_to_save_model)