The description will be updated on 22.10.2020

# Compare2020 Breathing SC

## Project organization

------------

    ├── LICENSE
    ├── README.md                                                <- README for developers using this project.
    │
    ├── Submissions                                              <- Python scripts to reproduse results of all 5 submissions. 
    |                                                                To reproduce it, run every script independently.
    |
    │
    ├── requirements.txt                                                <- The requirements file for reproducing the analysis environment
    ├── CNN_1D                                                           <- Source code for use 1D CNN + LSTM model in this project
    |   ├── __init__.py                                                  <- Makes src a Python module
    |   │
    |   ├── check_performance.py                                        <- Python script, which allows to check performance of model
    |   |                                                                 To run it, please, specify paths to data and weights of model 
    |   │
    |   ├── cross_validation_2N_check_test.py                           <- Python script, which allows to generate test predictions with the help 
    |   |                                                                 of models obtained via cross-validation technique described in the article
    |   │
    |   ├── cross_validation_2N_check_train_dev.py                      <- Python script, which allows to calculate ensemble performance (Pearson correlation) 
    |   |                                                                    as separately for each model in ensemble as performance of ensemble over all  
    |   |                                                                  data (train+development). Ensemble is generated via cross-validation
    |   |
    |   ├── cross_validation_2N_train.py                                 <- Python script, which allows to generate 2N models trained independently
    |   |                                                                  by cross-validation technique.
    |   |
    |   ├── train.py                                                     <- Python script allowed to train model on train data.
    |   |
    |   ├── utils.py                                                     <- Python script, which contains util functions.
    |   
    |   
    ├── Fusion_system                                                    <- Scripts for fusion system.
            ├── Feature_extraction_1D_CNN.py                             <- Python script for extracting deep embeddings by given models.
            |
            ├── feature_level_fusion_model_check_performance.py          <- Check performance of LSTM model trained on deep embeddings from 1D CNN + LSTM
            |                                                               and 2D CNN + GRU.
            |
            ├── feature_level_fusion_model_generate_test_predictions.py  <- Python script for generating test predictions by LSTM model trained on deep 
            |                                                               embeddings from 1D CNN + LSTM and 2D CNN + GRU.
            |                                                               
            ├── feature_level_fusion_model_training.py                   <- Python script for training fusion model based on LSTM. The LSTM model will
            |                                                               train on deep embeddings from 1D CNN + LSTM and 2D CNN + GRU.
            |
            ├── fusion_utils.py                                          <- Utils for fusion scripts.
 
--------


Note:
* All experiments were carried out in PyCharm and Jupiter Notebook.
* Random seed was not fixed.
* Firstly, install required packages. Execute: <br/>
```pip install -r requirements.txt``` <br/> preferably on a separate virtualenv.
* To run every script, you should configure the data paths in it.

# Reproduction of Results
You can find the weights for models via this [link](https://drive.google.com/drive/folders/11JZduaDgUttLHfH1b9tB2H4CycrH3qY8?usp=sharing).
To repeat result, please, run corresponding python scripts in Submissions folder.
