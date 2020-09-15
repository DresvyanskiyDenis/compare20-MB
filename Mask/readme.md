Compare2020 Mask SC
==============================

Project Organization
------------

    ├── LICENSE
    ├── README.md                           <- README for developers using this project.
    │
    ├── notebooks                           <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                          the creator's initials, and a short `-` delimited description, e.g.
    │                                          `1.0-jqp-initial-data-exploration`.
    │
    ├── references                          <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── logs                                <- Serialized models and confusion matrices.
    │   └── tb                              <- Tensorboard files
    │
    ├── requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
    └── src                                 <- Source code for use in this project.
        ├── __init__.py                     <- Makes src a Python module
        │
        ├── applications                    <- Baseline classifiers (SVM, RF, GBM)
        │   ├── baseline_v1.py              <- Organizers baseline
        │   ├── baseline_v2.py              <- beseline_v1 + SVM, RF, GBM as well as new set of features (MFCC, MEL)
        │   └── fusion_v1.py                <- Fusion of baseline classifiers
        │
        ├── data                            <- Scripts related with objects
        │   ├── __init__.py                 <- Makes data a Python module
        │   └── data_sample.py              <- Object that stores all information of files
        │
        ├── features                        <- Scripts related with features
        │   ├── __init__.py                 <- Makes features a Python module
        │   └── librosa_extractor.py        <- Wrapper for librosa
        │
        └── utils                           <- Additional scripts
            ├── __init__.py                 <- Makes utils a Python module
            ├── accuracy_utils.py           <- Wrapper for sklearn metrics
            └── configuration_utils.py      <- Initialization of configuration

--------

Note:
* All experiments were carried out in Jupiter Notebook.
* Despite the fact that a random seed is fixed, different computers will give different results.
* Firstly, install required packages. Execute: <br/>
```pip install -r requirements.txt``` <br/> preferably on a separate virtualenv.
* Secondly, you should create folders logs/tb and extract features for reproduce results. In addition, you should configure the data paths.

# Reproduction of Results
## Feature extraction
Use notebook ```1.0-Maxim-Spectrogram.ipynb``` to extract features. Before doing this configure the data paths.

## Submission 1
Use notebook ```2.0-Maxim-Submit1.ipynb``` to reproduce the results of 1st submission. We used results of 68 epoch.

## Submission 2
Use notebook ```1.0-Maxim-Submit2.ipynb``` to reproduce the results of 2nd submission. You should train 2 models * 4 folds (in total 8 models) with such optimizers as Adam and SGD.

## Submission 3
Use notebooks ```1.0-Maxim-Submit3.1.ipynb``` and ```1.0-Maxim-Submit3.2.ipynb``` to reproduce the results of 3rd submission. Firstly, you should train SVM model (```1.0-Maxim-Submit3.1.ipynb```). Secondly, you should train Resnet model and fuse resnet predicts with SVM model (```1.0-Maxim-Submit3.2.ipynb```).

## Submission 4
Use notebook ```1.0-Maxim-Submit4.ipynb``` to reproduce the results of 4th submission. You should train 2 models * 10 folds (in total 20 models) with such optimizers as Adam and SGD.

## Submission 5
Use notebook ```1.0-Maxim-Fusion.ipynb``` to reproduce the results of 5th submission. We used a weighted fusion (WF) to all neural networks from 4th submission. In each fold, we weighted predictions of two networks (ResNet18v2 with Adam, and Resnet18v2 with SGD), where the fusion weight is optimized on the respective validation set. Then, we calculated mean prediction of the foldwise decisions.

# Citing
Please cite our paper if you use any code of our solution for ComParE2020 in your research work. \
```M. Markitantov, D. Dresvyanskiy, D. Mamontov, H. Kaya, W. Minker, and A. Karpov, “Ensembling end-to-end deep modelsfor computational paralinguistics tasks: ComParE 2020 Mask andBreathing Sub-challenges,” in INTERSPEECH, Shanghai, China, October 2020, to appear.```
