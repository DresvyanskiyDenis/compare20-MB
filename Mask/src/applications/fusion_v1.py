import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import PredefinedSplit

from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

sys.path.append('./src')

from utils.configuration_utils import create_logger


def read_data(feature_set, task_name, features_path, labels_path):
    # Mapping each available feature set to tuple
    # (number of features, offset/index of first feature, separator, header option)
    feat_conf = {
        'ComParE': (6373, 1, ';', 'infer'),
        'BoAW-125': (250, 1, ';', None),
        'BoAW-250': (500, 1, ';', None),
        'BoAW-500': (1000, 1, ';', None),
        'BoAW-1000': (2000, 1, ';', None),
        'BoAW-2000': (4000, 1, ';', None),
        'auDeep-30': (1024, 2, ',', 'infer'),
        'auDeep-45': (1024, 2, ',', 'infer'),
        'auDeep-60': (1024, 2, ',', 'infer'),
        'auDeep-75': (1024, 2, ',', 'infer'),
        'auDeep-fused': (4096, 2, ',', 'infer'),
        'DeepSpectrum_resnet50': (2048, 1, ',', 'infer'),

        
        'mel_64_r': (6208, 1, ',', 'infer'),
        'mfcc_30_0-2_r': (8730, 1, ',', 'infer'),
        'mfcc_30_0_r': (2910, 1, ',', 'infer'),
        'mfcc_30_1_r': (2910, 1, ',', 'infer'),
        'mfcc_30_2_r': (2910, 1, ',', 'infer'),
    }

    num_feat = feat_conf[feature_set][0]
    ind_off = feat_conf[feature_set][1]
    sep = feat_conf[feature_set][2]
    header = feat_conf[feature_set][3]

    # Load features and labels
    x_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    x_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    x_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep, header=header,
                         usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(labels_path)
    y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    train_fn = df_labels['file_name'][df_labels['file_name'].str.startswith('train')].values

    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
    devel_fn = df_labels['file_name'][df_labels['file_name'].str.startswith('devel')].values

    test_fn = df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values

    return x_train, y_train, train_fn, x_devel, y_devel, devel_fn, x_test, test_fn


def save_scv(file_names, features, file_name):
    cols = ['file_name']
    cols.extend(['feat_{0}'.format(i) for i in range(features.shape[1])])

    df = pd.DataFrame(np.hstack((file_names, features)), columns=cols)
    df.to_csv(file_name, index=False)


def train(features_to_cls, 
          features_path, labels_path, classes,
          task_name, team_name, submission_index,
          show_confusion=True, save_predicts=False):

    # Start
    logging.info('Running {0} ... (this might take a while)\n'.format(task_name))

    train_predicts = []
    test_predicts = []

    train_fn = None
    y_train = None
    test_fn = None
    for estimator in features_to_cls:
        logging.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logging.info('Running {0} estimator'.format(estimator['estimator'].__class__.__name__))
        estimator_st_time = time.time()

        for k, v in estimator['features_parameters'].items():
            logging.info('Running {0} feature set'.format(k))
            features_st_time = time.time()

            x_train, y_train, train_fn, x_devel, y_devel, devel_fn, x_test, test_fn = read_data(feature_set=k, task_name=task_name,
                                                                                                features_path=features_path,
                                                                                                labels_path=labels_path)
            
            # Feature normalisation
            scaler = MinMaxScaler()
            x_devel = scaler.fit_transform(x_devel)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)


            v['cls'] = clone(estimator['estimator'])
            v['cls'].set_params(**v['params'])

            v['cls'].fit(x_devel, y_devel)

            train_predicts.append(v['cls'].predict_proba(x_train))
            test_predicts.append(v['cls'].predict_proba(x_test))

            logging.info('--- Feature set {0} execution time {1} seconds ---'.format(k, time.time() - features_st_time))
            logging.info('---------------------------------------------------')

        logging.info('--- Estimator {0} execution time {1} seconds ---\n'.format(estimator['estimator'].__class__.__name__,
                                                                                 time.time() - estimator_st_time))
    x_new_train = np.hstack(train_predicts)
    save_scv(np.expand_dims(train_fn, axis=1), x_new_train, 'ComParE2020_Mask.devel_train_predicts_for_RF.devel.csv')

    x_new_test = np.hstack(test_predicts)
    save_scv(np.expand_dims(test_fn, axis=1), x_new_test, 'ComParE2020_Mask.devel_train_predicts_for_RF.test.csv')

    logging.info('---------------------------------------------------')
    logging.info('--------------- Running RF ---------------')
    # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # support for parallelized ensembles but is necessary for tracking the OOB
    # error trajectory during training.
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'", RandomForestClassifier(warm_start=True, oob_score=True, max_features="sqrt", random_state=0)),
        ("RandomForestClassifier, max_features='log2'", RandomForestClassifier(warm_start=True, max_features='log2', oob_score=True, random_state=0)),
        ("RandomForestClassifier, max_features=None", RandomForestClassifier(warm_start=True, max_features=None, oob_score=True, random_state=0))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(x_new_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def main():
    # Initialization
    create_logger('.',
                  'Features for RF.log',
                  console_level=logging.INFO,
                  file_level=logging.NOTSET)

    # Task
    task_name = 'ComParE2020_Mask'
    classes = ['clear', 'mask']

    # Enter your team name HERE
    team_name = 'baseline'

    # Enter your submission number HERE
    submission_index = 1

    # For all available options, see the dictionary feat_conf
    features_to_cls = [
        # {
        #     'estimator': LinearSVC(),
        #     'features_parameters': {
        #         'ComParE': {
        #             'params': {'random_state': 0, 'C': 0.01},
        #             'cls': None,
        #         },
        #         'BoAW-2000': {
        #             'params': {'random_state': 0, 'C': 0.001},
        #             'cls': None,
        #         },
        #         'auDeep-fused': {
        #             'params': {'random_state': 0, 'C': 0.1},
        #             'cls': None,
        #         },
        #         'DeepSpectrum_resnet50': {
        #             'params': {'random_state': 0, 'C': 0.01},
        #             'cls': None,
        #         },    
        #     }
        # },
        {
            'estimator': SVC(),
            'features_parameters': {
                'ComParE': {
                    'params': {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 0, 'probability': True},
                    'cls': None,
                },
                'BoAW-2000': {
                    'params': {'random_state': 0, 'probability': True},
                    'cls': None,
                },
                'auDeep-fused': {
                    'params': {'C': 0.01, 'kernel': 'linear', 'random_state': 0, 'probability': True},
                    'cls': None,
                },
                'DeepSpectrum_resnet50': {
                    'params': {'C': 10, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 0, 'probability': True},
                    'cls': None,
                },
            }
        },
        {
            'estimator': RandomForestClassifier(),
            'features_parameters': {
                'ComParE': {
                    'params': {'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000, 'n_jobs': 2, 'random_state': 0},
                    'cls': None,
                },
                'BoAW-2000': {
                    'params': {'criterion': 'entropy', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000, 'n_jobs': 2, 'random_state': 0},
                    'cls': None,
                },
                'auDeep-75': {
                    'params': {'criterion': 'entropy', 'max_depth': 12, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1000, 'n_jobs': 2, 'random_state': 0},
                    'cls': None,
                },
                'DeepSpectrum_resnet50': {
                    'params': {'criterion': 'entropy', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 1000, 'n_jobs': 2, 'random_state': 0},
                    'cls': None,
                },
            }
        },
        {
            'estimator': XGBClassifier(),
            'features_parameters': {
                'ComParE': {
                    'params': {'learning_rate': 0.2, 'n_estimators': 2000, 'nthread': 2, 'seed': 0},
                    'cls': None,
                },
                'BoAW-1000': {
                    'params': {'learning_rate': 0.3, 'n_estimators': 1500, 'nthread': 2, 'seed': 0},
                    'cls': None,
                },
                'auDeep-75': {
                    'params': {'learning_rate': 0.15, 'n_estimators': 100, 'nthread': 2, 'seed': 0},
                    'cls': None,
                },
                'DeepSpectrum_resnet50': {
                    'params': {'learning_rate': 0.05, 'n_estimators': 2000, 'nthread': 2, 'seed': 0},
                    'cls': None,
                },
            }
        },
    ]

    # Path of the features and labels
    features_path = '/media/maxim/SStorage/ComParE2020_Mask/features/'
    labels_path = '/media/maxim/SStorage/ComParE2020_Mask/lab/labels.csv'

    train(features_to_cls, features_path, labels_path, classes,
          task_name, team_name, submission_index,
          show_confusion=True, save_predicts=False)


if __name__ == "__main__":
    main()
