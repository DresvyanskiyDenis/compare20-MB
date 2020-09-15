import sys
import time
import logging
import numpy as np
import pandas as pd

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
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
    test_fn = df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values

    return x_train, y_train, x_devel, y_devel, x_test, test_fn


def train(estimators, feature_sets,
          features_path, labels_path, classes,
          task_name, team_name, submission_index,
          show_confusion=True, save_predicts=False):
    # Start
    logging.info('Running {0} ... (this might take a while)\n'.format(task_name))

    for feature_set in feature_sets:
        logging.info('Running {0} feature set'.format(feature_set))
        features_st_time = time.time()

        # Reading features
        x_train, y_train, x_devel, y_devel, x_test, test_fn = read_data(feature_set=feature_set,
                                                                        task_name=task_name,
                                                                        features_path=features_path,
                                                                        labels_path=labels_path)

        x_traindevel = None
        y_traindevel = None
        if save_predicts:
            # Concatenate training and development for final training
            x_traindevel = np.concatenate((x_train, x_devel))
            y_traindevel = np.concatenate((y_train, y_devel))

        # Feature normalisation
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_devel = scaler.transform(x_devel)

        for estimator in estimators:
            logging.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logging.info('Running {0} estimator'.format(estimator['estimator'].__class__.__name__))
            estimator_st_time = time.time()

            # when using a validation set, set the test_fold to 0 for all samples
            # that are part of the validation set, and to -1 for all other samples.
            test_fold = np.hstack((np.full(len(x_train), -1), np.full(len(x_devel), 0)))
            x = np.concatenate((x_train, x_devel))
            y = np.concatenate((y_train, y_devel))

            clf = GridSearchCV(estimator=estimator['estimator'],
                               param_grid=estimator['parameters'],
                               n_jobs=11,
                               cv=PredefinedSplit(test_fold=test_fold),
                               scoring='recall_macro',
                               verbose=10)

            clf.fit(x, y)

            logging.info('Estimator results')
            for idx, value in enumerate(clf.cv_results_['params']):
                logging.info(
                    'Parameters: {0}. UAR on Devel: {1}'.format(value,
                                                                clf.cv_results_['split0_test_score'][idx] * 100))

            logging.info('Best parameters are: {0}'.format(clf.best_params_))
            logging.info('Maximum UAR on Devel {0:.3f}'.format(clf.best_score_ * 100))

            logging.info(
                '--- Estimator {0} execution time {1} seconds ---\n'.format(estimator['estimator'].__class__.__name__,
                                                                            time.time() - estimator_st_time))

            if show_confusion:
                logging.info('Confusion matrix of Best Estimator (Devel):')
                best_estimator = clone(clf.best_estimator_)
                best_estimator.fit(x_train, y_train)
                logging.info('\n{0}\n{1}'.format(classes, confusion_matrix(y_devel,
                                                                           best_estimator.predict(x_devel),
                                                                           labels=classes)))

            if save_predicts:
                best_estimator = clone(clf.best_estimator_)

                x_traindevel = scaler.fit_transform(x_traindevel)
                x_test = scaler.transform(x_test)

                best_estimator.fit(x_traindevel, y_traindevel)
                y_pred = best_estimator.predict(x_test)

                # Write out predictions to csv file (official submission format)
                pred_file_name = '{0}__{1}.{2}.test.{3}_{4}.csv'.format(
                    estimator['estimator'].__class__.__name__,
                    task_name,
                    feature_set,
                    team_name,
                    submission_index,
                )

                logging.info('Writing file {0}'.format(pred_file_name))
                df = pd.DataFrame(
                    data={'file_name': test_fn,
                          'prediction': y_pred.flatten()},
                    columns=['file_name', 'prediction'])
                df.to_csv(pred_file_name, index=False)

        logging.info('--- Feature set {0} execution time {1} seconds ---'.format(feature_set,
                                                                                 time.time() - features_st_time))
        logging.info('---------------------------------------------------')


def define_estimators():
    return [
        {
            'estimator': LinearSVC(),
            'parameters': [
                {'random_state': [0]},
                {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1, 10], 'random_state': [0], 'max_iter': [100000]}
            ]
        },
        {
            'estimator': SVC(),
            'parameters': [
                {'random_state': [0]},
                {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'random_state': [0]},
                {
                    'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1, 1e-0, 1, 10, 100],
                    'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'random_state': [0]
                },
            ]
        },
        {
            'estimator': RandomForestClassifier(),
            'parameters': [
                {'random_state': [0], 'n_jobs': [3]},
                {
                    'n_estimators': [50, 200, 500, 1000], 'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth': [None, 3, 7, 12, 15], 'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_leaf': [1, 2, 4], 'random_state': [0], 'n_jobs': [2]
                }
            ],
        },
        {
            'estimator': XGBClassifier(),
            'parameters': [
                {'seed': [0], 'nthread': [2]},
                {
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'n_estimators': [25, 50, 100, 200, 500, 750, 1000, 1500, 2000],
                    'seed': [0], 'nthread': [2],
                }
            ],
        },
        # {
        #     'estimator': XGBClassifier(),
        #     'parameters': [
        #         {'seed': [0], 'nthread': [3]},
        #         {
        #             'learning_rate': [0.01, 0.1, 0.2, 0.3], 'gamma': [0.01, 0.1, 0.5, 1, 2],
        #             'max_depth': [3, 7, 12, 15], 'min_child_weight': [1, 3, 5, 10],
        #             'subsample': [0.3, 0.5, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
        #             'lambda': [0.01, 0.1, 0, 1, 10], 'alpha': [0.01, 0.1, 0, 1, 10],
        #             'objective': ['binary:logistic'], 'seed': [0], 'nthread': [2],
        #         }
        #     ],
        # },
        # {
        #     'estimator': KNeighborsClassifier(),
        #     'parameters': [
        #         {'random_state': [0]},
        #     ],
        # },
        # {
        #     'estimator': AdaBoostClassifier(),
        #     'parameters': [],
        # },
        # {
        #     'estimator': GradientBoostingClassifier(),
        #     'parameters': [],
        # },
        # {
        #     'estimator': BaggingClassifier(),
        #     'parameters': [],
        # }
    ]


def main():
    # Initialization
    create_logger('.',
                  'NewFeatures.log',
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
    # feature_sets = ['ComParE', 'BoAW-125', 'BoAW-250', 'BoAW-500', 'BoAW-1000', 'BoAW-2000', 'auDeep-30',
                    # 'auDeep-45', 'auDeep-60', 'auDeep-75', 'auDeep-fused', 'DeepSpectrum_resnet50']
    
    feature_sets = ['mel_64_r', 'mfcc_30_0-2_r', 'mfcc_30_0_r', 'mfcc_30_1_r', 'mfcc_30_2_r']

    # Path of the features and labels
    features_path = '/media/maxim/SStorage/ComParE2020_Mask/features/'
    labels_path = '/media/maxim/SStorage/ComParE2020_Mask/lab/labels.csv'

    train(define_estimators(), feature_sets,
          features_path, labels_path, classes,
          task_name, team_name, submission_index,
          show_confusion=True, save_predicts=False)


if __name__ == "__main__":
    main()
