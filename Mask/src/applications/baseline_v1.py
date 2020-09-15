import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix


def main():
    # Task
    task_name = 'ComParE2020_Mask'  # os.getcwd().split('/')[-2]
    classes = ['clear', 'mask']

    # Enter your team name HERE
    team_name = 'baseline'

    # Enter your submission number HERE
    submission_index = 1

    # Option
    show_confusion = True  # Display confusion matrix on devel

    # Configuration
    feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]  # SVM complexities (linear kernel)

    # Mapping each available feature set to tuple
    # (number of features, offset/index of first feature, separator, header option)
    feat_conf = {'ComParE': (6373, 1, ';', 'infer'),
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
                 'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')}
    num_feat = feat_conf[feature_set][0]
    ind_off = feat_conf[feature_set][1]
    sep = feat_conf[feature_set][2]
    header = feat_conf[feature_set][3]

    # Path of the features and labels
    features_path = '/media/maxim/SStorage/ComParE2020_Mask/features/'
    label_file = '/media/maxim/SStorage/ComParE2020_Mask/lab/labels.csv'

    # Start
    print('\nRunning ' + task_name + ' ' + feature_set + ' baseline ... (this might take a while) \n')

    # Load features and labels
    x_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    x_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    x_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep, header=header,
                         usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(label_file)
    y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

    # Concatenate training and development for final training
    x_traindevel = np.concatenate((x_train, x_devel))
    y_traindevel = np.concatenate((y_train, y_devel))

    # Feature normalisation
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_devel = scaler.transform(x_devel)
    x_traindevel = scaler.fit_transform(x_traindevel)
    x_test = scaler.transform(x_test)

    # Train SVM model with different complexities and evaluate
    uar_scores = []
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=100000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_devel)
        uar_scores.append(recall_score(y_devel, y_pred, labels=classes, average='macro'))
        print('UAR on Devel {0}'.format(uar_scores[-1] * 100))
        if show_confusion:
            print('Confusion matrix (Devel):')
            print(classes)
            print(confusion_matrix(y_devel, y_pred, labels=classes))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(uar_scores)]
    print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(optimum_complexity,
                                                                                 np.max(uar_scores) * 100))

    clf = svm.LinearSVC(C=optimum_complexity, random_state=0, max_iter=100000)
    clf.fit(x_traindevel, y_traindevel)
    y_pred = clf.predict(x_test)

    # Write out predictions to csv file (official submission format)
    pred_file_name = task_name + '.' + feature_set + '.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': y_pred.flatten()},
                      columns=['file_name', 'prediction'])
    df.to_csv(pred_file_name, index=False)

    print('Done.\n')


if __name__ == "__main__":
    main()
