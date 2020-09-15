import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix


class Accuracy:
    def __init__(self, classes=None, fn=None, torch_wrapper=False):
        if fn is None:
            fn = ['precision', 'recall', 'f1']

        if classes is None:
            classes = []

        self.classes = classes
        self.fn = fn
        self.torch_wrapper = torch_wrapper

    @staticmethod
    def convert_to_numpy(targets, predicts):
        return np.asarray(targets), np.argmax(np.asarray(predicts), axis=1)

    @staticmethod
    def calculate_precision(targets, predicts, **kwargs):
        return precision_score(targets, predicts, **kwargs)

    @staticmethod
    def calculate_recall(targets, predicts, **kwargs):
        return recall_score(targets, predicts, **kwargs)

    @staticmethod
    def calculate_f1(targets, predicts, **kwargs):
        return f1_score(targets, predicts, **kwargs)

    def calculate_metrics(self, targets, predicts):
        if self.torch_wrapper:
            targets, predicts = Accuracy.convert_to_numpy(targets, predicts)

        res_str = ""
        res = {}
        if 'precision' in self.fn:
            res['precision'] = Accuracy.calculate_precision(targets, predicts, average='macro')
            str_precision = " — precision: {0}".format(res['precision'])
            res_str += str_precision

        if 'recall' in self.fn:
            res['recall'] = Accuracy.calculate_recall(targets, predicts, average='macro')
            str_rec = " — recall: {0}".format(res['recall'])
            res_str += str_rec

        if 'f1' in self.fn:
            res['f1'] = Accuracy.calculate_f1(targets, predicts, average='macro')
            str_f1 = " — f1: {0}".format(res['f1'])
            res_str += str_f1

        res['res_str'] = res_str
        return res

    def metrics_report(self, targets, predicts):
        if self.torch_wrapper:
            targets, predicts = Accuracy.convert_to_numpy(targets, predicts)

        return classification_report(targets, predicts, target_names=self.classes)

    def calculate_confusion_matrix(self, targets, predicts):
        return confusion_matrix(targets, predicts, self.classes)

    def confusion_matrix(self, targets, predicts, **kwargs):
        if self.torch_wrapper:
            targets, predicts = Accuracy.convert_to_numpy(targets, predicts)

        cm = self.calculate_confusion_matrix(targets=targets, predicts=predicts)
        self.plot_confusion_matrix(cm=cm, **kwargs)

    def plot_confusion_matrix(self, cm, normalize=False, title='Confusion Matrix', color_map=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix,
        Normalization can be applied by setting `normalize=True`
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

            # Compute confusion matrix
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        np.set_printoptions(precision=3)
        print(cm)
        np.set_printoptions(precision=6)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=color_map)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.classes, yticklabels=self.classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        ax.set_xticks(np.arange(cm.shape[1] + 1)-.5)
        ax.set_yticks(np.arange(cm.shape[0] + 1)-.5)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.show(block=False)
