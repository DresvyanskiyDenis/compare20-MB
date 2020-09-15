import os
import numpy as np


class DataSample:
    def __init__(self, wav_path=None, file_name=None, file_size=None, label=None, features=None):
        self.wav_path: str = wav_path
        self.file_name: str = file_name
        self.file_size: int = file_size
        self.label: int = label
        self.features: np.ndarray = features

    def display(self):
        print(self.__str__())

    def to_dict(self, reshape=False):
        res = {}
        res['file_name'] = self.file_name
        if reshape:
            self.features = np.reshape(self.features, -1)
            for i, features in enumerate(self.features):
                res['feat_{0}'.format(i)] = features
        else:
            res['features'] = self.features.tolist()

        return res

    def __repr__(self):
        res = ['<DataSample wav_path:{0}'.format(self.wav_path),
               'file_name:{0}'.format(self.file_name),
               'file_size:{0}'.format(self.file_size),
               'label:{0}'.format(self.label),
               'features:{0}>'.format(self.features)]
        return ' '.join(res)

    def __str__(self):
        res = ['File Info: \nPath: {0}, Name: {1}, Size: {2}'.format(self.wav_path, self.file_name, self.file_size),
               'Classification Info: \nLabel: {0}'.format(self.label), 'Features: \n{0}'.format(self.features)]
        return '\n'.join(res)
