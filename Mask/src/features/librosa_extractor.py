import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
class LibrosaFeatureExtractor:
    def __init__(self, params, alignment=None):
        self.feature_type = params['type']
        self.sr = params['sample_rate']
        self.n_fft = int(self.sr * params['window_width'])
        self.hop_length = int(self.sr * params['stride'])
        self.center = params['center']

        self.alignment = alignment

        self.wave_size = None
        self.seq_size = None

        self.params = params[self.feature_type]

    def load_wave(self, full_path):
        wave, sr = librosa.load(full_path, self.sr)
        return wave

    def get_frame(self, sample):
        wave = self.load_wave(sample.wav_path)
        return librosa.util.frame(wave, self.n_fft, self.hop_length)

    def get_speech(self, sample, vad_mask):
        wave = self.load_wave(sample.wav_path)
        # Add 0 to start and to end of file
        vad_mask = np.pad(vad_mask, (1, 1), 'constant')
        speech_st_idx = np.where(vad_mask[:-1] < vad_mask[1:])[0] + 1
        speech_en_idx = np.where(vad_mask[:-1] > vad_mask[1:])[0] + 1

        speech_st_samples = librosa.core.frames_to_samples(speech_st_idx,
                                                           hop_length=self.hop_length,
                                                           n_fft=self.n_fft)
        speech_en_samples = librosa.core.frames_to_samples(speech_en_idx,
                                                           hop_length=self.hop_length,
                                                           n_fft=self.n_fft)
        return np.concatenate(
            [wave[speech_st_samples[idx]:speech_en_samples[idx]] for idx, i in enumerate(speech_en_samples)])

    def save_wave(self, wave, path):
        sf.write(path, wave, self.sr)

    def get_mel_features(self, wave):
        n_mels = int(self.params['n_mels'])
        s = librosa.feature.melspectrogram(y=wave,
                                           sr=self.sr,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           n_mels=n_mels, center=self.center)
        log_s = librosa.power_to_db(s, ref=np.max)
        return log_s

    def get_mfcc_features(self, wave):
        n_mfcc = int(self.params['n_mfcc'])
        deltas = self.params['order']

        mfcc = librosa.feature.mfcc(y=wave,
                                    sr=self.sr,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    n_mfcc=n_mfcc, center=self.center)

        result = []
        for delta in deltas:
            if delta == 0:
                result.append(mfcc)
            else:
                result.append(librosa.feature.delta(mfcc, order=int(delta)))

        return np.vstack(result)

    def set_wave_size(self, *samples_lists):
        all_samples = []
        for i in samples_lists:
            all_samples.extend(i)

        max_node = max(all_samples, key=lambda sample: sample.file_size)
        wave = self.load_wave(max_node.wav_path)
        if self.alignment == 'global':
            self.wave_size = len(wave)
            self.seq_size = self.process(max_node).shape

    def process(self, sample, vad_mask=None):
        wave = self.load_wave(sample.wav_path)
        if self.wave_size and self.alignment == 'global':
            wave = np.pad(wave, (0, self.wave_size - len(wave)), mode='constant')
        features = None
        if self.feature_type == 'mel':
            features = self.get_mel_features(wave=wave)
        elif self.feature_type == 'mfcc':
            features = self.get_mfcc_features(wave=wave)

        if vad_mask is not None:
            return features[:, vad_mask > 0]
        else:
            return features

    def features_to_img(self, sample, legend=False, title='MFCC', fig_size=(10, 4), ax=None, save=False, save_path=None, vad_mask=None):
        features = self.process(sample, vad_mask)
        if save_path is None:
            path = '{0}_{1}.png'.format(sample.wav_path.split('.wav')[0], self.feature_type)
        else:
            path = save_path

        fig = None
        if ax is None:
            fig = plt.figure(figsize=fig_size, frameon=False)

        librosa.display.specshow(features, sr=self.sr,
                                 ax=ax,
                                 y_axis='mel')

        if ax is None:
            if legend:
                plt.colorbar(format='%+2.0f dB')
                plt.title(title)
            else:
                plt.tight_layout()
                fig.subplots_adjust(bottom=0)
                fig.subplots_adjust(top=1)
                fig.subplots_adjust(right=1)
                fig.subplots_adjust(left=0)

            if save:
                fig.savefig(path)

            plt.close(fig)
            return fig
        else:
            if legend:
                ax.set_title(title)
            else:
                plt.tight_layout()

            if save:
                print('Save option is not available')
            
            return ax
