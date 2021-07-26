import os
import json
import pathlib

import numpy as np
import keras


class StressModel:
    def __init__(self, model_dir):
        if model_dir is None:
            model_dir = str(pathlib.Path(__file__).resolve().parent)

        with open(os.path.join(model_dir, 'nn_stress.cfg'), 'r') as f:
            cfg = json.load(f)
            self.max_len = cfg['max_len']
            self.char2index = cfg['char2index']

        self.model = keras.models.load_model(os.path.join(model_dir, 'nn_stress.model'))
        self.X = np.zeros((1, self.max_len), dtype=np.int)

    def predict(self, word):
        self.X.fill(0)
        for ich, c in enumerate(word.lower()[:self.max_len]):
            if c in self.char2index:
                self.X[0, ich] = self.char2index[c]
        y = self.model.predict({'input': self.X}, verbose=0)
        stress_pos = np.argmax(y, axis=-1)[0]
        return stress_pos


if __name__ == '__main__':
    model = StressModel('../../../tmp/stress_model')
    for word in 'чаков кошка мультипликация обсервация'.split():
        i = model.predict(word)
        stress = word[:i] + '^' + word[i:]
        print('{} => {}'.format(word, stress))
