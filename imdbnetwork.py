from time import time

import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import metrics

def onehot_vectorize(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


class IMDBNetwork(object):
    def __init__(self, num_words=10000, epochs=20, batch_size=512):
        self.trained = False
        self.num_words = num_words
        self.word_index = imdb.get_word_index()
        self.epochs = epochs
        self.batch_size = batch_size

        (self.train_data, self.train_labels), (self.test_data, self.test_labels) =\
            imdb.load_data(num_words=self.num_words)

        self.x_train = onehot_vectorize(self.train_data, self.num_words)
        self.x_test  = onehot_vectorize(self.test_data, self.num_words)

        self.y_train = np.asarray(self.train_labels).astype('float32')
        self.y_test = np.asarray(self.test_labels).astype('float32')

        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(num_words,)),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def decode_imdb_review(self, review):
        reverse_word_index = dict(
            [(value, key) for (key, value) in self.word_index.items()]
        )

        ## Index is offset by three: 0, 1, and 2 are reserved indices,
        ## meaning "padding", "start of sequence", and "unknown", respectively.
        decoded_review = ' '.join(
            [reverse_word_index.get(i - 3, '?') for i in review]
        )

        return decoded_review


    def train(self, save=True, save_path="imdb_{}.h5"):
        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=[metrics.binary_accuracy])

        x_val = self.x_train[:10000]
        partial_x_train = self.x_train[10000:]
        y_val = self.y_train[:10000]
        partial_y_train = self.y_train[10000:]

        self.model.fit(partial_x_train,
                       partial_y_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_data=(x_val, y_val)
        )

        if save:
            if save_path == "imdb_{}.h5": save_path = save_path.format(time())
            self.model.save(save_path)
        self.trained = True


    def load_model(self, model_path):
        self.model = models.load_model(model_path)
        self.trained = True


    def evaluate(self):
        if not self.trained:
            raise ValueError("Train the network before evaluating it!")

        return self.model.evaluate(self.x_test, self.y_test)


def main():
    net = IMDBNetwork(num_words=20000)
    net.train()
    print(net.evaluate())


if __name__ == '__main__':
    main()
