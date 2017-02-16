"""
Contains class for usage of Neural Network from keras library
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.utils.np_utils import to_categorical

from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm

def create_keras_model(features_count):
    model = Sequential()
    model.add(Dense(100, input_dim=features_count, init='uniform', activation='relu'))
    model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dense(KerasNeuralNetworkAlgorithm.categories_count, init='uniform', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class KerasNeuralNetworkAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Neural Network method
    """
    seed = 12345
    categories_count = 1 # to be edited in config file

    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        np.random.seed(KerasNeuralNetworkAlgorithm.seed)
        self.sentence_embedding = sentence_embedding

        # suppress this param, as neural network doesn't use such parameter
        if 'probability' in kwargs:
            del kwargs['probability']
        if "n_jobs" in kwargs:
            del kwargs["n_jobs"]  # multi-threading not available here

    def fit(self, features, labels):
        self.model = create_keras_model(features.shape[1])
        self.model.fit(features, to_categorical(labels), nb_epoch=150, batch_size=10, verbose=0)

    def predict(self, sentence):
        return self.model.predict_classes(np.array([self.sentence_embedding[sentence]]))[0]

    def predict_proba(self, sentence):
        probabilities = self.model.predict_proba(np.array([self.sentence_embedding[sentence]]), verbose=0)[0]
        prob_sum = sum(probabilities)
        return [prob/prob_sum for prob in probabilities]

    def visualize_2d(self, xs, ys, ax, color_map):
        z = self.model.predict_classes(np.c_[xs.ravel(), ys.ravel()])
        z = z.reshape(xs.shape)
        ax.contourf(xs, ys, z, cmap=color_map)
