# Annoying warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import keras
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
from common import save_obj, load_obj
from random import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import Sequence
from keras.activations import relu, sigmoid
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback
from keras import metrics


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test


def dataset1():
    train, test = load_data()
    return datasetA(train, test)


def dataset2():
    train, test = load_data()
    return datasetA(train, test, 0.99)


# Kaggle MNIST training and test (submission) sets
# Uses 2/3rds of the training set for training and the rest to test
def datasetA(train, test, TRAIN_FRACTION=2.0/3.0):
    train = train.sample(frac=1, random_state=1).reset_index(drop=True)
    x_dataset = (train.iloc[:,1:].values).astype('float32')
    y_dataset = train.iloc[:,0].values.astype('int32')
    train_size = int(TRAIN_FRACTION*x_dataset.shape[0])
    test_size = x_dataset.shape[0] - train_size
    x_train = x_dataset[:train_size]
    y_train = y_dataset[:train_size]
    x_test = x_dataset[train_size:]
    y_test = y_dataset[train_size:]
    x_submission = test.values.astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    auto_train = np.concatenate((np.array(x_submission), np.array(x_dataset)))
    return (x_train / 255.0, y_train), (x_test / 255.0, y_test), x_submission / 255.0, auto_train / 255.0


def modelA(latent_size=14, dropout=0.5, leak=0.1, layers=[512, 128]):
    x = inputs1 = Input(shape=(784,), name="Input")
    for n in layers:
        x = Dense(n)(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(leak)(x)
    x = Dense(latent_size, name='Latent')(x)
    for n in reversed(layers):
        x = Dense(n)(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(leak)(x)
    # Output is positive
    x = Dense(784, activation="relu", name="Output")(x)
    model = Model(inputs=inputs1, outputs=x)

    # TODO: Redirect to logging
    # TODO: Add image
    # model.summary()
    model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mean_absolute_percentage_error'])

    return model


def modelA1():
    return modelA()


def train(model, indices, model_name, fullset):
    epochs = 50
    batch_size = 128
    dataset = fullset[indices]
    file_name = model_name + '.h5'
    if not os.path.exists(file_name):
        if callable(model): model = model()
        if isinstance(model, str): model = load_model(model)
        history = model.fit(dataset, dataset,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[earlystop],
                            validation_data=(dataset, dataset))
        model.save(file_name)

    model = load_model(file_name)
    results = np.sum(np.absolute(model.predict(dataset, verbose=0) - dataset), axis=1) / 784 * 255
    print(model_name, " mean error per pixel (data set): ", np.mean(results))
    results = np.sum(np.absolute(model.predict(fullset, verbose=0) - fullset), axis=1) / 784 * 255
    print(model_name, " mean error per pixel (full set): ", np.mean(results))
    return indices, results


_, __, ___, auto_train = dataset1()
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
indices, results = train(modelA1, np.arange(auto_train.shape[0]), 'modelA1', auto_train)
midway = np.sort(results)[int(results.shape[0] / 2)]
train(modelA1, np.argwhere(results < midway).flatten(), 'modelA1_low', auto_train)
train(modelA1, np.argwhere(results >= midway).flatten(), 'modelA1_high', auto_train)

#plt.hist(results, bins=140)
#plt.show()
