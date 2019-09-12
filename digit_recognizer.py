import os
import keras
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import PReLU, LeakyReLU


# Training log
# Dataset | Model | Epochs | Training Acc | Test Acc | Validation Acc |
#    1    |  A1   |   20   |    25.73%    |  27.59%  |      N/A       |
#    1    |  A2   |   20   |    55.17%    |  53.76%  |      N/A       |
#    1    |  A3   |   20   |    52.19%    |  55.36%  |      N/A       |
#    1    |  A4   |   20   |    69.45%    |  74.52%  |      N/A       |
#    1    |  A5   |   20   |    62.63%    |  62.24%  |      N/A       |
#    1    |  A6   |   20   |    68.50%    |  71.56%  |      N/A       |
#    1    |  A7   |   20   |    76.30%    |  80.44%  |      N/A       |
#    1    |  A8   |   20   |    68.35%    |  71.98%  |      N/A       |
#    1    |  A9   |   20   |    59.01%    |  64.47%  |      N/A       |
#    1    |  A10  |   20   |    61.85%    |  63.82%  |      N/A       |
#    2    |  A7   |   20   |    98.41%    |  97.32%  |     97.50%     |
#    3    |  A7   |   20   |    98.29%    |  98.33%  |     97.76%     |
#    4    |  A7   |   20   |    98.45%    |  97.45%  |      N/A       |
#    4    |  A7   |   20   |    98.45%    |  97.45%  |      N/A       |
#    4    |  B1   |   20   |    97.24%    |  97.26%  |      N/A       |
#    4    |  B2   |   20   |    97.82%    |  96.91%  |      N/A       |
#    4    |  B3   |   20   |    97.86%    |  97.14%  |      N/A       |
#    4    |  C1   |   20   |    98.43%    |  97.48%  |      N/A       |
#    4    |  C2   |   20   |    98.34%    |  97.64%  |      N/A       |


# Kaggle MNIST training and test (submission) sets
# Uses 2/3rds of the training set for training and the rest to test
def dataset1(ratio=2.0/3.0):
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train = train.sample(frac=1, random_state=1).reset_index(drop=True)
    x_dataset = (train.iloc[:,1:].values).astype('float32')
    y_dataset = train.iloc[:,0].values.astype('int32')
    train_size = int(ratio*x_dataset.shape[0])
    test_size = x_dataset.shape[0] - train_size
    x_train = x_dataset[:train_size]
    y_train = y_dataset[:train_size]
    x_test = x_dataset[train_size:]
    y_test = y_dataset[train_size:]
    x_submission = test.values.astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print("Training shape: ", x_train.shape, " Test shape: ", x_test.shape)
    return (x_train, y_train), (x_test, y_test), x_submission


def dataset2(ratio=2.0/3.0):
    (x_train, y_train), (x_test, y_test), x_submission = dataset1()
    return (x_train / 255, y_train), (x_test / 255, y_test), x_submission / 255


def dataset3(ratio=0.99):
    (x_train, y_train), (x_test, y_test), x_submission = dataset2(ratio)
    return (x_train, y_train), (x_test, y_test), x_submission


def dataset4(ratio=2.0/3.0):
    # Shuffle the columns to destroy any spatial relations
    train = pd.read_csv("train.csv")
    random.seed(1)
    cols = train.columns[1:].to_list()
    random.shuffle(cols)
    train_cols = ['label']
    train_cols.extend(cols)
    train = train[train_cols]
    test = pd.read_csv("test.csv")[cols]
    train = train.sample(frac=1, random_state=1).reset_index(drop=True)
    x_dataset = (train.iloc[:,1:].values).astype('float32')
    y_dataset = train.iloc[:,0].values.astype('int32')
    train_size = int(ratio*x_dataset.shape[0])
    test_size = x_dataset.shape[0] - train_size
    x_train = x_dataset[:train_size]
    y_train = y_dataset[:train_size]
    x_test = x_dataset[train_size:]
    y_test = y_dataset[train_size:]
    x_submission = test.values.astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print("Training shape: ", x_train.shape, " Test shape: ", x_test.shape)
    return (x_train / 255, y_train), (x_test / 255, y_test), x_submission / 255


def modelA(dropout):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    return model


def modelA1():
    return modelA(0.2)


def modelA2():
    return modelA(0.4)


def modelA3():
    return modelA(0.6)


def modelA4():
    return modelA(0.5)


def modelA5():
    return modelA(0.45)


def modelA6():
    return modelA(0.55)


def modelA7():
    return modelA(0.525)


def modelA8():
    return modelA(0.5375)


def modelA9():
    return modelA(0.5125)


def modelA10():
    return modelA(0.52)


def modelB(alpha=0.1, dropout=0.525):
    model = Sequential()
    model.add(Dense(512, activation=LeakyReLU(alpha), input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation=LeakyReLU(alpha)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    return model


def modelB1():
    return modelB(0.1)


def modelB2():
    return modelB(0.2)


def modelB3():
    return modelB(0.3)


def modelC1(dropout=0.525):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
    return model




batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test), x_submission = dataset3()


if not os.path.exists('mnist.h5'):
    model = modelC1()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('mnist.h5')

model = load_model('mnist.h5')

# predict results
results = model.predict(x_submission, verbose=1)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("results.csv",index=False)