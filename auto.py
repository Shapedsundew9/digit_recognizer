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
import keras.backend as K
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
from functools import partial


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


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
              metrics=['mean_squared_error'])

    return model


def modelA1():
    return modelA()


def abs_pixel_err(delta):
    return np.sum(np.absolute(delta), axis=1) / 784 * 255


def mean_abs_pixel_err(delta):
    return np.mean(abs_pixel_err(delta))


def abs_pixel_err_sets(y_true, y_pred, mask):
    delta = abs_pixel_err(y_pred - y_true)
    return delta, delta[mask], delta[np.logical_not(mask)] 


def cluster_stats(r):
    lhs = max((np.min(r['a']), np.min(r['b'])))
    rhs = min((np.max(r['a']), np.max(r['b'])))
    a_samples_in_overlap = np.sum(r['a'][np.logical_and(r['a'] >= lhs, r['a'] <= rhs)])
    b_samples_in_overlap = np.sum(r['b'][np.logical_and(r['b'] >= lhs, r['b'] <= rhs)])
    frac = (a_samples_in_overlap + b_samples_in_overlap) / r['full'].shape[0]
    frac_a = a_samples_in_overlap / r['a'].shape[0]
    frac_b = b_samples_in_overlap / r['b'].shape[0]

    stats = {"a_min": np.min(r['a']), "a_max": np.max(r['a']), "a_mean": np.mean(r['a'])}
    stats.update({"b_min": np.min(r['b']), "b_max": np.max(r['b']), "b_mean": np.mean(r['b'])})
    stats.update({"full_min": np.min(r['full']), "full_max": np.max(r['full']), "full_mean": np.mean(r['full'])})
    stats.update({"overlap": r['overlap'], "lhs": lhs, "rhs": rhs, "frac": frac, "frac_a": frac_a, "frac_b": frac_b})
    stats.update(r)
    return stats


# The overlap in mean absolute pixel error between the high distribution
# and the low distribution. A -ve value is a separation.
def overlap_distance(y_true, y_pred, mask):
    full, a, b = abs_pixel_err_sets(y_true, y_pred, mask)
    lhs = max((np.min(a), np.min(b)))
    rhs = min((np.max(a), np.max(b)))
    overlap_count = np.sum(np.logical_and(a > lhs, a < rhs))
    return rhs - lhs, int(overlap_count), full, a, b


def analyse(y_true, y_pred, mask, name="model"):
    STEPS = 100
    overlap, overlap_cnt, full, a, b = overlap_distance(y_true, y_pred, mask)
    f_min, f_max = np.min(full), np.max(full)

    # Distribution chart
    # Data
    step = (f_max - f_min) / STEPS
    bins = np.arange(f_min - step, f_max + step, step)
    a_hist = np.histogram(a, bins)
    b_hist = np.histogram(b, bins)

    # Plot
    fig, ax = plt.subplots(figsize=(16,8))
    pa = ax.bar(a_hist[1][:-1], a_hist[0], width=0.4, color='r')
    pb = ax.bar(b_hist[1][:-1], b_hist[0], width=0.4, bottom=a_hist[0], color='b')

    # Analysis
    out_str1 = "Mean absolute pixel error\nFull={0:.2f}\na={1:.2f}\nb={2:.2f}"
    out_str1_plt = out_str1.format(np.mean(full), np.mean(a), np.mean(b))
    out_str1 = out_str1.replace("{", color.BOLD + "{").replace("}", "}" + color.END).replace("\n", ", ")
    out_str1_prt = out_str1.format(np.mean(full), np.mean(a), np.mean(b))
    out_str2 = "a(min, max)=({0:.2f},{1:.2f})\nb(min, max)=({2:.2f},{3:.2f})\noverlap={4:.2f}"
    out_str2_plt = out_str2.format(np.min(a), np.max(a), np.min(b), np.max(b), overlap)
    out_str2 = out_str2.replace("{", color.BOLD + "{").replace("}", "}" + color.END).replace("\n", ", ")
    out_str2_prt = out_str2.format(np.min(a), np.max(a), np.min(b), np.max(b), overlap)
    out_str3 = "a_cnt={0:d}\noverlap_cnt={1:d}"
    out_str3_plt = out_str3.format(a.shape[0], overlap_cnt)
    out_str3 = out_str3.replace("{", color.BOLD + "{").replace("}", "}" + color.END).replace("\n", ", ")
    out_str3_prt = out_str3.format(a.shape[0], overlap_cnt)

    out_str_plt = out_str1_plt + "\n" + out_str2_plt + "\n" + out_str3_plt
    out_str_prt = name + " " + out_str1_prt + ", " + out_str2_prt + ", " + out_str3_prt

    # Annotation
    bbox_props = dict(boxstyle="Round,pad=0.3", fc="w", lw=1)
    ax.text(0.8, 0.5, out_str_plt, ha="left", va="center", size=12, bbox=bbox_props, transform=ax.transAxes)
    ax.set_ylabel('Count')
    ax.set_xlabel('Mean Absolute Pixel Error')
    ax.set_title(name + ' Distribution of Sets')
    ax.legend((pa[0], pb[0]), ('a', 'b'))
    fig.savefig(name + ".png")
    plt.clf()

    # Text Output
    print(out_str_prt)

    analysis = {"full": full, "a": a, "b": b, "overlap": overlap, "mask": mask, "name": name}
    analysis.update({"a_cnt": a.shape[0], "b_cnt": b.shape[0]})
    analysis.update({"overlap_cnt": overlap_cnt, "full_cnt": full.shape[0]})
    return analysis

    
def train(model, mask, model_name, fullset):
    epochs = 50
    batch_size = 128
    dataset = fullset[mask]
    file_name = model_name + '.h5'
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    if not os.path.exists(file_name):
        if callable(model): model = model()
        if isinstance(model, str): model = load_model(model + '.h5')
        history = model.fit(dataset, dataset,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[earlystop],
                            validation_data=(dataset, dataset))
        model.save(file_name)

    # TODO: If analysis exists don't do this
    model = load_model(file_name)
    return analyse(fullset, model.predict(fullset, verbose=0), mask, model_name)


class model_name_generator:

    def __init__(self, base_name, start_num=0):
        self.base_name = base_name
        self.next_num = start_num

    def get_next(self, num=1):
        if num == 1:
            ret_val = self.base_name + '_' + str(self.next_num)
            self.next_num += 1
        else:
            ret_val = (self.base_name + '_' + str(i) for i in range(self.next_num, self.next_num + num))
            self.next_num += num
        return ret_val

    def get_last(self):
        return self.base_name + '_' + str(self.next_num - 1)


_, __, ___, auto_train = dataset1()
cluster_matrix = {}
model_names = model_name_generator('modelA1')
results = train(modelA1, np.arange(auto_train.shape[0]), model_names.get_next(), auto_train)
midway = np.sort(results['full'])[int(results['full'].shape[0] / 2)]
results_1 = train(modelA1, results['full'] < midway, model_names.get_next(), auto_train)
cluster_matrix[model_names.get_last()] = cluster_stats(results_1)

while cluster_matrix[model_names.get_last()]['overlap'] > 0:
    cluster = cluster_matrix[model_names.get_last()]
    mask = cluster['full'] < cluster['lhs']
    results = train(model_names.get_last(), mask, model_names.get_next(), auto_train)
    cluster_matrix[model_names.get_last()] = cluster_stats(results)
    save_obj(cluster_matrix, 'cluster_matrix.obj')





#plt.hist(results, bins=140)
#plt.show()
