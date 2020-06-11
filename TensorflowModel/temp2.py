from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dataset = pd.read_csv("Use.csv")

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Strain')
test_labels = test_dataset.pop('Strain')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset.values)
normed_test_data = norm(test_dataset.values)


def build_model():
    built_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    built_model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
    return built_model
