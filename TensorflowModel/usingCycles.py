from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import pandas as pd

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

voltage = pd.read_csv("voltageCycles.csv", header=0).to_numpy()
strain = pd.read_csv("strainCycles.csv", header=0).to_numpy()

# normalize data
# maxVoltage = np.amax(voltage)
# maxStrain = np.amax(strain)
# voltage = voltage / maxVoltage
# maxStrain = strain / maxStrain

#shuffled
train_voltage, test_voltage, train_strain, test_strain = train_test_split(voltage, strain,
                                                                          test_size=0.25, random_state=101)

#not shuffled
# train_voltage, test_voltage, train_strain, test_strain = train_test_split(voltage, strain,
#                                                                           test_size=0.25, shuffle=False)

# flatten the cycles into one long list of data
train_voltage = train_voltage.flatten()
test_voltage = test_voltage.flatten()
train_strain = train_strain.flatten()
test_strain = test_strain.flatten()

train_voltage = train_voltage[~np.isnan(train_voltage)]
test_voltage = test_voltage[~np.isnan(test_voltage)]
train_strain = train_strain[~np.isnan(train_strain)]
test_strain = test_strain[~np.isnan(test_strain)]


# turn into tensotflow dataset if I want
# train_dataset = tf.data.Dataset.from_tensor_slices((train_voltage, train_strain))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_voltage, test_strain))

def build_model():
    built_model = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    built_model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                        loss=keras.losses.mape,
                        metrics=[keras.losses.mape])
    return built_model


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_voltage, train_strain, epochs=1000,
                    validation_split=0.2, verbose=0, callbacks=[early_stop])

loss, mape = model.evaluate(test_voltage, test_strain, verbose=2)

run = 'stop'

# voltageData = tf.data.Dataset.from_tensor_slices(voltage.values)
# strainData = tf.data.Dataset.from_tensor_slices(strain.values)
# dataset = tf.data.Dataset.zip((voltageData, strainData))
#
# (train_voltage, train_Strain), (test_voltage, test_strain) = dataset.load_data()

# firstVoltageCycle = voltageCycles[0]
# firstStrainCycle = strainCycles[0]
