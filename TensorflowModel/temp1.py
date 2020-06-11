from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import pandas as pd

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.set_printoptions(precision=3, suppress=True)

df = pd.read_csv("use.csv")

label = df.pop('Strain')

train_values, test_values, train_labels, test_labels = train_test_split(df.values, label.values,
                                                                        test_size=0.25, random_state=101)

train_dataset = tf.data.Dataset.from_tensor_slices((train_values, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_values, test_labels))
train_dataset = train_dataset.shuffle(len(train_values)).batch(1)
test_dataset = test_dataset.shuffle(len(test_values)).batch(1)

for feat, targ in train_dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['MeanAbsolutePercentageError'])
    return model


strain_model = get_compiled_model()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

strain_model.fit(train_dataset, epochs=15, callbacks=[cp_callback])

predictions = strain_model.evaluate(test_values, test_labels, verbose=1)

total_error = tf.reduce_sum(tf.square(tf.subtract(test_labels, tf.reduce_mean(test_labels))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(test_labels, predictions)))
R_squared = tf.subtract(1, tf.divide(unexplained_error, total_error))
R = tf.multiply(tf.sign(R_squared), tf.sqrt(tf.abs(R_squared)))

print(R_squared)
print(R)
