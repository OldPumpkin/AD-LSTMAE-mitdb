import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

segLen = 360*7

ecgSeg = np.load('ecgSeg.npy')
ecgSeg = ecgSeg.reshape((-1,segLen,1))

ecgLabel = np.load('ecgLabel.npy')

print(ecgSeg.shape)


netInput = layers.Input(shape=(segLen,1))

lstmAE = tf.keras.models.Sequential()
lstmAE.add(tf.keras.layers.Input((segLen,1)))
lstmAE.add(tf.keras.layers.LSTM(128,activation='relu', return_sequences=True))
lstmAE.add(tf.keras.layers.LSTM(128,activation='relu', return_sequences=False))
lstmAE.add(tf.keras.layers.RepeatVector(segLen))
lstmAE.add(tf.keras.layers.LSTM(128,activation='relu', return_sequences=True))
lstmAE.add(tf.keras.layers.LSTM(128,activation='relu', return_sequences=True))
lstmAE.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

lstmAE.compile(optimizer='adam', loss='mse')

lstmAE.fit(ecgSeg[:128], ecgSeg[:128], epochs=5)