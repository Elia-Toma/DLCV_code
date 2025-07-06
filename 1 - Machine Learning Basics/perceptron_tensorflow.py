import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

model = Sequential()
model.add(Dense(1, input_shape=(4,), activation='hard_sigmoid', kernel_initializer='zeros'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Create a simple dataset
data = np.array([
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [6.2, 3.4, 5.4, 2.3, 1],
    [5.9, 3.0, 5.1, 1.8, 1]
])
X_train = pd.DataFrame(data, columns=['feat1', 'feat2', 'feat3', 'feat4', 'label'])

X = X_train.values[:, 0:4]
y = X_train.values[:, 4]

model.fit(X, y, epochs=10, batch_size=1)

_, accuracy = model.evaluate(X, y)

print("%0.3f" % accuracy)