import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist


# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# computer the number of labels
num_labels = len(np.unique(y_train))


# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# resize and normalize
img_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, img_size, img_size])
x_test = np.reshape(x_test,[-1, img_size, img_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# network parameters
input_shape = (img_size, img_size)
batch_size = 128
units = 256
dropout = 0.2



# model is RNN with 256 units, input is 28-dim vector 28 timesteps
model = Sequential()
model.add(SimpleRNN(units=units, dropout=dropout, input_shape=input_shape))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='rnn-mnist.png', show_shapes=True)



# loss function for one-hot vector
# use of sgd optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train the network
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))