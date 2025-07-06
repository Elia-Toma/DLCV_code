from keras import models, layers

IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10  # number of outputs = number of digits

#define the convnet
def build(input_shape, classes):
    model = models.Sequential()
    # CONV => RELU => POOL
    model.add(layers.Convolution2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    model.add(layers.Convolution2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten => RELU layers
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    # a softmax classifier
    model.add(layers.Dense(classes, activation='softmax'))
    return model