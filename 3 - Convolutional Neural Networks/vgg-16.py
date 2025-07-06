def VGG_16(weights_path=None):
	model = models.Sequential()
	model.add(layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
	model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.ZeroPadding2D((1, 1)))
	model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(layers.Flatten())
	
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)
		
	return model




import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab import drive

drive.mount('/content/gdrive')


# prebuild model with pre-trained weights on imagenet
model = VGG16(weights='imagenet', include_top=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy')

# resize into VGG16 trained images' format
im = cv2.resize(cv2.imread('/content/gdrive/MyDrive/ColabNotebooks/cat.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)
im.astype(np.float32)

# predict
out = model.predict(im)
index = np.argmax(out)
print(index)

#plt.plot(out.ravel())
#plt.show()