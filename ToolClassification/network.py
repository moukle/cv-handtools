from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import cv2

# loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# converting it to RGB
x_train = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')
num_classes = len(set(y_train))


# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Freeze vgg16 base weights
for layer in model_vgg16_conv.layers:
    layer.trainable = False

model_vgg16_conv.summary()

# Create your own input format (here 3x32x32)
input = Input(shape=(128,128,3),name = 'image_input')

# Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

# Create your own model 
my_model = Model(inputs=input, outputs=x)

# In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

# training the model
my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = my_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=1)