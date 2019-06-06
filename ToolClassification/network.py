from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras import metrics

import tensorflow as tf
import numpy as np

import load_dataset as data

# --------------------------------------------------------------------------
# LOADING THE DATA
# -------------------------------------
x_train, y_train, num_classes = data.training()
x_val, y_val, _ = data.validation()
x_test, y_test, _ = data.test()

image_shape = x_train[0].shape
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# DEFINING THE MODEL
# --------------------------------------------------------------------------
# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Freeze vgg16 base weights
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Create your own input format
input = Input(shape=image_shape,name = 'image_input')

# Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dense(num_classes, activation='sigmoid', name='predictions')(x)
# x = Dense(num_classes, activation='softmax', name='predictions')(x)

# Create your own model 
my_model = Model(inputs=input, outputs=x)

# model_vgg16_conv.summary()
# my_model.summary()
# --------------------------------------------------------------------------

def class_accuracy(y_true, y_pred):
        return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))

# --------------------------------------------------------------------------
# USING THE MODEL
# --------------------------------------------------------------------------
# training the model
my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
# my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = my_model.fit(x_train, y_train, epochs=8, validation_data=(x_val, y_val), verbose=1)
my_model.save('tool_model.h5')

# predict
prediction = my_model.predict(x_test)
print(class_accuracy(y_test, prediction))
