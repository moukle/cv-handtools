from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# LOADING THE DATA
# -------------------------------------
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('dataset/unsorted',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# DEFINING THE MODEL
# --------------------------------------------------------------------------
# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze vgg16 base weights
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Add the fully-connected layers 
x = model_vgg16_conv.output
x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dense(num_classes, activation='sigmoid', name='predictions')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

# Create your own model 
my_model = Model(inputs=model_vgg16_conv.input, outputs=x)

# model_vgg16_conv.summary()
my_model.summary()
# --------------------------------------------------------------------------

def class_accuracy(y_true, y_pred):
        # return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))
        return np.mean(np.equal(y_true, np.argmax(y_pred, axis=-1)))

# --------------------------------------------------------------------------
# USING THE MODEL
# --------------------------------------------------------------------------
# training the model
# my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
# my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = my_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val), verbose=1)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
history = my_model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)

my_model.save('tool_model.h5')