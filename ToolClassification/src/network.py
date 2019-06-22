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
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2) #included in our dependencies
train_generator=train_datagen.flow_from_directory('./data/training',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 subset='training')

validation_generator=train_datagen.flow_from_directory('./data/training',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 subset='validation')
num_classes = 4


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# DEFINING THE MODEL
# --------------------------------------------------------------------------
# Get back the convolutional part of a VGG network trained on ImageNet
model_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze vgg16 base weights
for layer in model_base.layers:
    layer.trainable = False

# Add the fully-connected layers 
x = model_base.output
x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dense(num_classes, activation='sigmoid', name='predictions')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

# Create your own model 
model = Model(inputs=model_base.input, outputs=x)

# model_base.summary()
# my_model.summary()

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# TRAINING THE MODEL
# --------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_validation=validation_generator.n//validation_generator.batch_size
model.fit_generator(
                generator=train_generator,
                steps_per_epoch=step_size_train,
                validation_data=validation_generator,
                validation_steps=step_size_validation,
                epochs=10)

model.save('./models/tool_model.h5')