# --------------------------------------------------------------------------
# LOADING THE DATA
# --------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('./data/training',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('./data/training',
                                                         target_size=(224, 224),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         subset='validation')
NUMBER_OF_CLASSES = 4


# --------------------------------------------------------------------------
# DEFINING THE MODEL
# --------------------------------------------------------------------------
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

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
x = Dense(NUMBER_OF_CLASSES, activation='softmax', name='predictions')(x)

# Create model 
model = Model(inputs=model_base.input, outputs=x)

# model_base.summary()
# my_model.summary()

# --------------------------------------------------------------------------
# TRAINING THE MODEL
# --------------------------------------------------------------------------
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = validation_generator.n//validation_generator.batch_size

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(
                generator=train_generator,
                steps_per_epoch=step_size_train,
                validation_data=validation_generator,
                validation_steps=step_size_valid,
                epochs=10)

model.save('./models/tool_model.h5')
