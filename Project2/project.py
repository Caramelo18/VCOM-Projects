from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.applications import VGG16, VGG19, InceptionV3, Xception
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import backend as K
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    ## load dataset
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # ind_train = random.sample(list(range(x_train.shape[0])), 2000)
    # x_train = x_train[ind_train]
    # y_train = y_train[ind_train]
    #
    # ind_test = random.sample(list(range(x_test.shape[0])), 1000)
    # x_test = x_test[ind_test]
    # y_test = y_test[ind_test]
    #
    # inception_x_train = resize_data(x_train, [x_train.shape[0], 139, 139, 3])/255
    # inception_x_test = resize_data(x_test, [x_test.shape[0], 139, 139, 3])/255
    #
    #
    # y_train_hot_encoded = to_categorical(y_train)
    # y_test_hot_encoded = to_categorical(y_test)

    ## train the model
    #hist_inception = inception_fine_tune(inception_x_train, y_train_hot_encoded)

    #show_history(hist_inception, 'inception')
    ## test the model
    test_tune()

def inception_fine_tune(x_train, y_train):
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inception_model.input, outputs=predictions)
    for layer in inception_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    for i, layer in enumerate(model.layers):

        if i < 249:
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=256, epochs=50, shuffle=True, validation_split=0.1)
    return history

def test_tune():
    image_size = 299
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    for layer in inception_model.layers[:-4]:
        layer.trainable = False

    model = inception_model.output
    model = GlobalAveragePooling2D()(model)
    # let's add a fully-connected layer
    model = Dense(1024, activation='relu')(model)
    predictions = Dense(5, activation='softmax')(model)

    model = Model(inputs=inception_model.input, outputs=predictions)

    for layer in inception_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/images/',
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode='categorical')


    validation_generator = test_datagen.flow_from_directory(
            'data/images/',
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode='categorical')



    model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=400)

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    model.summary()


if __name__=='__main__':
    main()
