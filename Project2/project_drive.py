from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, image
from keras import models
from keras import layers
from keras import backend as K
from argparse import ArgumentParser
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from google.colab import drive 
drive.mount('/content/drive/')


image_size = 299

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/content/drive/My Drive/VCOM/data/train/',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        '/content/drive/My Drive/VCOM/data/validation/',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--classify", type=classify, action="store", help="classify an image on the test folder")
    parser.add_argument("-tc", "--train_classification", action="store_true")

    args = parser.parse_args()
    if args.train_classification:
        train_classification()

def classify(file):
    model = load_model('/content/drive/My Drive/VCOM/128d-1-100-5-120.h5')

    path = '/content/drive/My Drive/VCOM/data/test/test_images/' + file
    
    img = image.load_img(path, target_size=(image_size, image_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img)
    print(pred)
    
    predicted_class_indices = np.argmax(pred,axis=1)
    #print(predicted_class_indices)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    #print(labels)
    predictions = [labels[k] for k in predicted_class_indices]
    print(predictions)


def train_classification():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=50,
            verbose = 1)


    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

    model.fit_generator(
            train_generator,
            steps_per_epoch=120,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=75,
            verbose = 1)
    
    model.save('/content/drive/My Drive/VCOM/my_model.h5')  # creates a HDF5 file 'my_model.h5'


if __name__=='__main__':
    main()
