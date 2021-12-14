import os
import numpy as np
from tensorflow import keras
import tensorflow
from tensorflow.python.keras.layers.advanced_activations import ReLU
import config as cf
#import prepare_data as predata
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tensorflow.keras import optimizers
import tensorflow as tf
import config as cf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def Model_EfficientNetB6():
    
    backbone_base = EfficientNetB6(include_top=False, weights='imagenet', input_shape=(cf.IMG_WIDTH, cf.IMG_HEIGHT, cf.NUM_COLORS))
    model = models.Sequential()
    model.add(backbone_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    model.add(layers.Dense(2304, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(cf.NUM_CLASS, activation='softmax'))
    backbone_base.trainable = True
    return model