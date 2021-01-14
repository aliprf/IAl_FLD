from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import cv2
import os.path
from scipy.spatial import distance
import scipy.io as sio
import efficientnet.tfkeras as efn
from HG_Network import HGNet

# from keras.engine import InputLayer
# import coremltools

# import efficientnet.keras as efn

# from tfkerassurgeon.operations import delete_layer, insert_layer


class CNNModel:
    def get_model(self, arch, num_landmark):
        if arch == 'hgNet':
            hg_net = HGNet(input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                           num_landmark=num_landmark // 2)
            return hg_net.create_model()














