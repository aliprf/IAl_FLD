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
import tensorflow as tf
from tensorflow import keras
# import keras
from skimage.transform import resize

from keras.regularizers import l2, l1

from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, \
    GlobalMaxPool2D
import efficientnet.tfkeras as efn

from hrNet import HrNet
import HourglassNet as hg
from hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile


class CNNModel:
    def get_model(self, arch, num_landmark, use_inter=True):
        if arch == 'hgNet':
            hg_net = HGNet(input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                           num_landmark=num_landmark // 2)
            return hg_net.create_model()

            # return hg.create_hourglass_network(num_classes=num_landmark // 2,
            #                                    num_stacks=4,
            #                                    num_channels=256,
            #                                    inres=InputDataSize.image_input_size,
            #                                    outres=InputDataSize.hm_size,
            #                                    bottleneck=bottleneck_block)

        elif arch == 'hrnet':
            hrnet = HrNet(input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                          num_landmark=num_landmark // 2, use_inter=use_inter)
            return hrnet.create_hr_net()

        elif arch == 'efn':
            return self.create_efficientNet_b3(
                input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                num_landmark=num_landmark)

        elif arch == 'arch_1d':
            return self.create_efficientNet_1d(
                input_shape=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                num_landmark=num_landmark)

    def create_efficientNet_1d(self, input_shape, num_landmark):
        eff_net = efn.EfficientNetB0(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=input_shape,
                                     pooling=None)
        # eff_net = mobilenet_v2.MobileNetV2(include_top=True,
        #                                    weights=None,
        #                                    input_tensor=None,
        #                                    input_shape=input_shape,
        #                                    pooling=None)

        eff_net.layers.pop()
        # eff_net.summary()

        inp = eff_net.input

        # top_activation = eff_net.get_layer('out_relu').output
        top_activation = eff_net.get_layer('top_activation').output

        '''8, 8, 256'''
        x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 1), padding='same')(
            top_activation)  # 16, 8, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding='same')(x)  # 16, 4, 256
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        '''16, 4, 256'''
        x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)  # 32, 4, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding='same')(x)  # 32, 2, 256
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        '''32, 2, 256'''
        x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 1), padding='same')(x)  # 64, 2, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        '''64, 2, 256'''
        out_heatmap = Conv2D(num_landmark // 2, kernel_size=1, padding='same', name='out_heatmap')(x)
        eff_net = Model(inp, [out_heatmap])
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("./model_arch/eff_net_1d.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def create_efficientNet_b3(self, input_shape, num_landmark):
        initializer = tf.keras.initializers.glorot_uniform()

        eff_net = efn.EfficientNetB3(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=input_shape,
                                     pooling=None)
        eff_net.layers.pop()
        # eff_net.summary()

        inp = eff_net.input

        top_activation = eff_net.get_layer('top_activation').output
        top_avg_pool = GlobalAveragePooling2D()(top_activation)
        ''''''
        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(
            top_activation)  # 16, 16, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)  # 32, 32, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)  # 64, 64, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        out_heatmap = Conv2D(num_landmark // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        '''reg part'''
        x = Dense(num_landmark * 3)(top_avg_pool)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(num_landmark * 2)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(num_landmark)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(rate=0.2)(x)

        out_reg = Dense(num_landmark, activation=keras.activations.linear, kernel_initializer=initializer,
                        use_bias=True, name='out')(x)

        eff_net = Model(inp, [out_heatmap, out_reg])
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("eff_net_H_R.json", "w") as json_file:
            json_file.write(model_json)

        return eff_net
