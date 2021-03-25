from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, Add, \
    MaxPool2D, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, \
    GlobalMaxPool2D, ReLU, UpSampling2D, ZeroPadding2D

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime
import os.path
from scipy.spatial import distance
import scipy.io as sio


class HGNet:
    def __init__(self, input_shape, num_landmark):
        self.initializer = 'glorot_uniform'
        # self.initializer = tf.random_normal_initializer(0., 0.02)
        self.num_landmark = num_landmark
        self.input_shape = input_shape

    def _create_residual_block_blocks(self, input_layer, filters=256, use_bias=True):

        x = Conv2D(filters//2, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        x = Conv2D(filters//2, 3, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        x = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Add()([input_layer, x])

        out = ReLU()(x)
        return out

    def _create_bottle_neck_blocks(self, input_layer, filters=256, use_bias=True):
        x = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(
            input_layer)
        o_1 = BatchNormalization(momentum=0.9)(x)

        x = Conv2D(filters // 2, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(o_1)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        x = Conv2D(filters // 2, 3, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        x = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=True)(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Add()([o_1, x])

        x = ReLU()(x)

        return x

    def _create_conv_layer(self, input_layer, kernel_size, strides, filters):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer=self.initializer)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        # x = ReLU()(x)
        return x

    def _create_block(self, inp, is_first=False, is_last=False):
        l1 = self._create_bottle_neck_blocks(input_layer=inp)
        l1_res = self._create_bottle_neck_blocks(input_layer=l1)
        x = MaxPool2D(pool_size=2, strides=2)(l1)  # 32/28

        l2 = self._create_bottle_neck_blocks(input_layer=x)
        l2_res = self._create_bottle_neck_blocks(input_layer=l2)
        x = MaxPool2D(pool_size=2, strides=2)(l2)  # 16/14

        l3 = self._create_bottle_neck_blocks(input_layer=x)
        l3_res = self._create_bottle_neck_blocks(input_layer=l3)
        x = MaxPool2D(pool_size=2, strides=2)(l3)  # 8/7

        l4 = self._create_bottle_neck_blocks(input_layer=x)
        l4_res = self._create_bottle_neck_blocks(input_layer=l4)
        x = MaxPool2D(pool_size=2, strides=2)(l4)  # 4/3

        x = self._create_bottle_neck_blocks(input_layer=x)
        x = self._create_bottle_neck_blocks(input_layer=x)
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)

        x = Add()([l4_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Add()([l3_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Add()([l2_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Add()([l1_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        '''final'''
        x = Conv2D(filters=256, kernel_size=1, strides=1,
                   padding='same', kernel_initializer=self.initializer)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = ReLU()(x)

        o_2_out = Conv2D(filters=256, kernel_size=1, strides=1,
                         padding='same', kernel_initializer=self.initializer)(x)
        o_2_out = BatchNormalization(momentum=0.9)(o_2_out)

        if is_last:
            x = Conv2D(filters=256, kernel_size=1, strides=1,
                       padding='same', kernel_initializer=self.initializer)(x)
            x = BatchNormalization(momentum=0.9)(x)
            # x = ReLU()(x)

        out_loss_1 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1,
                            padding='same', kernel_initializer=self.initializer)(x)
        # out_loss_1 = BatchNormalization(momentum=0.9)(out_loss_1)

        # x = BatchNormalization(momentum=0.9)(out_loss_1)
        # x = ReLU()(x)

        # out_loss_next = self._create_bottle_neck_blocks(input_layer=x)
        out_loss_next = Conv2D(filters=256, kernel_size=1, strides=1,
                               padding='same', kernel_initializer=self.initializer)(out_loss_1)
        out_loss_next = BatchNormalization(momentum=0.9)(out_loss_next)

        if is_first:
            finish = Add()([o_2_out, out_loss_next])
        else:
            finish = Add()([inp, o_2_out, out_loss_next])

        return out_loss_1, finish

    def create_model(self):
        # 256 -> 128 -> 64
        # 224 -> 112 -> 56
        inp = Input(shape=self.input_shape)
        '''create header for 256 --> 64'''

        x = self._create_conv_layer(input_layer=inp, filters=256, kernel_size=7, strides=2)
        x = self._create_residual_block_blocks(input_layer=x, filters=256)

        # x = self._create_bottle_neck_blocks(input_layer=x, filters=64)
        # x = self._create_bottle_neck_blocks(input_layer=x, filters=64)
        # x = self._create_bottle_neck_blocks(input_layer=x, filters=128)

        x = MaxPool2D(pool_size=2, strides=2)(x)  # 64

        '''main blocks'''
        out_loss_1, finish_1 = self._create_block(inp=x, is_first=True)
        out_loss_2, finish_2 = self._create_block(inp=finish_1)
        out_loss_3, finish_3 = self._create_block(inp=finish_2)
        out_loss_4, finish_4 = self._create_block(inp=finish_3, is_last=True)
        '''new'''

        # out_loss_1: batch_size, hm, hm , lnd/2* : 100, 56 ,56 ,68
        # out_pnt_1: batch_size, 2, #lnd/2        : 100, 2, 68
        revised_model = Model(inp, [out_loss_1, out_loss_2, out_loss_3, out_loss_4])

        revised_model.summary()
        # tf.keras.utils.plot_model(revised_model, show_shapes=True, dpi=64)
        model_json = revised_model.to_json()
        with open("./model_arch/myHGN.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model
