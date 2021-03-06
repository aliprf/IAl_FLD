from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, Add, \
    MaxPool2D, Softmax, \
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


class HrNet:
    BN_MOMENTUM = 0.01

    def __init__(self, input_shape, num_landmark):
        self.input_shape = input_shape
        self.num_landmark = num_landmark

    def create_hr_net(self):
        stage_repeat = [1, 1, 4, 3]
        residual_repeat = 4

        inp = Input(shape=self.input_shape)

        '''first stem'''
        stem_x = self._create_front_steam(inp)

        '''create stage one'''
        _inp_layer = stem_x
        for i in range(residual_repeat):
            res_bl1 = self._make_bottleneck_stg1(input_layer=_inp_layer, filters=64, strides=1)
            _inp_layer = res_bl1
        '''stage 1 creating the transition layers'''
        x_resolution_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(_inp_layer)
        x_resolution_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_resolution_1)
        x_resolution_1 = ReLU()(x_resolution_1)

        x_resolution_2 = Conv2D(filters=64 * 2, kernel_size=3, strides=2, padding='same', use_bias=False)(_inp_layer)
        x_resolution_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_resolution_2)
        x_resolution_2 = ReLU()(x_resolution_2)

        '''create stage two'''
        _inp_layer_bl2_br1 = x_resolution_1
        _inp_layer_bl2_br2 = x_resolution_2
        for i in range(residual_repeat):
            res_bl2_br1 = self._make_bottleneck(input_layer=_inp_layer_bl2_br1, filters=64, strides=1)
            _inp_layer_bl2_br1 = res_bl2_br1
            res_bl2_br2 = self._make_bottleneck(input_layer=_inp_layer_bl2_br2, filters=64*2, strides=1)
            _inp_layer_bl2_br2 = res_bl2_br2
        '''stage 2 connecting different resolutions'''
        _inp_layer_bl3_br1, _inp_layer_bl3_br2, _inp_layer_bl3_br3 = self._create_transition_layers_stage_2(
            _inp_layer_bl2_br1=_inp_layer_bl2_br1, _inp_layer_bl2_br2=_inp_layer_bl2_br2)

        '''create stage three'''
        for k in range(stage_repeat[2]):
            for i in range(residual_repeat):
                res_bl3_br1 = self._make_bottleneck(input_layer=_inp_layer_bl3_br1, filters=64, strides=1)
                _inp_layer_bl3_br1 = res_bl3_br1
                res_bl3_br2 = self._make_bottleneck(input_layer=_inp_layer_bl3_br2, filters=64*2, strides=1)
                _inp_layer_bl3_br2 = res_bl3_br2
                res_bl3_br3 = self._make_bottleneck(input_layer=_inp_layer_bl3_br3, filters=64 * 4, strides=1)
                _inp_layer_bl3_br3 = res_bl3_br3
        '''stage 3 connecting different resolutions'''
        _inp_layer_bl4_br1, _inp_layer_bl4_br2, _inp_layer_bl4_br3, _inp_layer_bl4_br4 = \
            self._create_transition_layers_stage_3(_inp_layer_bl3_br1=_inp_layer_bl3_br1,
                                                   _inp_layer_bl3_br2=_inp_layer_bl3_br2,
                                                   _inp_layer_bl3_br3=_inp_layer_bl3_br3)
        '''create stage four'''
        for k in range(stage_repeat[3]):
            for i in range(residual_repeat):
                res_bl4_br1 = self._make_bottleneck(input_layer=_inp_layer_bl4_br1, filters=64, strides=1)
                _inp_layer_bl4_br1 = res_bl4_br1
                res_bl4_br2 = self._make_bottleneck(input_layer=_inp_layer_bl4_br2, filters=64*2, strides=1)
                _inp_layer_bl4_br2 = res_bl4_br2
                res_bl4_br3 = self._make_bottleneck(input_layer=_inp_layer_bl4_br3, filters=64 * 4, strides=1)
                _inp_layer_bl4_br3 = res_bl4_br3
                res_bl4_br4 = self._make_bottleneck(input_layer=_inp_layer_bl4_br4, filters=64 * 8, strides=1)
                _inp_layer_bl4_br4 = res_bl4_br4
        '''stage 3 connecting different resolutions'''
        _inp_layer_bl5_br1, _inp_layer_bl5_br2, _inp_layer_bl5_br3, _inp_layer_bl5_br4 =\
            self._create_transition_layers_stage_4(_inp_layer_bl4_br1=_inp_layer_bl4_br1,
                                                   _inp_layer_bl4_br2=_inp_layer_bl4_br2,
                                                   _inp_layer_bl4_br3=_inp_layer_bl4_br3,
                                                   _inp_layer_bl4_br4=_inp_layer_bl4_br4)
        '''fuse last layers'''
        out_1 = self._fuse_layer_1(_inp_layer_bl2_br1=_inp_layer_bl2_br1,
                                   _inp_layer_bl2_br2=_inp_layer_bl2_br2)

        out_2 = self._fuse_layers_2(_inp_layer_bl3_br1=_inp_layer_bl3_br1,
                                    _inp_layer_bl3_br2=_inp_layer_bl3_br2,
                                    _inp_layer_bl3_br3=_inp_layer_bl3_br3)

        out_3 = self._fuse_layers_3(_inp_layer_bl4_br1=_inp_layer_bl4_br1,
                                    _inp_layer_bl4_br2=_inp_layer_bl4_br2,
                                    _inp_layer_bl4_br3=_inp_layer_bl4_br3,
                                    _inp_layer_bl4_br4=_inp_layer_bl4_br4)

        out_4 = self._fuse_last_layers(_inp_layer_bl5_br1=_inp_layer_bl5_br1,
                                       _inp_layer_bl5_br2=_inp_layer_bl5_br2,
                                       _inp_layer_bl5_br3=_inp_layer_bl5_br3,
                                       _inp_layer_bl5_br4=_inp_layer_bl5_br4)
        '''finish'''
        out_loss_1 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1, name='out_loss_1')(out_1)
        out_loss_2 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1, name='out_loss_2')(out_2)
        out_loss_3 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1, name='out_loss_3')(out_3)
        out_loss_4 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1, name='out_loss_4')(out_4)

        '''depict and creat model'''
        revised_model = Model(inp, [out_loss_1, out_loss_2, out_loss_3, out_loss_4])
        revised_model.summary()

        model_json = revised_model.to_json()
        with open("./model_arch/hrNet.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def _create_front_steam(self, input_layer):
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                   use_bias=False)(input_layer)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)
        # x = Softmax(1)(x)
        return x

    def _make_bottleneck_stg1(self, input_layer, filters, strides=1):
        x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same',
                   use_bias=False)(input_layer)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters*4, kernel_size=1, strides=strides, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)

        '''down sample:'''
        x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)
        ''''''
        x = Add()([input_layer, x])
        x = ReLU()(x)

        return x

    def _make_bottleneck(self, input_layer, filters, strides=1):
        x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                   use_bias=False)(input_layer)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)
        # '''down sample:'''

        x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)

        '''down sample:'''
        if x.shape[-1] != input_layer.shape[-1]:
            x = Conv2D(filters=input_layer.shape[-1], kernel_size=1, strides=strides, padding='same',
                       use_bias=False)(x)
            x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
            x = ReLU()(x)
        ''''''

        x = Add()([input_layer, x])

        x = ReLU()(x)

        return x

    def _create_transition_layers_stage_2(self, _inp_layer_bl2_br1, _inp_layer_bl2_br2 ):
        # create 256
        x_up_l1 = UpSampling2D(size=(2, 2))(_inp_layer_bl2_br2)
        x_up_l1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1)
        x_up_l1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1)
        x_up_l1 = ReLU()(x_up_l1)
        _inp_layer_bl3_br1 = Add()([_inp_layer_bl2_br1, x_up_l1])

        # create 128
        x_down_2 = Conv2D(filters=64 * 2, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl2_br1)
        x_down_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_2)
        x_down_2 = ReLU()(x_down_2)
        _inp_layer_bl3_br2 = Add()([_inp_layer_bl2_br2, x_down_2])

        # create 64
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl2_br1)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_3_0)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)

        x_down_3_1 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl2_br2)
        x_down_3_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_1)
        x_down_3_1 = ReLU()(x_down_3_1)

        _inp_layer_bl3_br3 = Add()([x_down_3_0, x_down_3_1])
        return _inp_layer_bl3_br1, _inp_layer_bl3_br2, _inp_layer_bl3_br3

    def _create_transition_layers_stage_3(self, _inp_layer_bl3_br1, _inp_layer_bl3_br2, _inp_layer_bl3_br3):
        # create 256
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl3_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        x_up_l1_1 = UpSampling2D(size=(2, 2))(_inp_layer_bl3_br3)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)
        x_up_l1_1 = UpSampling2D(size=(2, 2))(x_up_l1_1)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)

        _inp_layer_bl4_br1 = Add()([_inp_layer_bl3_br1, x_up_l1_0, x_up_l1_1])

        # create 128
        x_down_2 = Conv2D(filters=64 * 2, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl3_br1)
        x_down_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_2)
        x_down_2 = ReLU()(x_down_2)

        x_up_l2_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl3_br3)
        x_up_l2_0 = Conv2D(filters=64 * 2, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l2_0)
        x_up_l2_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l2_0)
        x_up_l2_0 = ReLU()(x_up_l2_0)

        _inp_layer_bl4_br2 = Add()([_inp_layer_bl3_br2, x_down_2, x_up_l2_0])

        # create 64
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl3_br1)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_3_0)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)
        #
        x_down_3_1 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl3_br2)
        x_down_3_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_1)
        x_down_3_1 = ReLU()(x_down_3_1)
        #
        _inp_layer_bl4_br3 = Add()([_inp_layer_bl3_br3, x_down_3_0, x_down_3_1])
        #######
        # create 32
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl3_br1)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_0)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_0)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        #
        x_down_4_1 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl3_br2)
        x_down_4_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_1)
        x_down_4_1 = ReLU()(x_down_4_1)
        x_down_4_1 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_1)
        x_down_4_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_1)
        x_down_4_1 = ReLU()(x_down_4_1)
        #
        x_down_4_2 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl3_br3)
        x_down_4_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_2)
        x_down_4_2 = ReLU()(x_down_4_2)

        _inp_layer_bl4_br4 = Add()([x_down_4_0, x_down_4_1, x_down_4_2])

        return _inp_layer_bl4_br1, _inp_layer_bl4_br2, _inp_layer_bl4_br3, _inp_layer_bl4_br4

    def _create_transition_layers_stage_4(self, _inp_layer_bl4_br1, _inp_layer_bl4_br2, _inp_layer_bl4_br3,
                                          _inp_layer_bl4_br4):
        # 256
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        x_up_l1_1 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br3)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)
        x_up_l1_1 = UpSampling2D(size=(2, 2))(x_up_l1_1)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)

        x_up_l1_2 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br4)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)

        _inp_layer_bl5_br1 = Add()([_inp_layer_bl4_br1, x_up_l1_0, x_up_l1_1, x_up_l1_2])

        # create 128
        x_down_2 = Conv2D(filters=64 * 2, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl4_br1)
        x_down_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_2)
        x_down_2 = ReLU()(x_down_2)

        x_up_l2_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br3)
        x_up_l2_0 = Conv2D(filters=64*2, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l2_0)
        x_up_l2_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l2_0)
        x_up_l2_0 = ReLU()(x_up_l2_0)

        x_up_l2_2 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br4)
        x_up_l2_2 = Conv2D(filters=64*2, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l2_2)
        x_up_l2_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l2_2)
        x_up_l2_2 = ReLU()(x_up_l2_2)
        x_up_l2_2 = UpSampling2D(size=(2, 2))(x_up_l2_2)
        x_up_l2_2 = Conv2D(filters=64*2, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l2_2)
        x_up_l2_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l2_2)
        x_up_l2_2 = ReLU()(x_up_l2_2)

        _inp_layer_bl5_br2 = Add()([_inp_layer_bl4_br2, x_down_2, x_up_l2_0, x_up_l2_2])

        # create 64
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl4_br1)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)
        x_down_3_0 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_3_0)
        x_down_3_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_0)
        x_down_3_0 = ReLU()(x_down_3_0)

        x_down_3_1 = Conv2D(filters=64 * 4, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl4_br2)
        x_down_3_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_3_1)
        x_down_3_1 = ReLU()(x_down_3_1)

        x_up_l3_2 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br4)
        x_up_l3_2 = Conv2D(filters=64*4, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l3_2)
        x_up_l3_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l3_2)
        x_up_l3_2 = ReLU()(x_up_l3_2)

        _inp_layer_bl5_br3 = Add()([_inp_layer_bl4_br3, x_down_3_0, x_down_3_1, x_up_l3_2])

        # create 64
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl4_br1)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_0)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        x_down_4_0 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_0)
        x_down_4_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_0)
        x_down_4_0 = ReLU()(x_down_4_0)
        #
        x_down_4_1 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(
            _inp_layer_bl4_br2)
        x_down_4_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_1)
        x_down_4_1 = ReLU()(x_down_4_1)
        x_down_4_1 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(x_down_4_1)
        x_down_4_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_1)
        x_down_4_1 = ReLU()(x_down_4_1)
        #
        x_down_4_2 = Conv2D(filters=64 * 8, kernel_size=1, strides=2, padding='same', use_bias=False)(_inp_layer_bl4_br3)
        x_down_4_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_down_4_2)
        x_down_4_2 = ReLU()(x_down_4_2)

        _inp_layer_bl5_br4 = Add()([x_down_4_0, x_down_4_1, x_down_4_2, _inp_layer_bl4_br4])

        return _inp_layer_bl5_br1, _inp_layer_bl5_br2, _inp_layer_bl5_br3, _inp_layer_bl5_br4

    def _fuse_last_layers(self, _inp_layer_bl5_br1, _inp_layer_bl5_br2, _inp_layer_bl5_br3, _inp_layer_bl5_br4):
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl5_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        x_up_l1_1 = UpSampling2D(size=(2, 2))(_inp_layer_bl5_br3)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)
        x_up_l1_1 = UpSampling2D(size=(2, 2))(x_up_l1_1)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)

        x_up_l1_2 = UpSampling2D(size=(2, 2))(_inp_layer_bl5_br4)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)

        return Concatenate()([_inp_layer_bl5_br1, x_up_l1_0, x_up_l1_1, x_up_l1_2])

    def _fuse_layer_1(self, _inp_layer_bl2_br1, _inp_layer_bl2_br2):
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl2_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        return Concatenate()([_inp_layer_bl2_br1, x_up_l1_0])

    def _fuse_layers_2(self, _inp_layer_bl3_br1, _inp_layer_bl3_br2, _inp_layer_bl3_br3):
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl3_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        x_up_l1_1 = UpSampling2D(size=(2, 2))(_inp_layer_bl3_br3)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)
        x_up_l1_1 = UpSampling2D(size=(2, 2))(x_up_l1_1)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)

        return Concatenate()([_inp_layer_bl3_br1, x_up_l1_0, x_up_l1_1])

    def _fuse_layers_3(self, _inp_layer_bl4_br1, _inp_layer_bl4_br2, _inp_layer_bl4_br3, _inp_layer_bl4_br4):
        x_up_l1_0 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br2)
        x_up_l1_0 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_0)
        x_up_l1_0 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_0)
        x_up_l1_0 = ReLU()(x_up_l1_0)

        x_up_l1_1 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br3)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)
        x_up_l1_1 = UpSampling2D(size=(2, 2))(x_up_l1_1)
        x_up_l1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_1)
        x_up_l1_1 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_1)
        x_up_l1_1 = ReLU()(x_up_l1_1)

        x_up_l1_2 = UpSampling2D(size=(2, 2))(_inp_layer_bl4_br4)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)
        x_up_l1_2 = UpSampling2D(size=(2, 2))(x_up_l1_2)
        x_up_l1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False)(x_up_l1_2)
        x_up_l1_2 = BatchNormalization(momentum=self.BN_MOMENTUM)(x_up_l1_2)
        x_up_l1_2 = ReLU()(x_up_l1_2)

        return Concatenate()([_inp_layer_bl4_br1, x_up_l1_0, x_up_l1_1, x_up_l1_2])

