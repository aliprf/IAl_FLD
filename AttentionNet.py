from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, Add, \
    MaxPool2D, \
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, \
    GlobalMaxPool2D, ReLU, UpSampling2D, ZeroPadding2D, Multiply

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


class AttNet:
    BN_MOMENTUM = 0.01
    NUM_FILTERS = 64

    def __init__(self, input_shape, num_landmark):
        self.initializer = 'glorot_uniform'
        # self.initializer = tf.random_normal_initializer(0., 0.02)
        self.num_landmark = num_landmark
        self.input_shape = input_shape

    def create_attention_net(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        l_front_steam = self._create_front_steam(input_layer=inputs, num_filters=self.NUM_FILTERS)
        '''========================1========================'''
        '''block_1'''
        b_1_attention = self._create_block(input_layer=l_front_steam, num_filters=self.NUM_FILTERS, num_out_channels=3)
        b_1_feature = self._create_block(input_layer=l_front_steam, num_filters=self.NUM_FILTERS,
                                         num_out_channels=self.num_landmark)
        '''fuse block_1'''
        b_1_fused = self._fuse_attention_features(att_layer=b_1_attention, fea_layer=b_1_feature)
        '''========================2========================'''
        '''block_2'''
        b_2_attention = self._create_block(input_layer=b_1_fused, num_filters=self.NUM_FILTERS, num_out_channels=3)
        b_2_feature = self._create_block(input_layer=b_1_fused, num_filters=self.NUM_FILTERS,
                                         num_out_channels=self.num_landmark)
        '''fuse block_2'''
        b_1_b_2_attention = Add()([b_1_attention, b_2_attention])
        b_1_b_2_feature = Add()([b_1_feature, b_2_feature])
        b_2_fused = self._fuse_attention_features(att_layer=b_1_b_2_attention, fea_layer=b_1_b_2_feature)
        '''========================3========================'''
        '''block_3'''
        b_3_attention = self._create_block(input_layer=b_2_fused, num_filters=self.NUM_FILTERS, num_out_channels=3)
        b_3_feature = self._create_block(input_layer=b_2_fused, num_filters=self.NUM_FILTERS,
                                         num_out_channels=self.num_landmark)
        '''fuse block_3'''
        b_2_b_3_attention = Add()([b_2_attention, b_3_attention])
        b_2_b_3_feature = Add()([b_2_feature, b_3_feature])
        b_3_fused = self._fuse_attention_features(att_layer=b_2_b_3_attention, fea_layer=b_2_b_3_feature)
        '''=======================4========================='''
        '''block_4'''
        b_4_attention = self._create_block(input_layer=b_3_fused, num_filters=self.NUM_FILTERS, num_out_channels=3)
        b_4_feature = self._create_block(input_layer=b_3_fused, num_filters=self.NUM_FILTERS,
                                         num_out_channels=self.num_landmark)
        '''fuse block_4'''
        b_3_b_4_attention = Add()([b_4_attention, b_3_attention])
        b_3_b_4_feature = Add()([b_4_feature, b_3_feature])
        b_4_fused = self._fuse_attention_features(att_layer=b_3_b_4_attention, fea_layer=b_3_b_4_feature)
        '''=======================5========================='''
        '''block_5'''
        b_5_attention = self._create_block(input_layer=b_4_fused, num_filters=self.NUM_FILTERS, num_out_channels=3)
        b_5_feature = self._create_block(input_layer=b_4_fused, num_filters=self.NUM_FILTERS,
                                         num_out_channels=self.num_landmark)
        '''fuse block_5'''
        b_4_b_5_attention = Add()([b_5_attention, b_4_attention])
        b_4_b_5_feature = Add()([b_5_feature, b_4_feature])
        b_5_fused = self._fuse_attention_features(att_layer=b_4_b_5_attention, fea_layer=b_4_b_5_feature)
        '''create model'''
        att_model = Model(inputs, [b_1_attention, b_1_fused,
                                   b_2_attention, b_2_fused,
                                   b_3_attention, b_3_fused,
                                   b_4_attention, b_4_fused,
                                   b_5_attention, b_5_fused
                                   ])

        att_model.summary()
        # tf.keras.utils.plot_model(revised_model, show_shapes=True, dpi=64)
        model_json = att_model.to_json()
        with open("./model_arch/att_model.json", "w") as json_file:
            json_file.write(model_json)
        return att_model

    def _fuse_attention_features(self, att_layer, fea_layer):
        initializer = tf.random_normal_initializer(0., 0.02)
        '''equalize both channels:'''
        # att
        att_layer_x = tf.keras.layers.Conv2D(self.NUM_FILTERS, 1, strides=1, padding='same',
                                             kernel_initializer=initializer, use_bias=False)(att_layer)
        att_layer_x = tf.keras.layers.BatchNormalization()(att_layer_x)
        att_layer_x = tf.keras.layers.LeakyReLU()(att_layer_x)

        # fea
        fea_layer_x = tf.keras.layers.Conv2D(self.NUM_FILTERS, 1, strides=1, padding='same',
                                             kernel_initializer=initializer, use_bias=False)(fea_layer)
        fea_layer_x = tf.keras.layers.BatchNormalization()(fea_layer_x)
        fea_layer_x = tf.keras.layers.LeakyReLU()(fea_layer_x)
        '''Multiply'''
        m_x = Multiply()([att_layer_x, fea_layer_x])

        x = Conv2D(filters=self.num_landmark, kernel_size=1, padding='same',
                   use_bias=False)(m_x)
        return x

    def _create_front_steam(self, input_layer, num_filters):
        x = Conv2D(filters=num_filters, kernel_size=3, strides=2, padding='same',
                   use_bias=False)(input_layer)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)

        x = Conv2D(filters=num_filters, kernel_size=3, strides=2, padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(momentum=self.BN_MOMENTUM)(x)
        x = ReLU()(x)
        return x

    def _create_block(self, input_layer, num_filters, num_out_channels):
        down_stack = [
            self._downsample(num_filters, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self._downsample(num_filters * 1, 4),  # (bs, 64, 64, 128)
            self._downsample(num_filters * 1, 4),  # (bs, 32, 32, 256)
            self._downsample(num_filters * 1, 4),  # (bs, 16, 16, 512)
            self._downsample(num_filters * 1, 4),  # (bs, 8, 8, 512)
            self._downsample(num_filters * 1, 4),  # (bs, 4, 4, 512)
            # self._downsample(num_filters * 8, 4),  # (bs, 2, 2, 512)
            # self._downsample(num_filters * 8, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            # self._upsample(num_filters * 8, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self._upsample(num_filters * 1, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self._upsample(num_filters * 1, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self._upsample(num_filters * 1, 4),  # (bs, 16, 16, 1024)
            self._upsample(num_filters * 1, 4),  # (bs, 32, 32, 512)
            self._upsample(num_filters * 1, 4),  # (bs, 64, 64, 256)
            self._upsample(num_filters, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(num_out_channels, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (bs, 256, 256, 3)

        x = input_layer

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        return x

        # return tf.keras.Model(inputs=input_layer, outputs=x)

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            UpSampling2D(size=(2, 2))
            # tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
            #                                 padding='same',
            #                                 kernel_initializer=initializer,
            #                                 use_bias=False)
        )

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
