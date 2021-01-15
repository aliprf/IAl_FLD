from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
from numpy import load, save
from numpy import log as ln


class CustomLoss:
    def __init__(self, dataset_name, theta_0, theta_1, omega_bg, omega_fg2, omega_fg1):
        """
        :param dataset_name:
        :param theta_0: [0< theta_0) => bg  [theta_0, theta_1) => fg2
        :param theta_2: [theta_1, 1] => fg1
        :param omega_bg:
        :param omega_fg2:
        :param omega_fg1:
        """
        self.dataset_name = dataset_name
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.omega_bg = omega_bg
        self.omega_fg2 = omega_fg2
        self.omega_fg1 = omega_fg1

    def intensive_aware_loss(self, hm_gt, hm_prs, anno_gt, anno_prs):
        loss_bg, loss_fg2, loss_fg1 = self.hm_intensive_loss(hm_gt, hm_prs)
        # loss_reg = self.regression_loss(anno_gt, anno_prs)
        loss_total = loss_bg + 5*loss_fg2 + 10*loss_fg1
        return loss_total, loss_bg, loss_fg2, loss_fg1
        # return loss_total, loss_bg, loss_fg2, loss_fg1, loss_reg

    def hm_intensive_loss(self, hm_gt, hm_prs):
        # return tf.reduce_mean(tf.math.square(hm_gt - hm_prs[0])),\
        #        tf.reduce_mean(tf.math.square(hm_gt - hm_prs[1])),\
        #        tf.reduce_mean(tf.math.square(hm_gt - hm_prs[2]) )+ tf.reduce_mean(tf.math.square(hm_gt - hm_prs[3]))

        """return hm intensive loss"""
        '''create weight map for each hm_layer --hm : [batch, 56, 56, 68] '''
        # hm_gt = np.array(hm_gt)  # convert tf to np
        # '''create weigh-tmap'''
        # weight_map = np.zeros_like(hm_gt)
        # weight_map[hm_gt < self.theta_0] = self.omega_bg
        # weight_map[np.where(np.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1))] = self.omega_fg2
        # weight_map[hm_gt >= self.theta_1] = self.omega_fg1

        hm_gt = tf.where(hm_gt < 0.1, 0.0, 1.0) * hm_gt  # get rid of values that are smaller than 0.1

        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1
        # ''''''
        loss_bg = 0
        loss_fg2 = 0
        loss_fg1 = 0
        for i, hm_pr in enumerate(hm_prs):
            delta_intensity = tf.math.abs(hm_gt - hm_pr)
            '''create high and low dif map'''
            high_dif_map = tf.where(delta_intensity >= 1.0, 1.0, 0.0)
            low_dif_map = tf.where(delta_intensity < 1.0, 1.0, 0.0)

            '''loss bg:'''
            loss_bg_low_dif = ((i + 1) * 2) * tf.math.reduce_mean(
                weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))
            loss_bg_high_dif = ((i + 1) * 2) * tf.math.reduce_mean(
                weight_map_bg * high_dif_map * 0.5 * tf.math.abs(hm_gt - hm_pr))
            loss_bg += (loss_bg_low_dif + loss_bg_high_dif)

            '''loss fg2'''
            loss_fg2 += ((i + 1) * 2) * tf.math.reduce_mean(weight_map_fg2 * tf.math.abs(hm_gt - hm_pr))

            '''loss fg1'''
            loss_fg1_high_dif = ((i + 1) * 2) * tf.math.reduce_mean(weight_map_fg1 * high_dif_map *
                                                                    (2 * (tf.math.abs(hm_gt - hm_pr) - 1) +
                                                                     LearningConfig.Loss_fg_k * ln(2)))
            loss_fg1_low_dif = ((i + 1) * 2) * tf.math.reduce_mean(
                weight_map_fg1 * low_dif_map * (
                            LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)))
            loss_fg1 = loss_fg1_high_dif + loss_fg1_low_dif

        return loss_bg, loss_fg2, loss_fg1

    def regression_loss(self, anno_gt, anno_prs):
        return 0
        # return tf.reduce_mean(tf.square(anno_gr - anno_pr))
