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
        loss_total = loss_bg + loss_fg2 + loss_fg1
        return loss_total, loss_bg, loss_fg2, loss_fg1
        # return loss_total, loss_bg, loss_fg2, loss_fg1, loss_reg

    def hm_intensive_loss(self, hm_gt, hm_prs):
        # hm_gt = tf.where(hm_gt < 0.1, 0.0, 1.0) * hm_gt  #

        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1
        # ''''''
        loss_bg = 0
        loss_fg2 = 0
        loss_fg1 = 0
        threshold = LearningConfig.Loss_threshold
        stack_weight = [0.25, 0.5, 0.75, 3.5]
        for i, hm_pr in enumerate(hm_prs):
            delta_intensity = tf.math.abs(hm_gt - hm_pr)
            '''create high and low dif map'''
            high_dif_map = tf.where(delta_intensity >= threshold, 1.0, 0.0)
            low_dif_map = tf.where(delta_intensity < threshold, 1.0, 0.0)

            '''loss bg:'''
            loss_bg_low_dif = tf.math.reduce_mean(
                weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))
            loss_bg_high_dif = tf.math.reduce_mean(
                weight_map_bg * high_dif_map * (tf.math.square(hm_gt - hm_pr) - 0.5 * threshold ** 2))
            loss_bg += stack_weight[i] * (loss_bg_low_dif + loss_bg_high_dif)

            '''loss fg2'''
            loss_fg2_low_dif = tf.math.reduce_mean(
                weight_map_fg2 * low_dif_map * tf.math.square(hm_gt - hm_pr))

            loss_fg2_high_dif = tf.math.reduce_mean(
                weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold**2))

            loss_fg2 += stack_weight[i] * (loss_fg2_low_dif + loss_fg2_high_dif)

            '''loss fg1'''
            loss_fg1_high_dif = tf.math.reduce_mean(weight_map_fg1 * high_dif_map *
                                                    (tf.math.square(hm_gt - hm_pr) +
                                                     LearningConfig.Loss_fg_k * ln(1 + LearningConfig.Loss_threshold) -
                                                     LearningConfig.Loss_threshold ** 2))

            loss_fg1_low_dif = tf.math.reduce_mean(
                weight_map_fg1 * low_dif_map * (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)))

            loss_fg1 = stack_weight[i] * (loss_fg1_high_dif + loss_fg1_low_dif)

        return loss_bg, loss_fg2, loss_fg1

    def regression_loss(self, anno_gt, anno_prs):
        return 0
        # return tf.reduce_mean(tf.square(anno_gr - anno_pr))
