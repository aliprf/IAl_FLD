from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
from numpy import load, save


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

        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1
        # ''''''
        loss_bg = 0
        loss_fg2 = 0
        loss_fg1 = 0
        for i, hm_pr in enumerate(hm_prs):
            hm_pr = np.array(hm_pr)  # convert tf to np
            loss_bg += ((i + 1) ** 2) * 0.5 * tf.math.reduce_mean(
                tf.math.multiply(weight_map_bg, tf.math.square(hm_gt - hm_pr)))

            loss_fg2 += ((i + 1) ** 2) * tf.math.reduce_mean(
                tf.math.multiply(weight_map_fg2, tf.math.abs(hm_gt - hm_pr)))

            loss_fg1 += ((i + 1) ** 2) * 0.5 * tf.math.reduce_mean(
                tf.math.multiply(weight_map_fg1,
                                 tf.math.square(hm_gt - hm_pr)))

        return loss_bg, loss_fg2, loss_fg1

    def regression_loss(self, anno_gt, anno_prs):
        return 0
        # return tf.reduce_mean(tf.square(anno_gr - anno_pr))
