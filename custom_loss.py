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
        loss_hm = self.hm_intensive_loss(hm_gt, hm_prs)
        loss_reg = self.regression_loss(anno_gt, anno_prs)
        loss_total = loss_hm + loss_reg
        return loss_total, loss_hm, loss_reg

    def hm_intensive_loss(self, hm_gt, hm_prs):
        """return hm intensive loss"""
        '''create weight map for each hm_layer'''
        weight_map = tf.zeros_like(hm_gt)
        weight_map[hm_gt < self.theta_0] = self.omega_bg
        weight_map[np.where(np.logical_and(sample_hm >= self.theta_0, sample_hm < self.theta_1))] = self.omega_fg2
        weight_map[hm_gt >= self.theta_1] = self.omega_fg1

    def regression_loss(self, anno_gt, anno_prs):
        return 0
        # return tf.reduce_mean(tf.square(anno_gr - anno_pr))

