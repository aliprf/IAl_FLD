import data_helper
from configuration import DatasetName, DatasetType, InputDataSize, LearningConfig, CategoricalLabels
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
from numpy import load, save
from numpy import log as ln
from skimage.transform import resize


class CustomLoss:
    def __init__(self, dataset_name, number_of_landmark, theta_0, theta_1, omega_bg, omega_fg2, omega_fg1):
        """
        :param dataset_name:
        :param theta_0: [0< theta_0) => bg  [theta_0, theta_1) => fg2
        :param theta_2: [theta_1, 1] => fg1
        :param omega_bg:
        :param omega_fg2:
        :param omega_fg1:
        """
        self.dataset_name = dataset_name
        self.number_of_landmark = number_of_landmark
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.omega_bg = omega_bg
        self.omega_fg2 = omega_fg2
        self.omega_fg1 = omega_fg1

    def intensity_aware_loss_1d(self, hm_gt, hm_pr, hm_gt_2d, multi_loss):
        """"""
        weight = 10

        if multi_loss:
            hm_pr_8_2d = hm_pr[1]
            hm_pr_16_2d = hm_pr[2]
            hm_pr_32_2d = hm_pr[3]
            hm_pr_64_2d = hm_pr[4]
            hm_pr = hm_pr[0]
            '''main'''
            hm_pr_2d_vectorized = self.create_hm_pr_2d(hm=hm_pr, hm_gt_2d=hm_gt_2d)
            hm_gt_2d_vectorized = self.create_hm_pr_2d(hm=hm_gt, hm_gt_2d=hm_gt_2d)

            _loss_bg, _loss_fg2, _loss_fg1, _loss_categorical = self.hm_intensive_loss_1d(hm_gt=hm_gt, hm_pr=hm_pr)
            _loss_bg_2d, _loss_fg2_2d, _loss_fg1_2d, _loss_categorical_2d =\
                self.hm_intensive_loss_1d(hm_gt=hm_gt_2d_vectorized, hm_pr=hm_pr_2d_vectorized)
            _loss_bg += 10 * _loss_bg_2d
            _loss_fg2 += 10 * _loss_fg2_2d
            _loss_fg1 += 10 * _loss_fg1_2d
            _loss_categorical += 10 * _loss_categorical_2d

            '''8'''
            hm_gt_2d_8 = tf.image.resize(hm_gt_2d, size=(8, 8))
            _loss_bg_8, _loss_fg2_8, _loss_fg1_8, _loss_categorical_8 = self.hm_intensive_loss_1d(hm_gt=hm_gt_2d_8,
                                                                                                  hm_pr=hm_pr_8_2d)
            '''16'''
            hm_gt_2d_16 = tf.image.resize(hm_gt_2d, size=(16, 16))
            _loss_bg_16, _loss_fg2_16, _loss_fg1_16, _loss_categorical_16 = self.hm_intensive_loss_1d(hm_gt=hm_gt_2d_16,
                                                                                                      hm_pr=hm_pr_16_2d)
            '''32'''
            hm_gt_2d_32 = tf.image.resize(hm_gt_2d, size=(32, 32))
            _loss_bg_32, _loss_fg2_32, _loss_fg1_32, _loss_categorical_32 = self.hm_intensive_loss_1d(hm_gt=hm_gt_2d_32,
                                                                                                      hm_pr=hm_pr_32_2d)
            '''64'''
            _loss_bg_64, _loss_fg2_64, _loss_fg1_64, _loss_categorical_64 = self.hm_intensive_loss_1d(hm_gt=hm_gt_2d,
                                                                                                      hm_pr=hm_pr_64_2d)

            loss_bg = _loss_bg + 0.01 * (_loss_bg_8 + _loss_bg_16 + _loss_bg_32 + _loss_bg_64)
            loss_fg2 = _loss_fg2 + 0.01 * (_loss_fg2_8 + _loss_fg2_16 + _loss_fg2_32 + _loss_fg2_64)
            loss_fg1 = _loss_fg1 + 0.01 * (_loss_fg1_8 + _loss_fg1_16 + _loss_fg1_32 + _loss_fg1_64)
            loss_categorical = _loss_categorical + 0.01 * (_loss_categorical_8 + _loss_categorical_16 +
                                                           _loss_categorical_32 + _loss_categorical_64)
        else:
            loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_1d(hm_gt=hm_gt,
                                                                                      hm_pr=hm_pr)

        loss_total = weight * (loss_bg + loss_fg2 + loss_fg1) + 0.1 * loss_categorical
        return loss_total, loss_bg, loss_fg2, loss_fg1, loss_categorical

    def intensity_aware_loss(self, hm_gt, hm_prs, att_gt, att_prs, use_inter):
        weight = 3
        if use_inter:
            loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_stacked(hm_gt, hm_prs)

        else:
            loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_stacked_single(hm_gt, hm_prs)

        loss_attention = self.attention_loss(att_gt, att_prs)

        loss_total = weight * (loss_bg + loss_fg2 + loss_fg1) + loss_attention + 0.1 * loss_categorical

        return loss_total, loss_bg, loss_fg2, loss_fg1, loss_categorical, loss_attention

    def attention_loss(self, att_gt, att_prs):
        loss_attention = 0
        stack_weight = [1.0, 1.0, 1.0, 1.0, 3.0]  # two times more than sum of the previous layers
        for i, att_pr in enumerate(att_prs):
            loss_attention += stack_weight[i]*tf.math.reduce_mean(tf.math.abs(att_pr - att_gt))
        return loss_attention


    def _intensity_aware_loss(self, hm_gt, hm_prs, use_inter):
        weight = 20
        if use_inter:
            loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_stacked(hm_gt, hm_prs)
        else:
            loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_stacked_single(hm_gt, hm_prs)

        loss_total = weight * (loss_bg + loss_fg2 + loss_fg1) + 1.0 * loss_categorical
        return loss_total, loss_bg, loss_fg2, loss_fg1, loss_categorical

    def intensity_aware_loss_with_reg(self, hm_gt, hm_pr, anno_gt, anno_pr):
        """"""
        weight = 2
        '''hm loss'''
        loss_bg, loss_fg2, loss_fg1, loss_categorical = self.hm_intensive_loss_efn(hm_gt, hm_pr)
        loss_total_hm = (weight * (loss_bg + loss_fg2 + loss_fg1) + 0.1 * loss_categorical)

        '''regression loss'''
        loss_reg = 2 * self.regression_loss(anno_gt, anno_pr)

        '''regularization part'''
        regularization = 0  # weight * self.calculate_regularization(hm_pr, anno_pr)

        loss_total = loss_total_hm + loss_reg + regularization
        return loss_total, loss_total_hm, loss_reg, regularization, loss_bg, loss_fg2, loss_fg1, loss_categorical

    def create_hm_pr_2d(self, hm, hm_gt_2d):
        """
        :param hm_gt_2d:
        :param hm_pr: bsize * 64 *2 * #lnd_points
        :return: hm_pr_2d: bsize * 64 * 64 * #lnd_points 
        """
        hm = np.array(hm)
        hm_2d = np.zeros_like(hm_gt_2d) * 0.01  # 0.01 is noise
        for bs_i in range(hm.shape[0]):
            for lnd_i in range(hm.shape[3]):
                x_indices = (-hm[bs_i, :, 0, lnd_i]).argsort()[:1]
                y_indices = (-hm[bs_i, :, 1, lnd_i]).argsort()[:1]
                hm_2d[bs_i, x_indices, :, lnd_i] = hm[bs_i, :, 1, lnd_i]
                hm_2d[bs_i, :, y_indices, lnd_i] = hm[bs_i, :, 0, lnd_i]
        return hm_2d

    def calculate_regularization(self, hm_pr, anno_pr):
        """"""
        '''convert hm to points'''
        batch_size = tf.shape(hm_pr)[0]

        converted_points = tf.stack([tf.reshape(
            [self._find_weighted_avg_of_n_biggest_points(heatmap=hm_pr[b_index, :, :, i], num_of_avg_points=2)
             for i in range(self.number_of_landmark // 2)], shape=[self.number_of_landmark]) for b_index in
            range(batch_size)], axis=0)
        '''calculate regularization'''
        regu = tf.reduce_mean(tf.math.abs(converted_points - anno_pr))
        return regu

    def regression_loss(self, anno_gt, anno_prs):
        """"""
        '''create high and low dif map'''
        threshold = LearningConfig.Loss_threshold
        delta_reg = tf.math.abs(anno_gt - anno_prs)
        high_dif_map = tf.where(delta_reg >= threshold, 1.0, 0.0)
        low_dif_map = tf.where(delta_reg < threshold, 1.0, 0.0)

        loss_high_dif = tf.math.reduce_mean(high_dif_map *
                                            (tf.math.square(anno_gt - anno_prs) +
                                             LearningConfig.Loss_fg_k * ln(1 + LearningConfig.Loss_threshold) -
                                             LearningConfig.Loss_threshold ** 2))
        loss_low_dif = tf.math.reduce_mean(
            low_dif_map * (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(anno_gt - anno_prs) + 1)))
        loss_reg = (loss_high_dif + loss_low_dif)

        return loss_reg

    def hm_intensive_loss_efn(self, hm_gt, hm_pr):
        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1

        gt_categorical = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.fg_1 + \
                         weight_map_fg2 / self.omega_fg2 * CategoricalLabels.fg_2 + \
                         weight_map_bg / self.omega_bg * CategoricalLabels.bg
        gt_categorical = tf.cast(gt_categorical, dtype=tf.int32)
        gt_categorical = tf.one_hot(gt_categorical, depth=3)
        gt_categorical_weight = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.w_fg_1 + \
                                weight_map_fg2 / self.omega_fg2 * CategoricalLabels.w_fg_2 + \
                                weight_map_bg / self.omega_bg * CategoricalLabels.w_bg
        # ''''''
        threshold = LearningConfig.Loss_threshold
        threshold_2 = LearningConfig.Loss_threshold_2

        '''loss categorical'''
        pr_categorical_map_bg = tf.where(hm_pr < self.theta_0, CategoricalLabels.bg, 0)
        pr_categorical_map_fg2 = tf.where(tf.logical_and(hm_pr >= self.theta_0, hm_pr < self.theta_1),
                                          CategoricalLabels.fg_2, 0)
        pr_categorical_map_fg1 = tf.where(hm_pr >= self.theta_1, CategoricalLabels.fg_1, 0)
        pr_categorical = pr_categorical_map_bg + pr_categorical_map_fg2 + pr_categorical_map_fg1
        '''on_hot'''
        pr_categorical = tf.one_hot(pr_categorical, depth=3)
        '''categorical loss'''
        cel_obj = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cat_loss_tensor = cel_obj(gt_categorical, pr_categorical, sample_weight=gt_categorical_weight)
        cat_loss_map = tf.cast(cat_loss_tensor > 0.0,
                               dtype=tf.float32)  # we use this as a 0-1 weight for regression
        loss_categorical = tf.math.reduce_mean(cat_loss_tensor)  # the categorical loss

        '''loss intensity'''
        delta_intensity = tf.math.abs(hm_gt - hm_pr)
        '''create high and low dif map'''
        high_dif_map = tf.where(delta_intensity >= threshold, 1.0, 0.0)
        low_dif_map = tf.where(delta_intensity < threshold, 1.0, 0.0)

        low_dif_map_fg_1 = tf.where(tf.logical_and(threshold_2 <= delta_intensity, delta_intensity < threshold), 1.0,
                                    0.0)
        fg_soft_low_dif_mapfg_1 = tf.where(delta_intensity < threshold_2, 1.0, 0.0)

        '''loss bg:'''
        loss_bg_low_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))

        loss_bg_high_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * high_dif_map * (tf.math.square(hm_gt - hm_pr) - 0.5 * threshold ** 2))

        loss_bg = (loss_bg_low_dif + loss_bg_high_dif)

        '''loss fg2'''
        loss_fg2_low_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))

        loss_fg2_high_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))

        loss_fg2 = (loss_fg2_low_dif + loss_fg2_high_dif)

        '''loss fg1'''
        '''we DONT multiply cat_loss_map in fg_1 region: since is is very important and and we don't stop'''

        '''main'''
        loss_fg1_low_dif_soft = tf.math.reduce_mean(
            weight_map_fg1 * fg_soft_low_dif_mapfg_1 * (tf.math.abs(hm_gt - hm_pr)))

        loss_fg1_low_dif = tf.math.reduce_mean(
            weight_map_fg1 * low_dif_map_fg_1 *
            (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2)))

        loss_fg1_high_dif = tf.math.reduce_mean(
            weight_map_fg1 * high_dif_map *
            (tf.math.square(hm_gt - hm_pr) +
             LearningConfig.Loss_fg_k * ln(threshold + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2) - threshold ** 2))

        loss_fg1 = (loss_fg1_low_dif_soft + loss_fg1_low_dif + loss_fg1_high_dif)

        return loss_bg, loss_fg2, loss_fg1, loss_categorical

    def hm_intensive_loss_stacked_single(self, hm_gt, hm_pr):
        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1

        # gt_categorical_map_bg = tf.where(hm_gt < self.theta_0, CategoricalLabels.bg, 0.0)
        # _categorical_map_fg2 = tf.where(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1), CategoricalLabels.fg_2, 0.0)
        # _categorical_map_fg1 = tf.where(hm_gt >= self.theta_1, CategoricalLabels.fg_1, 0.0)
        # gt_categorical_map = _categorical_map_fg2 + _categorical_map_fg1

        gt_categorical = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.fg_1 + \
                         weight_map_fg2 / self.omega_fg2 * CategoricalLabels.fg_2 + \
                         weight_map_bg / self.omega_bg * CategoricalLabels.bg
        gt_categorical = tf.cast(gt_categorical, dtype=tf.int32)
        gt_categorical = tf.one_hot(gt_categorical, depth=3)
        gt_categorical_weight = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.w_fg_1 + \
                                weight_map_fg2 / self.omega_fg2 * CategoricalLabels.w_fg_2 + \
                                weight_map_bg / self.omega_bg * CategoricalLabels.w_bg

        # ''''''
        threshold = LearningConfig.Loss_threshold
        threshold_2 = LearningConfig.Loss_threshold_2
        '''loss categorical'''
        pr_categorical_map_bg = tf.where(hm_pr < self.theta_0, CategoricalLabels.bg, 0)
        pr_categorical_map_fg2 = tf.where(tf.logical_and(hm_pr >= self.theta_0, hm_pr < self.theta_1),
                                          CategoricalLabels.fg_2, 0)
        pr_categorical_map_fg1 = tf.where(hm_pr >= self.theta_1, CategoricalLabels.fg_1, 0)
        pr_categorical = pr_categorical_map_bg + pr_categorical_map_fg2 + pr_categorical_map_fg1
        '''on_hot'''
        pr_categorical = tf.one_hot(pr_categorical, depth=3)
        '''categorical loss'''
        cel_obj = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cat_loss_tensor = cel_obj(gt_categorical, pr_categorical, sample_weight=gt_categorical_weight)
        cat_loss_map = tf.cast(cat_loss_tensor > 0.0,
                               dtype=tf.float32)  # we use this as a 0-1 weight for regression
        loss_categorical = tf.math.reduce_mean(cat_loss_tensor)  # the categorical loss

        '''loss intensity'''
        delta_intensity = tf.math.abs(hm_gt - hm_pr)
        '''create high and low dif map'''
        high_dif_map = tf.where(delta_intensity >= threshold, 1.0, 0.0)
        low_dif_map = tf.where(delta_intensity < threshold, 1.0, 0.0)

        low_dif_map_fg_1 = tf.where(tf.logical_and(threshold_2 <= delta_intensity, delta_intensity < threshold),
                                    1.0, 0.0)
        fg_soft_low_dif_mapfg_1 = tf.where(delta_intensity < threshold_2, 1.0, 0.0)

        '''loss bg:'''
        loss_bg_low_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))

        loss_bg_high_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * high_dif_map * (tf.math.square(hm_gt - hm_pr) - 0.5 * threshold ** 2))

        loss_bg = loss_bg_low_dif + loss_bg_high_dif

        '''loss fg2'''
        loss_fg2_low_dif = tf.math.reduce_mean(
            # weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))
            cat_loss_map * weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))

        loss_fg2_high_dif = tf.math.reduce_mean(
            # weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))
            cat_loss_map * weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))

        loss_fg2 = loss_fg2_low_dif + loss_fg2_high_dif

        '''loss fg1'''
        '''we DONT multiply cat_loss_map in fg_1 region: since is is very important and and we don't stop'''

        '''main'''
        loss_fg1_low_dif_soft = tf.math.reduce_mean(
            weight_map_fg1 * fg_soft_low_dif_mapfg_1 * (tf.math.abs(hm_gt - hm_pr)))

        loss_fg1_low_dif = tf.math.reduce_mean(
            weight_map_fg1 * low_dif_map_fg_1 *
            (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2)))

        loss_fg1_high_dif = tf.math.reduce_mean(
            weight_map_fg1 * high_dif_map *
            (tf.math.square(hm_gt - hm_pr) +
             LearningConfig.Loss_fg_k * ln(threshold + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2) - threshold ** 2))

        loss_fg1 = loss_fg1_low_dif_soft + loss_fg1_low_dif + loss_fg1_high_dif

        return loss_bg, loss_fg2, loss_fg1, loss_categorical

    def hm_intensive_loss_stacked(self, hm_gt, hm_prs):
        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1

        # gt_categorical_map_bg = tf.where(hm_gt < self.theta_0, CategoricalLabels.bg, 0.0)
        # _categorical_map_fg2 = tf.where(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1), CategoricalLabels.fg_2, 0.0)
        # _categorical_map_fg1 = tf.where(hm_gt >= self.theta_1, CategoricalLabels.fg_1, 0.0)
        # gt_categorical_map = _categorical_map_fg2 + _categorical_map_fg1

        gt_categorical = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.fg_1 + \
                         weight_map_fg2 / self.omega_fg2 * CategoricalLabels.fg_2 + \
                         weight_map_bg / self.omega_bg * CategoricalLabels.bg
        gt_categorical = tf.cast(gt_categorical, dtype=tf.int32)
        gt_categorical = tf.one_hot(gt_categorical, depth=3)
        gt_categorical_weight = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.w_fg_1 + \
                                weight_map_fg2 / self.omega_fg2 * CategoricalLabels.w_fg_2 + \
                                weight_map_bg / self.omega_bg * CategoricalLabels.w_bg

        # ''''''
        loss_bg = 0.0
        loss_fg2 = 0.0
        loss_fg1 = 0.0
        loss_categorical = 0.0

        threshold = LearningConfig.Loss_threshold
        threshold_2 = LearningConfig.Loss_threshold_2
        # stack_weight = [1.0, 1.0, 1.0, 3.0]  # two times more than sum of the previous layers
        stack_weight = [1.0, 1.0, 1.0, 1.0, 3.0]  # two times more than sum of the previous layers
        for i, hm_pr in enumerate(hm_prs):
            '''loss categorical'''
            pr_categorical_map_bg = tf.where(hm_pr < self.theta_0, CategoricalLabels.bg, 0)
            pr_categorical_map_fg2 = tf.where(tf.logical_and(hm_pr >= self.theta_0, hm_pr < self.theta_1),
                                              CategoricalLabels.fg_2, 0)
            pr_categorical_map_fg1 = tf.where(hm_pr >= self.theta_1, CategoricalLabels.fg_1, 0)
            pr_categorical = pr_categorical_map_bg + pr_categorical_map_fg2 + pr_categorical_map_fg1
            '''on_hot'''
            pr_categorical = tf.one_hot(pr_categorical, depth=3)
            '''categorical loss'''
            cel_obj = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            cat_loss_tensor = cel_obj(gt_categorical, pr_categorical, sample_weight=gt_categorical_weight)
            cat_loss_map = tf.cast(cat_loss_tensor > 0.0,
                                   dtype=tf.float32)  # we use this as a 0-1 weight for regression
            loss_categorical += stack_weight[i] * tf.math.reduce_mean(cat_loss_tensor)  # the categorical loss

            '''loss intensity'''
            delta_intensity = tf.math.abs(hm_gt - hm_pr)
            '''create high and low dif map'''
            high_dif_map = tf.where(delta_intensity >= threshold, 1.0, 0.0)
            low_dif_map = tf.where(delta_intensity < threshold, 1.0, 0.0)

            low_dif_map_fg_1 = tf.where(tf.logical_and(threshold_2 <= delta_intensity, delta_intensity < threshold),
                                        1.0, 0.0)
            fg_soft_low_dif_mapfg_1 = tf.where(delta_intensity < threshold_2, 1.0, 0.0)

            '''loss bg:'''
            loss_bg_low_dif = tf.math.reduce_mean(
                cat_loss_map * weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))

            loss_bg_high_dif = tf.math.reduce_mean(
                cat_loss_map * weight_map_bg * high_dif_map * (tf.math.square(hm_gt - hm_pr) - 0.5 * threshold ** 2))

            loss_bg += stack_weight[i] * (loss_bg_low_dif + loss_bg_high_dif)

            '''loss fg2'''
            loss_fg2_low_dif = tf.math.reduce_mean(
                cat_loss_map * weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))

            loss_fg2_high_dif = tf.math.reduce_mean(
                cat_loss_map * weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))

            loss_fg2 += stack_weight[i] * (loss_fg2_low_dif + loss_fg2_high_dif)

            '''loss fg1'''
            '''we DONT multiply cat_loss_map in fg_1 region: since is is very important and and we don't stop'''

            '''main'''
            loss_fg1_low_dif_soft = tf.math.reduce_mean(
                weight_map_fg1 * fg_soft_low_dif_mapfg_1 * (tf.math.abs(hm_gt - hm_pr)))

            loss_fg1_low_dif = tf.math.reduce_mean(
                weight_map_fg1 * low_dif_map_fg_1 *
                (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)
                 + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2)))

            loss_fg1_high_dif = tf.math.reduce_mean(
                weight_map_fg1 * high_dif_map *
                (tf.math.square(hm_gt - hm_pr) +
                 LearningConfig.Loss_fg_k * ln(threshold + 1)
                 + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2) - threshold ** 2))

            loss_fg1 = stack_weight[i] * (loss_fg1_low_dif_soft + loss_fg1_low_dif + loss_fg1_high_dif)

        return loss_bg, loss_fg2, loss_fg1, loss_categorical

    # some utilitis functions

    def _find_weighted_avg_of_n_biggest_points(self, heatmap, num_of_avg_points):
        weights, indices = self._top_n_indexes_tensor(heatmap, num_of_avg_points)

        x_indices = tf.cast(indices[:, 1], tf.float32)
        y_indices = tf.cast(indices[:, 0], tf.float32)
        '''weighted average over x and y'''
        w_avg_x = tf.scalar_mul(4 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(x_indices, weights)]))
        w_avg_y = tf.scalar_mul(4 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(y_indices, weights)]))
        #  now, w_avg_x is from 0, to 256, we need to normalize it to be [-0.5, +0.5]
        center = InputDataSize.image_input_size // 2
        w_avg_x_n = (w_avg_x - center) / InputDataSize.image_input_size
        w_avg_y_n = (w_avg_y - center) / InputDataSize.image_input_size
        return tf.stack([w_avg_x_n, w_avg_y_n])

    def _top_n_indexes_tensor(self, arr, n):
        shape = tf.shape(arr)
        top_values, top_indices = tf.nn.top_k(tf.reshape(arr, (-1,)), n)
        top_indices = tf.stack(((top_indices // shape[1]), (top_indices % shape[1])), -1)
        return top_values, top_indices

    def hm_intensive_loss_1d(self, hm_gt, hm_pr):
        # hm_pr = tf.convert_to_tensor(hm_pr)
        # hm_pr = tf.squeeze(hm_pr, axis=4)
        # hm_pr = tf.reshape(hm_pr, [tf.shape(hm_pr)[1],
        #                    tf.shape(hm_pr)[2],
        #                    tf.shape(hm_pr)[3],
        #                    tf.shape(hm_pr)[0]])

        weight_map_bg = tf.cast(hm_gt < self.theta_0, dtype=tf.float32) * self.omega_bg
        weight_map_fg2 = tf.cast(tf.logical_and(hm_gt >= self.theta_0, hm_gt < self.theta_1),
                                 dtype=tf.float32) * self.omega_fg2
        weight_map_fg1 = tf.cast(hm_gt >= self.theta_1, dtype=tf.float32) * self.omega_fg1

        gt_categorical = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.fg_1 + \
                         weight_map_fg2 / self.omega_fg2 * CategoricalLabels.fg_2 + \
                         weight_map_bg / self.omega_bg * CategoricalLabels.bg
        gt_categorical = tf.cast(gt_categorical, dtype=tf.int32)
        gt_categorical = tf.one_hot(gt_categorical, depth=3)
        gt_categorical_weight = weight_map_fg1 / self.omega_fg1 * CategoricalLabels.w_fg_1 + \
                                weight_map_fg2 / self.omega_fg2 * CategoricalLabels.w_fg_2 + \
                                weight_map_bg / self.omega_bg * CategoricalLabels.w_bg

        # ''''''
        threshold = LearningConfig.Loss_threshold
        threshold_2 = LearningConfig.Loss_threshold_2
        '''loss categorical'''
        pr_categorical_map_bg = tf.where(hm_pr < self.theta_0, CategoricalLabels.bg, 0)
        pr_categorical_map_fg2 = tf.where(tf.logical_and(hm_pr >= self.theta_0, hm_pr < self.theta_1),
                                          CategoricalLabels.fg_2, 0)
        pr_categorical_map_fg1 = tf.where(hm_pr >= self.theta_1, CategoricalLabels.fg_1, 0)
        pr_categorical = pr_categorical_map_bg + pr_categorical_map_fg2 + pr_categorical_map_fg1
        '''on_hot'''
        pr_categorical = tf.one_hot(pr_categorical, depth=3)
        '''categorical loss'''
        cel_obj = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cat_loss_tensor = cel_obj(gt_categorical, pr_categorical, sample_weight=gt_categorical_weight)
        cat_loss_map = tf.cast(cat_loss_tensor > 0.0,
                               dtype=tf.float32)  # we use this as a 0-1 weight for regression
        loss_categorical = tf.math.reduce_mean(cat_loss_tensor)  # the categorical loss

        '''loss intensity'''
        delta_intensity = tf.math.abs(hm_gt - hm_pr)
        '''create high and low dif map'''
        high_dif_map = tf.where(delta_intensity >= threshold, 1.0, 0.0)
        low_dif_map = tf.where(delta_intensity < threshold, 1.0, 0.0)

        low_dif_map_fg_1 = tf.where(tf.logical_and(threshold_2 <= delta_intensity, delta_intensity < threshold),
                                    1.0, 0.0)
        fg_soft_low_dif_mapfg_1 = tf.where(delta_intensity < threshold_2, 1.0, 0.0)

        '''loss bg:'''
        loss_bg_low_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * low_dif_map * 0.5 * tf.math.square(hm_gt - hm_pr))

        loss_bg_high_dif = tf.math.reduce_mean(
            cat_loss_map * weight_map_bg * high_dif_map * (tf.math.square(hm_gt - hm_pr) - 0.5 * threshold ** 2))

        loss_bg = loss_bg_low_dif + loss_bg_high_dif

        '''loss fg2'''
        loss_fg2_low_dif = tf.math.reduce_mean(
            weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))
        # cat_loss_map * weight_map_fg2 * low_dif_map * tf.math.abs(hm_gt - hm_pr))

        loss_fg2_high_dif = tf.math.reduce_mean(
            weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))
        # cat_loss_map * weight_map_fg2 * high_dif_map * (tf.math.square(hm_gt - hm_pr) + threshold ** 2))

        loss_fg2 = loss_fg2_low_dif + loss_fg2_high_dif

        '''loss fg1'''
        '''we DONT multiply cat_loss_map in fg_1 region: since is is very important and and we don't stop'''

        '''main'''
        loss_fg1_low_dif_soft = tf.math.reduce_mean(
            weight_map_fg1 * fg_soft_low_dif_mapfg_1 * (tf.math.abs(hm_gt - hm_pr)))

        loss_fg1_low_dif = tf.math.reduce_mean(
            weight_map_fg1 * low_dif_map_fg_1 *
            (LearningConfig.Loss_fg_k * tf.math.log(tf.math.abs(hm_gt - hm_pr) + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2)))

        loss_fg1_high_dif = tf.math.reduce_mean(
            weight_map_fg1 * high_dif_map *
            (tf.math.square(hm_gt - hm_pr) +
             LearningConfig.Loss_fg_k * ln(threshold + 1)
             + threshold_2 - LearningConfig.Loss_fg_k * ln(1 + threshold_2) - threshold ** 2))

        loss_fg1 = loss_fg1_low_dif_soft + loss_fg1_low_dif + loss_fg1_high_dif

        '''graph model loss:'''

        return loss_bg, loss_fg2, loss_fg1, loss_categorical
