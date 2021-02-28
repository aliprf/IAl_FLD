from configuration import InputDataSize, DatasetName
import os
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import colors
from numpy import log as ln
import math


class DataHelper:
    def create_image_and_labels_name(self, img_path, hm_path):
        img_filenames = []
        lbls_filenames = []

        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                lbl_file = str(file)[:-3] + "npy"  # just name
                if os.path.exists(hm_path + lbl_file):
                    img_filenames.append(str(file))
                    lbls_filenames.append(lbl_file)

        return np.array(img_filenames), np.array(lbls_filenames)

    def load_and_normalize(self, point_path):
        annotation = load(point_path)

        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((annotation[p] - x_center) / width)
            annotation_norm.append((annotation[p + 1] - y_center) / height)
        return annotation_norm

    def depict_loss(self, theta_0, theta_1):
        """we create the loss using the loss and |y-y`|"""
        dela_intensity_values = np.linspace(-3.0, 3.0, 1000)

        loss_val_bg = np.zeros_like(dela_intensity_values)
        der_loss_val_bg = np.zeros_like(dela_intensity_values)
        loss_val_fg_2 = np.zeros_like(dela_intensity_values)
        der_loss_val_fg_2 = np.zeros_like(dela_intensity_values)
        loss_val_fg_1 = np.zeros_like(dela_intensity_values)
        der_loss_val_fg_1 = np.zeros_like(dela_intensity_values)

        threshol = 0.5
        for i in range(len(dela_intensity_values)):
            if -threshol <= dela_intensity_values[i] <= threshol:
                '''bf loss:'''
                loss_val_bg[i] = 0.5 * dela_intensity_values[i] ** 2  # y = 0.5 x^2
                der_loss_val_bg[i] = abs(dela_intensity_values[i])  # y` = x
                '''fg_2'''
                loss_val_fg_2[i] = np.abs(dela_intensity_values[i])  # y = x
                der_loss_val_fg_2[i] = 1  # * abs(dela_intensity_values[i])  # y` = 4x
                '''fg_1'''
                loss_val_fg_1[i] = 10 * ln(abs(dela_intensity_values[i]) + 1)  # y = 10 ln(|x|+1)
                der_loss_val_fg_1[i] = 10 / (abs(dela_intensity_values[i]) + 1)  # y` = 10/(|x|+1)

            else:
                '''bf loss:'''
                loss_val_bg[i] = np.square(dela_intensity_values[i]) - 0.5 * threshol ** 2  # y = x^2
                der_loss_val_bg[i] = 2 * abs(dela_intensity_values[i])  # y` = 2x
                '''fg_2'''
                loss_val_fg_2[i] = np.square(dela_intensity_values[i])+threshol**2    # y = x^2
                der_loss_val_fg_2[i] = 2 * abs(dela_intensity_values[i])  # y` = 2x  # y` = 1
                '''fg_1'''
                loss_val_fg_1[i] = np.square(dela_intensity_values[i]) + 10 * ln(1 + threshol) - threshol ** 2  # y = x^2+ln(2)
                der_loss_val_fg_1[i] = 2 * abs(dela_intensity_values[i])  # y` = 2x

        '''print loss values:'''
        dpi = 80
        width = 3 * 700
        height = 2 * 700
        figsize = width / float(dpi), height / float(dpi)
        fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=figsize,
                                gridspec_kw={'width_ratios': [1, 1, 1]})

        for i in range(2):
            for j in range(3):
                axs[i, j].set_xlim(-2.5, 2.5)
                axs[i, j].set_ylim(-0.5, 4.5)
                axs[i, j].xaxis.set_minor_locator(AutoMinorLocator(4))
                axs[i, j].yaxis.set_minor_locator(AutoMinorLocator(4))
                axs[i, j].grid(which='major', color='#968c83', linestyle='--', linewidth=0.7)
                axs[i, j].grid(which='minor', color='#9ba4b4', linestyle=':', linewidth=0.5)
                # axs[i,j].text(-0.2, -0.2, r'|y-y`|')

        axs[0, 2].set_xlim(-3.2, 3.2)
        axs[0, 2].set_ylim(-0.2, 10.2)

        axs[1, 2].set_xlim(-3.2, 3.2)
        axs[1, 2].set_ylim(-0.2, 10.2)

        fig_bg_loss, = axs[0, 0].plot(dela_intensity_values[:], loss_val_bg[:], '#09015f', linewidth=4.0,
                                      label='bg Region Loss', alpha=1.0)
        fig_d_bg_loss, = axs[1, 0].plot(dela_intensity_values[:], der_loss_val_bg[:], '#654062', linewidth=3.0,
                                        label='Derivative of bg Region Loss', alpha=1.0)

        fig_fg2_loss, = axs[0, 1].plot(dela_intensity_values[:], loss_val_fg_2[:], '#ff577f', linewidth=4.0,
                                       label='fg-2 Region Loss', alpha=1.0)
        fig_d_fg2_loss, = axs[1, 1].plot(dela_intensity_values[:], der_loss_val_fg_2[:], '#ff884b', linewidth=3.0,
                                         label='Derivative of fg-2 Region Loss', alpha=1.0)

        fig_fg1_loss, = axs[0, 2].plot(dela_intensity_values[:], loss_val_fg_1[:], '#295939', linewidth=4.0,
                                       label='fg-1 Region Loss', alpha=1.0)
        fig_d_fg1_loss, = axs[1, 2].plot(dela_intensity_values[:], der_loss_val_fg_1[:], '#83a95c', linewidth=3.0,
                                         label='Derivative of fg-1 Region Loss', alpha=1.0)
        plt.tight_layout()
        plt.savefig('loss.png', bbox_inches='tight')

    def depict_weight_map_function(self, theta_0, theta_1):
        """create both loss and weightmap"""
        '''create sample heatmap'''
        sample_hm = self._create_single_hm(width=56, height=56, x0=23, y0=23, sigma=7)
        '''create weight'''
        weight_map = np.zeros_like(sample_hm)
        weight_map[sample_hm < theta_0] = 1
        weight_map[np.where(np.logical_and(sample_hm >= theta_0, sample_hm < theta_1))] = 5
        weight_map[sample_hm >= theta_1] = 15

        '''depict weight'''
        dpi = 80
        width = 1400
        height = 1400
        figsize = width / float(dpi), height / float(dpi)
        fig_1 = plt.figure(figsize=figsize)
        ax = fig_1.gca(projection='3d')
        x = np.linspace(0, 56, 56)
        y = np.linspace(0, 56, 56)
        X, Y = np.meshgrid(x, y)

        cmap = colors.ListedColormap(['#9088d4', '#cbbcb1', '#16697a', '#cbbcb1', '#a20a0a'])
        boundaries = [0, 1.01, 5, 5.1, 14.99, 15]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

        surf = ax.plot_surface(X, Y, weight_map, alpha=0.8, linewidth=1.5, antialiased=True, zorder=0.1,
                               vmin=0, vmax=15, rstride=1, cstride=1, cmap=cmap, norm=norm)
        # cmap=cm.coolwarm)
        ax.set_zlim(-1.0, 15.1)
        ax.grid(True)
        ax.zaxis.set_major_locator(LinearLocator(15))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        fig_1.colorbar(surf, shrink=0.2, aspect=15)
        plt.savefig('./weight_map_.png', bbox_inches='tight')

        # for i in range(len(sample_hm.shape[0])):
        #     for j in range(len(sample_hm.shape[1])):
        #         intensity_xy = sample_hm[i, j]
        #         if 0 <= intensity_xy < theta_0:
        #             # bg:
        #
        #         elif theta_0 <= intensity_xy < theta_1:
        #             # fg_2:
        #
        #         elif theta_0 <= intensity_xy =< 1:
        #             #fg_1:

        '''depict loss:'''

    def _create_single_hm(self, width, height, x0, y0, sigma):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        gaus = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        gaus[gaus <= 0.01] = 0
        return gaus

    def calc_NME_over_batch(self, anno_GTs, pr_hms, ds_name):
        sum_nme = 0
        fail_counter = 0
        fr_threshold = 0.1
        for i in range(pr_hms.shape[0]):
            pr_hm = pr_hms[i, :, :, :]
            _, _, anno_Pre = self._hm_to_points(heatmaps=pr_hm)
            anno_GT = anno_GTs[i]
            nme_i, norm_error = self._calculate_nme(anno_GT=anno_GT, anno_Pre=anno_Pre, ds_name=ds_name,
                                                    ds_number_of_points=pr_hm.shape[2]*2)
            sum_nme += nme_i
            if nme_i > fr_threshold:
                fail_counter += 1
        return sum_nme, fail_counter

    def _calculate_nme(self, anno_GT, anno_Pre, ds_name, ds_number_of_points):
        normalizing_distance = self.calculate_interoccular_distance(anno_GT=anno_GT, ds_name=ds_name)
        '''here we round all data if needed'''
        sum_errors = 0
        errors_arr = []
        for i in range(0, len(anno_Pre), 2):  # two step each time
            x_pr = anno_Pre[i]
            y_pr = anno_Pre[i + 1]
            x_gt = anno_GT[i]
            y_gt = anno_GT[i + 1]
            error = math.sqrt(((x_pr - x_gt) ** 2) + ((y_pr - y_gt) ** 2))

            manhattan_error_x = abs(x_pr - x_gt) / 224.0
            manhattan_error_y = abs(y_pr - y_gt) / 224.0

            sum_errors += error
            errors_arr.append(manhattan_error_x)
            errors_arr.append(manhattan_error_y)

        NME = sum_errors / (normalizing_distance * ds_number_of_points)
        norm_error = errors_arr
        return NME, norm_error

    def calculate_interoccular_distance(self, anno_GT, ds_name):
        if ds_name == DatasetName.ds_300W:
            left_oc_x = anno_GT[72]
            left_oc_y = anno_GT[73]
            right_oc_x = anno_GT[90]
            right_oc_y = anno_GT[91]
        elif ds_name == DatasetName.ds_cofw:
            left_oc_x = anno_GT[16]
            left_oc_y = anno_GT[17]
            right_oc_x = anno_GT[18]
            right_oc_y = anno_GT[19]
        elif ds_name == DatasetName.ds_wflw:
            left_oc_x = anno_GT[192]
            left_oc_y = anno_GT[193]
            right_oc_x = anno_GT[194]
            right_oc_y = anno_GT[195]

        distance = math.sqrt(((left_oc_x - right_oc_x) ** 2) + ((left_oc_y - right_oc_y) ** 2))
        return distance

    def _hm_to_points(self, heatmaps):
        x_points = []
        y_points = []
        xy_points = []
        # print(heatmaps.shape) 56,56,68
        for i in range(heatmaps.shape[2]):
            x, y = self._find_nth_biggest_avg(heatmaps[:, :, i], number_of_selected_points=5,
                                              scalar=4.0)
            x_points.append(x)
            y_points.append(y)
            xy_points.append(x)
            xy_points.append(y)
        return np.array(x_points), np.array(y_points), np.array(xy_points)

    def _find_nth_biggest_avg(self, heatmap, number_of_selected_points, scalar):
        indices = self._top_n_indexes(heatmap, number_of_selected_points)

        x_arr = []
        y_arr = []
        w_arr = []
        x_s = 0.0
        y_s = 0.0
        w_s = 0.0

        for index in indices:
            x_arr.append(index[0])
            y_arr.append(index[1])
            w_arr.append(heatmap[index[0], index[1]])
        #
        for i in range(len(x_arr)):
            x_s += w_arr[i]*x_arr[i]
            y_s += w_arr[i]*y_arr[i]
            w_s += w_arr[i]
        x = (x_s * scalar)/w_s
        y = (y_s * scalar)/w_s

        # x = (0.75 * x_arr[1] + 0.25 * x_arr[0]) * scalar
        # y = (0.75 * y_arr[1] + 0.25 * y_arr[0]) * scalar

        return y, x
    def _top_n_indexes(self, arr, n):
        import bottleneck as bn
        idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
        width = arr.shape[1]
        xxx = [divmod(i, width) for i in idx]
        # result = np.where(arr == np.amax(arr))
        # return result
        return xxx