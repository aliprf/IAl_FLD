from configuration import DatasetName, DatasetType, D300WConf, InputDataSize, CofwConf, WflwConf, LearningConfig
from custom_loss import CustomLoss
from cnn_model import CNNModel
from data_helper import DataHelper

import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
from skimage.io import imread


class Train:
    def __init__(self, dataset_name, use_augmented):
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.ds_300W:
            self.num_landmark = D300WConf.num_of_landmarks * 2
            if use_augmented:
                self.img_path = D300WConf.augmented_train_image
                self.annotation_path = D300WConf.augmented_train_annotation
                self.hm_path = D300WConf.augmented_train_hm
            else:
                self.img_path = D300WConf.no_aug_train_image
                self.annotation_path = D300WConf.no_aug_train_annotation
                self.hm_path = D300WConf.no_aug_train_hm

        if dataset_name == DatasetName.ds_cofw:
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.img_path = CofwConf.augmented_train_image
            self.annotation_path = CofwConf.augmented_train_annotation
            self.hm_path = CofwConf.augmented_train_hm

        if dataset_name == DatasetName.ds_wflw:
            self.num_landmark = WflwConf.num_of_landmarks * 2
            if use_augmented:
                self.img_path = WflwConf.augmented_train_image
                self.annotation_path = WflwConf.augmented_train_annotation
                self.hm_path = WflwConf.augmented_train_hm
            else:
                self.img_path = WflwConf.no_aug_train_image
                self.annotation_path = WflwConf.no_aug_train_annotation
                self.hm_path = WflwConf.no_aug_train_hm

    # @tf.function
    def train(self, arch, weight_path):
        """"""
        '''create loss'''
        c_loss = CustomLoss(dataset_name=self.dataset_name, theta_0=0.5, theta_1=0.9, omega_bg=1, omega_fg2=2e+1,
                            omega_fg1=2e+2)

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, num_landmark=self.num_landmark)
        if weight_path is not None:
            model.load_weights(weight_path)

        '''LearningRate'''
        _lr = 1e-3
        '''create optimizer'''
        optimizer = self._get_optimizer(lr=_lr)

        '''create sample generator'''
        img_train_filenames, img_val_filenames, hm_train_filenames, hm_val_filenames = self._create_generators()
        '''create train configuration'''
        step_per_epoch = len(img_train_filenames) // LearningConfig.batch_size

        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            img_train_filenames, hm_train_filenames = self._shuffle_data(img_train_filenames, hm_train_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, hm_gt, anno_gt = self._get_batch_sample(batch_index=batch_index,
                                                                img_train_filenames=img_train_filenames,
                                                                hm_train_filenames=hm_train_filenames)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                anno_gt = tf.cast(anno_gt, tf.float32)
                hm_gt = tf.cast(hm_gt, tf.float32)
                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images,
                                model=model, hm_gt=hm_gt, anno_gt=anno_gt, optimizer=optimizer,
                                summary_writer=summary_writer, c_loss=c_loss)

            '''evaluating part'''
            loss_eval = 0
            # img_batch_eval, hm_batch_eval, pn_batch_eval = self._create_evaluation_batch(img_val_filenames,
            #                                                                              hm_val_filenames)
            # loss_eval = self._eval_model(img_batch_eval, hm_batch_eval, model)
            # with summary_writer.as_default():
            #     tf.summary.scalar('Eval-LOSS', loss_eval, step=epoch)
            '''save weights'''
            model.save('./models/IAL' + str(epoch) + '_' + self.dataset_name + '_' + str(loss_eval) + '.h5')
            # model.save_weights(
            #     './models/asm_fw_weight_' + '_' + str(epoch) + self.dataset_name + '_' + str(loss_eval) + '.h5')
            '''calculate Learning rate'''
            # _lr = self._calc_learning_rate(iterations=epoch, step_size=10, base_lr=1e-5, max_lr=1e-2)
            # optimizer = self._get_optimizer(lr=_lr)

    def train_step(self, epoch, step, total_steps, images, model, hm_gt, anno_gt, optimizer, summary_writer, c_loss):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            hm_prs = model(images, training=True)

            # out_pnt_0 = self._convert_hm_to_pts(hm_prs[0])
            # out_pnt_1 = self._convert_hm_to_pts(hm_prs[1])
            # out_pnt_2 = self._convert_hm_to_pts(hm_prs[2])
            # out_pnt_3 = self._convert_hm_to_pts(hm_prs[3])  # the last is the most important

            '''calculate loss'''
            # loss_total, loss_bg, loss_fg2, loss_fg1, loss_reg = c_loss.intensive_aware_loss(hm_gt=hm_gt,
            loss_total, loss_bg, loss_fg2, loss_fg1, loss_categorical = c_loss.intensive_aware_loss(hm_gt=hm_gt,
                                                                                                    hm_prs=hm_prs,
                                                                                                    anno_gt=anno_gt,
                                                                                                    anno_prs=None,
                                                                                                    # anno_prs=[out_pnt_0, out_pnt_1,
                                                                                                    #          out_pnt_2, out_pnt_3],
                                                                                                    )
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps), ' -> : LOSS: ', loss_total,
                 ' -> : loss_fg1: ', loss_fg1, ' -> : loss_fg2: ', loss_fg2, ' -> : loss_bg: ',
                 loss_bg, ' -> : loss_categorical: ', loss_categorical)
        # print('==--==--==--==--==--==--==--==--==--')
        with summary_writer.as_default():
            tf.summary.scalar('LOSS', loss_total, step=epoch)
            tf.summary.scalar('loss_fg1', loss_fg1, step=epoch)
            tf.summary.scalar('loss_fg2', loss_fg2, step=epoch)
            tf.summary.scalar('loss_bg', loss_bg, step=epoch)
            tf.summary.scalar('loss_categorical', loss_categorical, step=epoch)
            # tf.summary.scalar('loss_reg', loss_reg, step=epoch)

    def _calc_learning_rate(self, iterations, step_size, base_lr, max_lr, gamma=0.99994):
        '''reducing triangle'''
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        '''exp'''
        # cycle = np.floor(1 + iterations / (2 * step_size))
        # x = np.abs(iterations / step_size - 2 * cycle + 1)
        # lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** (iterations)

        print('LR is: ' + str(lr))
        return lr

    def _eval_model(self, img_batch_eval, hm_batch_eval, model):
        hm_pr = model(img_batch_eval)
        los_eval = np.array(tf.reduce_mean(tf.abs(hm_batch_eval - hm_pr)))
        return los_eval

    def _get_optimizer(self, lr=1e-1, beta_1=0.9, beta_2=0.999, decay=1e-5):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _create_generators(self):
        dlp = DataHelper()
        filenames, hm_labels = dlp.create_image_and_labels_name(img_path=self.img_path, hm_path=self.hm_path)

        filenames_shuffled, y_labels_shuffled = shuffle(filenames, hm_labels)
        img_train_filenames, img_val_filenames, hm_train, hm_val = train_test_split(
            filenames_shuffled, y_labels_shuffled, test_size=LearningConfig.batch_size, random_state=1)

        return img_train_filenames, img_val_filenames, hm_train, hm_val

    def _shuffle_data(self, filenames, labels):
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def _get_batch_sample(self, batch_index, img_train_filenames, hm_train_filenames):
        dhl = DataHelper()

        '''create batch data and normalize images'''
        batch_x = img_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = hm_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([imread(self.img_path + file_name) for file_name in batch_x]) / 255.0
        hm_batch = np.array([load(self.hm_path + file_name) for file_name in batch_y])

        '''in this method, we dont normalize the points'''
        pn_batch = np.array([load(self.annotation_path + file_name) for file_name in batch_y])
        # if self.dataset_name == DatasetName.ds_cofw:
        #     pn_batch = np.array([load(self.annotation_path + file_name) for file_name in batch_y])
        # else:
        #     pn_batch = np.array([dhl.load_and_normalize(self.annotation_path + file_name) for file_name in batch_y])

        # '''test: print'''
        # image_utility = ImageUtility()
        # for i in range(LearningConfig.batch_size):
        #     gr_s, gr_px_1, gr_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch[i], 224, 224, 112, 112)
        #     imgpr.print_image_arr(str(batch_index + 1 * (i + 1)) + 'pts_gt', img_batch[i], gr_px_1, gr_Py_1)
        #
        #     for ac in [80, 85, 90, 95, 97]:
        #         pn_batch_asm = np.array([tf_utils.get_asm(input=self._load_and_normalize(pn_tr_path + file_name),
        #                                                   dataset_name=self.dataset_name, accuracy=ac)
        #                                  for file_name in batch_y])
        #
        #         asm_p_s, asm_px_1, asm_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch_asm[i], 224, 224,
        #                                                                                      112,
        #                                                                                      112)
        #         imgpr.print_image_arr(str(batch_index + 1 * (i + 1)) + 'pts_asm' + str(ac), img_batch[i], asm_px_1,
        #                               asm_Py_1)

        return img_batch, hm_batch, pn_batch

    def _create_evaluation_batch(self, img_eval_filenames, hm_eval_filenames):
        dhl = DataHelper()
        batch_x = img_eval_filenames[0:LearningConfig.batch_size]
        batch_y = hm_eval_filenames[0:LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([imread(self.img_path + file_name) for file_name in batch_x]) / 255.0
        hm_batch = np.array([load(self.hm_path + file_name) for file_name in batch_y])

        if self.dataset_name == DatasetName.ds_cofw:
            pn_batch = np.array([load(self.annotation_path + file_name) for file_name in batch_y])
        else:
            pn_batch = np.array([dhl.load_and_normalize(self.annotation_path + file_name) for file_name in batch_y])

        return img_batch, hm_batch, pn_batch

    @tf.autograph.experimental.do_not_convert
    def _convert_hm_to_pts(self, hm):
        x_center = InputDataSize.image_input_size // 2
        width = InputDataSize.image_input_size
        hm_arr = []

        for i in range(LearningConfig.batch_size):
            hm_t = self._from_heatmap_to_point_tensor(heatmaps=hm[i], number_of_points=3)
            # hm_t = tf.reshape(tensor=hm_t, shape=self.num_landmark*2)
            # '''hm is in [0,224] --> should be in [-0.5,+0.5]'''
            # hm_t_norm = tf.math.scalar_mul(scalar=1 / width,
            #                                x=tf.math.subtract(hm_t, np.repeat(x_center, self.num_landmark*2)))
            # hm_arr.append(hm_t_norm)
            hm_arr.append(hm_t)
        '''reshape hm'''
        hm_pts = tf.stack([hm_arr[i] for i in range(LearningConfig.batch_size)], 0)  # bs * self.num_landmark
        return hm_pts

    def _from_heatmap_to_point_tensor(self, heatmaps, number_of_points, scalar=4):
        x = tf.stack([self._find_nth_biggest_avg_tensor(heatmaps[:, :, i], number_of_points, scalar)
                      for i in range(self.num_landmark)], -1)
        return x

    def _find_nth_biggest_avg_tensor(self, heatmap, points, scalar):
        weights, indices = self._top_n_indexes_tensor(heatmap, points)

        x_indices = tf.cast(indices[:, 0], tf.float32)
        y_indices = tf.cast(indices[:, 1], tf.float32)
        '''weighted average over x and y'''
        w_avg_x = tf.scalar_mul(1 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(x_indices, weights)]))

        # if you wanna create regression face which is 56*56 except for 224*224
        # w_avg_x = tf.scalar_mul(1 / InputDataSize.hm_size, w_avg_x)

        w_avg_y = tf.scalar_mul(1 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(y_indices, weights)]))
        # w_avg_y = tf.scalar_mul(1 / InputDataSize.hm_size, w_avg_y)

        return tf.stack([w_avg_x, w_avg_y])

    def _top_n_indexes_tensor(self, arr, n):
        shape = tf.shape(arr)
        # first convert it to 1-d
        top_values, top_indices = tf.nn.top_k(tf.reshape(arr, (-1,)), n)
        # convert again to 2d
        top_indices = tf.stack(((top_indices // shape[1]), (top_indices % shape[1])), -1)
        return top_values, top_indices
