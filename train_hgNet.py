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
from PIL import Image
from tqdm import tqdm


class TrainHg:
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
            '''evaluation path:'''
            self.eval_img_path = D300WConf.test_image_path + 'challenging/'
            self.eval_annotation_path = D300WConf.test_annotation_path + 'challenging/'

        if dataset_name == DatasetName.ds_cofw:
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.img_path = CofwConf.augmented_train_image
            self.annotation_path = CofwConf.augmented_train_annotation
            self.hm_path = CofwConf.augmented_train_hm
            '''evaluation path:'''
            self.eval_img_path = D300WConf.test_image_path
            self.eval_annotation_path = D300WConf.test_annotation_path

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
            '''evaluation path:'''
            self.eval_img_path = D300WConf.test_image_path + 'pose/'
            self.eval_annotation_path = D300WConf.test_annotation_path + 'pose/'

    # @tf.function
    def train(self, arch, weight_path):
        """"""
        '''create loss'''
        c_loss = CustomLoss(dataset_name=self.dataset_name, theta_0=0.6, theta_1=0.9, omega_bg=1, omega_fg2=50,
                            omega_fg1=100, number_of_landmark=self.num_landmark)

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, num_landmark=self.num_landmark)
        if weight_path is not None:
            model.load_weights(weight_path)

        '''LearningRate'''
        _lr = 1e-4
        '''create optimizer'''
        optimizer = self._get_optimizer(lr=_lr, decay=1e-6)

        '''create sample generator'''
        # img_train_filenames, img_val_filenames, hm_train_filenames, hm_val_filenames = self._create_generators()
        img_train_filenames, hm_train_filenames = self._create_generators()
        img_val_filenames, pn_val_filenames = self._create_generators(img_path=self.eval_img_path,
                                                                      hm_path=self.eval_annotation_path)

        #
        nme, fr = self._eval_model(model, img_val_filenames, pn_val_filenames)
        print(nme)
        print(fr)
        '''create train configuration'''
        step_per_epoch = len(img_train_filenames) // LearningConfig.batch_size

        for epoch in range(LearningConfig.epochs):
            img_train_filenames, hm_train_filenames = self._shuffle_data(img_train_filenames, hm_train_filenames)
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, hm_gt, anno_gt = self._get_batch_sample(batch_index=batch_index,
                                                                img_train_filenames=img_train_filenames,
                                                                hm_train_filenames=hm_train_filenames)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                hm_gt = tf.cast(hm_gt, tf.float32)
                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images,
                                model=model, hm_gt=hm_gt, anno_gt=anno_gt, optimizer=optimizer,
                                summary_writer=summary_writer, c_loss=c_loss)

            '''evaluation part'''
            nme, fr = self._eval_model(model, img_val_filenames, pn_val_filenames)
            with summary_writer.as_default():
                tf.summary.scalar('Eval-nme', nme, step=epoch)
                tf.summary.scalar('Eval-fr', fr, step=epoch)

            '''save weights'''
            model.save('./models/IAL' + str(epoch) + '_' + self.dataset_name + '_nme_' + str(nme)
                       + '_fr_' + str(fr) + '.h5')

            '''calculate Learning rate'''
            # _lr = self._calc_learning_rate(iterations=epoch, step_size=10, base_lr=1e-5, max_lr=1e-2)
            # optimizer = self._get_optimizer(lr=_lr)

    def train_step(self, epoch, step, total_steps, images, model, hm_gt, anno_gt, optimizer, summary_writer, c_loss):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            hm_prs = model(images, training=True)

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

    def _eval_model(self, model, img_val_filenames, hm_val_filenames):
        dhl = DataHelper()
        # img_batch_eval, hm_batch_eval, pn_batch_eval = self._create_evaluation_batch(img_val_filenames,
        #                                                                              hm_val_filenames)
        nme_sum = 0
        fail_counter_sum = 0
        batch_size = 5  # LearningConfig.batch_size
        step_per_epoch = int(len(img_val_filenames) // (batch_size))
        for batch_index in tqdm(range(step_per_epoch)):
            images, hm_gts, anno_gts = self._get_batch_sample(batch_index=batch_index,
                                                              img_train_filenames=img_val_filenames,
                                                              hm_train_filenames=hm_val_filenames, is_eval=True,
                                                              batch_size=batch_size)
            '''predict:'''
            hm_prs = model.predict_on_batch(images)  # hm_pr: 4, bs, 64, 64, 68
            hm_prs_last_channel = np.array(hm_prs)[3, :, :, :]  # bs, 64, 64, 68
            '''calculate NME for batch'''
            bath_nme, bath_fr = dhl.calc_NME_over_batch(anno_GTs=anno_gts, pr_hms=hm_prs_last_channel,
                                                        ds_name=self.dataset_name)
            nme_sum += bath_nme
            fail_counter_sum += bath_fr

        '''calculate total'''
        fr = 100 * fail_counter_sum / len(img_val_filenames)
        nme = 100 * nme_sum / len(img_val_filenames)
        print(nme)
        print(fr)
        return nme, fr

    def _get_optimizer(self, lr=1e-1, beta_1=0.9, beta_2=0.999, decay=1e-4):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _create_generators(self, img_path=None, hm_path=None):
        dlp = DataHelper()
        if img_path is None:
            filenames, hm_labels = dlp.create_image_and_labels_name(img_path=self.img_path, hm_path=self.hm_path)
        else:
            filenames, hm_labels = dlp.create_image_and_labels_name(img_path=img_path, hm_path=hm_path)

        img_train_filenames, hm_train = shuffle(filenames, hm_labels)
        # filenames_shuffled, y_labels_shuffled = shuffle(filenames, hm_labels)
        # img_train_filenames, img_val_filenames, hm_train, hm_val = train_test_split(
        #     filenames_shuffled, y_labels_shuffled, test_size=LearningConfig.batch_size, random_state=1)

        # return img_train_filenames, img_val_filenames, hm_train, hm_val
        return img_train_filenames, hm_train

    def _shuffle_data(self, filenames, labels):
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        return filenames_shuffled, y_labels_shuffled

    def _get_batch_sample(self, batch_index, img_train_filenames, hm_train_filenames, is_eval=False, batch_size=None):
        """"""
        '''create batch data and normalize images'''
        '''create img and annotations'''
        if is_eval:
            batch_x = img_train_filenames[
                      batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_y = hm_train_filenames[
                      batch_index * batch_size:(batch_index + 1) * batch_size]

            img_batch = np.array([imread(self.eval_img_path + file_name) for file_name in batch_x]) / 255.0
            hm_batch = None  # np.array([load(self.eval_annotation_path + file_name) for file_name in batch_y])
            pn_batch = np.array([load(self.eval_annotation_path + file_name) for file_name in
                                 batch_y])  # for evaluation, we don't need normalized ground truth points

        else:
            batch_x = img_train_filenames[
                      batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
            batch_y = hm_train_filenames[
                      batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]

            img_batch = np.array([imread(self.img_path + file_name) for file_name in batch_x]) / 255.0
            hm_batch = np.array([load(self.hm_path + file_name) for file_name in batch_y])
            pn_batch = None  # np.array([dhl.load_and_normalize(self.annotation_path + file_name) for file_name in batch_y])

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
