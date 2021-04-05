class DatasetName:
    ds_300W = '300W'
    ds_cofw = 'cofw'
    ds_wflw = 'wflw'


class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2


class LearningConfig:
    Loss_fg_k = 10
    Loss_threshold = 0.5
    Loss_threshold_2 = 0.02

    virtual_batch_size = 250

    batch_size = 3
    # batch_size = 50
    epochs = 1500


class CategoricalLabels:
    bg = 0
    fg_2 = 1
    fg_1 = 2
    # max_2 = 4
    # max_1 = 5

    w_bg = 1
    w_fg_2 = 5
    w_fg_1 = 5
    # w_max_2 = 100
    # w_max_1 = 100

class InputDataSize:
    image_input_size = 256
    img_center = image_input_size // 2

    hm_size = image_input_size // 4
    hm_center = hm_size // 2


class WflwConf:
    # Wflw_prefix_path = '/media/data3/ali/FL/new_data/wflw/'  # --> Zeus
    Wflw_prefix_path = '/media/data2/alip/FL/new_data/wflw/'  # --> Atlas
    # Wflw_prefix_path = '/media/ali/data/wflw/'  # --> local

    '''     augmented version'''
    augmented_train_pose = Wflw_prefix_path + 'training_set_256/augmented/pose/'
    augmented_train_annotation = Wflw_prefix_path + 'training_set_256/augmented/annotations/'
    augmented_train_hm = Wflw_prefix_path + 'training_set_256/augmented/hm/'
    augmented_train_hm_1d = Wflw_prefix_path + 'training_set_256/augmented/hm_1d/'
    augmented_train_atr = Wflw_prefix_path + 'training_set_256/augmented/atrs/'
    augmented_train_image = Wflw_prefix_path + 'training_set_256/augmented/images/'
    augmented_train_tf_path = Wflw_prefix_path + 'training_set_256/augmented/tf/'
    '''     original version'''
    no_aug_train_annotation = Wflw_prefix_path + 'training_set_256/no_aug/annotations/'
    no_aug_train_hm = Wflw_prefix_path + 'training_set_256/no_aug/hm/'
    no_aug_train_hm_1d = Wflw_prefix_path + 'training_set_256/no_aug/hm_1d/'
    no_aug_train_atr = Wflw_prefix_path + 'training_set_256/no_aug/atrs/'
    no_aug_train_pose = Wflw_prefix_path + 'training_set_256/no_aug/pose/'
    no_aug_train_image = Wflw_prefix_path + 'training_set_256/no_aug/images/'
    no_aug_train_tf_path = Wflw_prefix_path + 'training_set_256/no_aug/tf/'

    '''test:'''
    test_s = 'testing_set_256'
    test_annotation_path = Wflw_prefix_path + test_s + '/annotations/'
    test_image_path = Wflw_prefix_path + test_s + '/images/'

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    number_of_all_sample = 270956  # just images. dont count both img and lbls
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 15  # create . image from 1
    num_of_landmarks = 98


class CofwConf:
    # Cofw_prefix_path = '/media/data3/ali/FL/new_data/cofw/'  # --> zeus
    Cofw_prefix_path = '/media/data2/alip/FL/new_data/cofw/'  # --> atlas
    # Cofw_prefix_path = '/media/ali/data/new_data/cofw/'  # --> local

    augmented_train_pose = Cofw_prefix_path + 'training_set_256/augmented/pose/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set_256/augmented/annotations/'
    augmented_train_hm = Cofw_prefix_path + 'training_set_256/augmented/hm/'
    augmented_train_hm_1d = Cofw_prefix_path + 'training_set_256/augmented/hm_1d/'
    augmented_train_image = Cofw_prefix_path + 'training_set_256/augmented/images/'

    augmented_train_tf_path = Cofw_prefix_path + 'training_set_256/augmented/tf/'
    no_aug_train_tf_path = Cofw_prefix_path + 'training_set_256/no_aug/tf/'

    test_s = 'testing_set_256'
    test_annotation_path = Cofw_prefix_path + test_s + '/annotations/'
    test_image_path = Cofw_prefix_path + test_s + '/images/'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    augmentation_factor = 10
    number_of_all_sample = orig_number_of_training * augmentation_factor  # afw, train_helen, train_lfpw
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 5  # create . image from 1
    augmentation_factor_rotate = 30  # create . image from 1
    num_of_landmarks = 29


class D300WConf:
    w300w_prefix_path = '/media/data3/ali/FL/new_data/300W/'  # --> zeus
    # w300w_prefix_path = '/media/data2/alip/FL/new_data/300W/'  # --> atlas
    # w300w_prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    orig_300W_train = w300w_prefix_path + 'orig_300W_train/'
    augmented_train_pose = w300w_prefix_path + 'training_set_256/augmented/pose/'
    augmented_train_annotation = w300w_prefix_path + 'training_set_256/augmented/annotations/'
    augmented_train_hm = w300w_prefix_path + 'training_set_256/augmented/hm/'
    augmented_train_hm_1d = w300w_prefix_path + 'training_set_256/augmented/hm_1d/'
    augmented_train_image = w300w_prefix_path + 'training_set_256/augmented/images/'
    augmented_train_tf_path = w300w_prefix_path + 'training_set_256/augmented/tf/'

    no_aug_train_annotation = w300w_prefix_path + 'training_set_256/no_aug/annotations/'
    no_aug_train_hm = w300w_prefix_path + 'training_set_256/no_aug/hm/'
    no_aug_train_hm_1d = w300w_prefix_path + 'training_set_256/no_aug/hm_1d/'
    no_aug_train_pose = w300w_prefix_path + 'training_set_256/no_aug/pose/'
    no_aug_train_image = w300w_prefix_path + 'training_set_256/no_aug/images/'
    no_aug_train_tf_path = w300w_prefix_path + 'training_set_256/no_aug/tf/'

    '''test:'''
    test_s = 'testing_set_256'
    test_annotation_path = w300w_prefix_path + test_s + '/annotations/'
    test_image_path = w300w_prefix_path + test_s + '/images/'
    #

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    '''after augmentation'''
    number_of_all_sample = 134688   # afw, train_helen, train_lfpw
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 20  # create . image from 1
    num_of_landmarks = 68

