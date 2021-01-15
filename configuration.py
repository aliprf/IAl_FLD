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

    CLR_METHOD = "triangular"
    MIN_LR = 1e-7
    MAX_LR = 1e-2
    STEP_SIZE = 10
    batch_size = 3
    # batch_size = 60
    epochs = 1500


class InputDataSize:
    image_input_size = 224
    img_center = image_input_size // 2

    hm_size = image_input_size // 4
    hm_center = hm_size // 2


class WflwConf:
    Wflw_prefix_path = '/media/data3/ali/FL/new_data/wflw/'  # --> Zeus
    # Wflw_prefix_path = '/media/data2/alip/FL/new_data/wflw/'  # --> Atlas
    # Wflw_prefix_path = '/media/ali/data/wflw/'  # --> local

    '''     augmented version'''
    augmented_train_pose = Wflw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Wflw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_hm = Wflw_prefix_path + 'training_set/augmented/hm/'
    augmented_train_atr = Wflw_prefix_path + 'training_set/augmented/atrs/'
    augmented_train_image = Wflw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Wflw_prefix_path + 'training_set/augmented/tf/'
    '''     original version'''
    no_aug_train_annotation = Wflw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_hm = Wflw_prefix_path + 'training_set/no_aug/hm/'
    no_aug_train_atr = Wflw_prefix_path + 'training_set/no_aug/atrs/'
    no_aug_train_pose = Wflw_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = Wflw_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = Wflw_prefix_path + 'training_set/no_aug/tf/'

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

    augmented_train_pose = Cofw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_hm = Cofw_prefix_path + 'training_set/augmented/hm/'
    augmented_train_image = Cofw_prefix_path + 'training_set/augmented/images/'

    augmented_train_tf_path = Cofw_prefix_path + 'training_set/augmented/tf/'
    no_aug_train_tf_path = Cofw_prefix_path + 'training_set/no_aug/tf/'

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
    # w300w_prefix_path = '/media/data3/ali/FL/new_data/300W/'  # --> zeus
    # w300w_prefix_path = '/media/data2/alip/FL/new_data/300W/'  # --> atlas
    w300w_prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    orig_300W_train = w300w_prefix_path + 'orig_300W_train/'
    augmented_train_pose = w300w_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = w300w_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_hm = w300w_prefix_path + 'training_set/augmented/hm/'
    augmented_train_image = w300w_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = w300w_prefix_path + 'training_set/augmented/tf/'

    no_aug_train_annotation = w300w_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_hm = w300w_prefix_path + 'training_set/no_aug/hm/'
    no_aug_train_pose = w300w_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = w300w_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = w300w_prefix_path + 'training_set/no_aug/tf/'

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

