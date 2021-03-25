from configuration import DatasetName, DatasetType, D300WConf, InputDataSize, CofwConf, WflwConf

from train_hgNet import TrainHg
from train_efn import TrainEfn

from data_helper import DataHelper

if __name__ == '__main__':
    dlp = DataHelper()
    '''loss'''
    # theta_0 = 0.5
    # theta_1 = 0.85
    # dlp.depict_weight_map_function(theta_0=theta_0, theta_1=theta_1)
    # dlp.depict_loss(theta_0=theta_0, theta_1=theta_1)

    # trainer = TrainHg(dataset_name=DatasetName.ds_300W, use_augmented=True)
    # trainer = TrainHg(dataset_name=DatasetName.ds_cofw, use_augmented=True)
    # # trainer = TrainHg(dataset_name=DatasetName.ds_wflw, use_augmented=True)
    # trainer.train(arch='hrnet', weight_path=None, use_inter=True)
    # # # trainer.train(arch='hrnet', weight_path='./models/last_hr.h5', use_inter=True)

    '''train hourGlass'''
    # trainer = TrainHg(dataset_name=DatasetName.ds_cofw, use_augmented=True)
    trainer = TrainHg(dataset_name=DatasetName.ds_wflw, use_augmented=True)
    # trainer = TrainHg(dataset_name=DatasetName.ds_300W, use_augmented=True)
    trainer.train(arch='hgNet', weight_path=None, use_inter=True)
    # trainer.train(arch='hgNet', weight_path='./models/last_hg.h5', use_inter=True)


    '''train hourGlass'''
    # trainer_efn = TrainEfn(dataset_name=DatasetName.ds_cofw, use_augmented=True)
    # trainer_efn = TrainEfn(dataset_name=DatasetName.ds_wflw, use_augmented=True)
    # trainer_efn = TrainEfn(dataset_name=DatasetName.ds_300W, use_augmented=True)
    # trainer_efn.train(arch='efn', weight_path=None)
    # trainer_efn.train(arch='efn', weight_path='./models/last_efn.h5')
