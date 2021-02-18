from configuration import DatasetName, DatasetType, D300WConf, InputDataSize, CofwConf, WflwConf
from train import Train
from data_helper import DataHelper
if __name__ == '__main__':
    dlp = DataHelper()
    '''loss'''
    # theta_0 = 0.2
    # theta_1 = 0.8
    # dlp.depict_weight_map_function(theta_0=theta_0, theta_1=theta_1)
    # dlp.depict_loss(theta_0=theta_0, theta_1=theta_1)
    trainer = Train(dataset_name=DatasetName.ds_wflw, use_augmented=True)
    trainer.train(arch='hgNet', weight_path=None)