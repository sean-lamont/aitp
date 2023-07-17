from data.data_modules import PremiseDataModule
from experiments.pyrallis_configs import DataConfig

def get_data(data_config: DataConfig):
    return PremiseDataModule(config=data_config)
