from data.data_modules import SequenceDataModule, GraphDataModule, RelationDataModule
from experiments.pyrallis_configs import DataConfig


def get_data(data_config: DataConfig):
    if data_config.type == 'graph':
        return GraphDataModule(config=data_config)
    if data_config.type == 'sequence':
        return SequenceDataModule(config=data_config)
    if data_config.type == 'relation':
        return RelationDataModule(config=data_config)
    # if data_config.type == 'holist_train':
    #     return HOListDataModule(dir=data_config.dir)
    else:
        raise NotImplementedError
