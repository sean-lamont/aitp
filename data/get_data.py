from data.mizar.mizar_data_module import SequenceDataModule, GraphDataModule, RelationDataModule
from experiments.pyrallis_configs import DataConfig


def get_data(data_config: DataConfig):
    if data_config.source == 'graph':
        return GraphDataModule(config=data_config)
    if data_config.source == 'sequence':
        return SequenceDataModule(dir=data_config.dir)
    if data_config.source == 'relation':
        return RelationDataModule(dir=data_config.dir)
    # if data_config.source == 'holist_train':
    #     return HOListDataModule(dir=data_config.dir)
    else:
        raise NotImplementedError
