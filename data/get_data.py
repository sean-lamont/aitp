from data.hol4.mongo_to_torch import HOL4DataModule, HOL4DataModuleGraph, HOL4SequenceModule
from data.mizar.mizar_data_module import MizarDataModule
from data.utils.dataset import H5DataModule
from experiments.pyrallis_test import DataConfig


def get_data(data_config: DataConfig):
    if data_config.source == 'h5':
        return H5DataModule(config=data_config)
    if data_config.source == 'hol4':
        return HOL4DataModule(dir=data_config.dir)
    if data_config.source == 'hol4_graph':
        return HOL4DataModuleGraph(dir=data_config.dir)
    if data_config.source == 'hol4_sequence':
        return HOL4SequenceModule(dir=data_config.dir)
    if data_config.source == 'mizar':
        return MizarDataModule(dir=data_config.dir)
    else:
        raise NotImplementedError
