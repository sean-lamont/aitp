from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
import pyrallis
from pyrallis import field


@dataclass
class GlobalConfig:
    GLOBAL_PATH: str = field(default="/home/sean/Documents/phd/repo/aitp/")


@dataclass
class EmbeddingModelConfig:
    model_attributes: Dict[str, Any]
    model_type: str = field(default="")


@dataclass
class OptimiserConfig:
    optimiser: str = field(default="AdamW")
    scheduler: Optional[str] = field(default=None)
    # Learning Rate
    learning_rate: float = field(default=1e-4)
    # Weight Decay
    weight_decay: float = field(default=1e-6)


@dataclass
class TacticConfig:
    MORE_TACTICS: bool = field(default=True)
    thms_tactic: List[str] = field(default=["simp", "fs", "metis_tac", "rw"], is_mutable=True)
    thm_tactic: List[str] = field(default=["irule", "drule"], is_mutable=True)
    term_tactic: List[str] = field(default=["Induct_on"], is_mutable=True)
    no_arg_tactic: List[str] = field(default=["strip_tac", "EQ_TAC"], is_mutable=True)

    @property
    def tactic_pool(self) -> List[str]:
        return self.thms_tactic + self.thm_tactic + self.term_tactic + self.no_arg_tactic

    def __post_init__(self):
        # A builtin method of dataclasses, used for post-processing our configuration.
        if not self.MORE_TACTICS:
            self.thms_tactic = ["simp", "fs", "metis_tac"]
            self.thm_tactic = ["irule"]
            self.term_tactic = ["Induct_on"]
            self.no_arg_tactic = ["strip_tac"]


@dataclass
class LoggingConfig:
    project: str
    notes: str = field(default=None)
    offline: bool = field(default=False)
    id: Optional[str] = field(default=None)


@dataclass
class ExperimentConfig:
    # Name of the experiment
    name: str
    # Type of experiment
    experiment: str
    # Directory for experiment
    directory: str
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    # CUDA Device ID(s) to use
    device: List[int] = field(default=[0], is_mutable=True)
    # Accelerator (one of "cpu", "gpu", or "tpu")
    accelerator: str = field(default='gpu')

    def __post_init__(self):
        if not self.logging_config.notes:
            self.logging_config.notes = self.name
        self.directory = self.directory + '/' + self.name + '_' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


@dataclass
class DataConfig:
    # Defines what DataModule to use
    type: str = field(default='graph')
    # Defines where to load data from. Currently, supports either local directories or MongoDB
    source: str = field(default='directory')
    # Options for data loading. If directory, specifies directory.
    # If MongoDB, specifies the database and collections to use
    data_options: Dict = field(default={}, is_mutable=True)
    # Specifies the batch size for the data loaders
    batch_size: int = field(default=32)
    # Specifies attributes to be included in the data objects, e.g. Positional Encoding
    attributes: Dict = field(default={}, is_mutable=True)


@dataclass
class PremiseSelectionConfig(GlobalConfig):
    exp_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    optimiser_config: OptimiserConfig = field(default_factory=OptimiserConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    # Number of Training Epochs
    epochs: int = field(default=30)
    # Batch Size
    batch_size: int = field(default=32)
    # Number of Samples to use in Validation
    val_size: int = field(default=4096)
    # Frequency with which to run validation
    val_frequency: int = field(default=2048)
    checkpoint_dir: str = field(default=None)
    limit_val_batches: bool = field(default=True)

    # @property
    # def checkpoint_dir(self) -> str:
    #     return self.directory + '/model_checkpoints'

    def __post_init__(self):
        if self.data_config.source == 'directory':
            self.data_config.attributes['dir'] = self.GLOBAL_PATH + self.data_config.dir
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.exp_config.directory + '/model_checkpoints'

# @dataclass
# class TacticZeroRLConfig(ExperimentConfig,
#                          OptimiserConfig,
#                          LoggingConfig,
#                          TacticConfig):
#
#     model_config: EmbeddingModelConfig
#     # gnn_layers
#     embedding_dim: int
#     exp_type: str
#     vocab_size: int
#     resume: bool
#     resume_id: str
#     pretrain: bool
#     notes: str
#     max_steps: int
#     gamma: float
#     lr: float
#     arg_len: int
#     data_type: str
#     train_goals: List[str]
#     test_goals: List[str]
#     token_enc: Any
#     graph_db: Dict
#     val_freq: int
#     reverse_database: Dict
#


# def main():
#     cfg = pyrallis.parse(config_class=PremiseSelectionConfig, config_path='configs/conf.yaml')
#     print(f'{cfg}')
#     print(f'{cfg.model_config.model_attributes["embedding_dim"]}')
#     print(pyrallis.dump(cfg))  # , open('./run_config.yaml','w'))
#
#
# if __name__ == '__main__':
#     main()
