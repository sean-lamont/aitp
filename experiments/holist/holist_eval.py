from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import hydra

from experiments.holist.process_config import get_prover_options, process_prover_flags

""""

DeepHOL non-distributed prover.

"""

from experiments.holist import prover_runner


@hydra.main(config_path="configs/new_confs", config_name="holist_eval")
def holist_eval(config):
  prover_options = get_prover_options(config)
  prover_runner.program_started()
  prover_runner.run_pipeline(process_prover_flags(config, prover_options))




if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  holist_eval()
