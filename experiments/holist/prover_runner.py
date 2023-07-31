"""Simple runner for the prover.

Iterate over the tasks sequentially.

This runner can run the prover on a set of tasks without the overhead of
starting a distributed job.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
from typing import List
from typing import Text

from absl import flags
from tqdm import tqdm

from environments.holist import proof_assistant_pb2
from experiments.holist import deephol_pb2
from experiments.holist import io_util
from experiments.holist import prover
from experiments.holist.utilities import stats

# placeholder
FLAGS = flags.FLAGS


def program_started():
    pass

def compute_stats(output):
    """Compute aggregate statistics given prooflog file."""
    logging.info('Computing aggregate statistics from %s', output)
    stat_list = [
        stats.proof_log_stats(log)
        for log in io_util.read_protos(output, deephol_pb2.ProofLog)
    ]
    if not stat_list:
        logging.info('Empty stats list.')
        return
    aggregate_stat = stats.aggregate_stats(stat_list)
    logging.info('Aggregated statistics:')
    logging.info(stats.aggregate_stat_to_string(aggregate_stat))
    return aggregate_stat


def run_pipeline(prover_tasks: List[proof_assistant_pb2.ProverTask],
                 prover_options: deephol_pb2.ProverOptions, path_output: Text):
    """Iterate over all prover tasks and store them in the specified file."""

    if FLAGS.output.split('.')[-1] != 'textpbs':
        logging.warning('Output file should end in ".textpbs"')

    prover.cache_embeddings(prover_options)
    this_prover = prover.create_prover(prover_options)
    proof_logs = []

    print (f"Running {len(prover_tasks)}..")
    random.shuffle(prover_tasks)
    for i, task in tqdm(enumerate(prover_tasks)):
        proof_log = this_prover.prove(task)
        # proof_log.build_data = build_data.BuildData()
        proof_logs.append(proof_log)

        if (i + 1) % 10 == 0:
            if path_output:
                logging.info('Writing %d proof logs as text proto to %s',
                             len(proof_logs), path_output)
                io_util.write_text_protos(path_output, proof_logs)

    if path_output:
        logging.info('Writing %d proof logs as text proto to %s',
                     len(proof_logs), path_output)
        io_util.write_text_protos(path_output, proof_logs)

    logging.info('Proving complete!')

    compute_stats(path_output)
