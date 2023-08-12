from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re
import time

from environments.holist import proof_assistant_pb2
from experiments.holist import deephol_pb2
from experiments.holist import io_util
from experiments.holist import prover_util

HIST_AVG = deephol_pb2.ProverOptions.HIST_AVG
HIST_CONV = deephol_pb2.ProverOptions.HIST_CONV
HIST_ATT = deephol_pb2.ProverOptions.HIST_ATT


def _verify_prover_options(prover_options: deephol_pb2.ProverOptions) -> None:
    """Asserts some (incomplete) consistency requirements over prover_options."""
    for field_name in [
        'path_tactics', 'path_tactics_replace', 'path_theorem_database',
        'path_model_prefix'
    ]:
        if not prover_options.HasField(field_name):
            logging.fatal('Missing field "%s" in ProverOptions', field_name)
    if prover_options.prover not in ['nobacktrack', 'bfs']:
        logging.fatal('Unsupported proof strategy: "%s"', prover_options.prover)

    history_dependent = [HIST_AVG, HIST_CONV, HIST_ATT]
    if prover_options.model_architecture in history_dependent:
        if not prover_options.path_emb_model_prefix:
            logging.fatal(
                'History dependent model %s requires embeddings checkpoint '
                'path_emb_model_prefix.',
                deephol_pb2.ProverOptions.ModelArchitecture.Name(
                    prover_options.model_architecture))

        # Light assertions on file naming conventions for embedding consistency.
        # Embedding checkpoint number should be the end of the input file.
        emb_checkpoint_num = next(
            re.finditer(r'\d+$', prover_options.path_emb_model_prefix)).group(0)
        if emb_checkpoint_num not in prover_options.path_model_prefix:
            logging.fatal(
                'Embeddings checkpoint number (%s) was not found '
                'in the path of predictions checkpoint (%s), indicating '
                'it was trained with different embeddings.', emb_checkpoint_num,
                prover_options.path_model_prefix)


def get_prover_options(config, prover_round_tag='manual',
                       prover_round=-1) -> deephol_pb2.ProverOptions:
    """Returns a ProverOptions proto based on FLAGS."""

    prover_options = deephol_pb2.ProverOptions()

    if config.max_theorem_parameters is not None:
        logging.warning(
            'Overring max_theorem_parameters in prover options to %d.',
            config.max_theorem_parameters)

        prover_options.action_generator_options.max_theorem_parameters = (
            config.max_theorem_parameters)

    if prover_options.builtin_library:
        logging.warning('builtin_library is deprecated. Do not provide.')

        if str(prover_options.builtin_library) not in ['core']:
            logging.fatal('Unsupported built in library: %s',
                          prover_options.builtin_library)

    if config.timeout_seconds is not None:
        prover_options.timeout_seconds = config.timeout_seconds

    if not config.output:
        logging.fatal('Missing flag --output [recordio_pattern]')

        prover_options.prover_round = deephol_pb2.ProverRound(
            start_seconds=int(round(time.time())),
            tag=prover_round_tag,
            round=prover_round)

    _verify_prover_options(prover_options)

    # Log prover options.
    logging.info('Using prover_options:\n %s', str(prover_options))
    return prover_options


def process_prover_flags(config, prover_options):
    """Process the flags and return tasks, options and output path."""
    if config.splits:
        logging.info(
            '--splits flag overrides prover options for split selection.')
        splits_to_prove = prover_util.translate_splits(config.splits)
    else:
        splits_to_prove = list(prover_options.splits_to_prove)
    if not splits_to_prove and not config.tasks_by_fingerprint:
        logging.fatal('No split specification!')
    logging.info(
        'Splits to prove: %s', ', '.join(
            map(proof_assistant_pb2.Theorem.Split.Name, splits_to_prove)))

    if config.libraries:
        logging.info(
            '--libraries flag overrides prover options for library_tag selection.')
        if config.libraries == 'all':
            library_tags = set()
        else:
            library_tags = set([tag for tag in config.libraries.split(',')])
    else:
        library_tags = set(prover_options.library_tags)
    if not library_tags:
        logging.info('Disregarding library tags.')
    else:
        logging.info('Library tags to prove: %s', ', '.join(
            sorted(list(library_tags))))

    # Fail fast in case error in specifying tactics.

    _ = io_util.load_tactics_from_file(
        str(prover_options.path_tactics),
        str(prover_options.path_tactics_replace))

    theorem_db = io_util.load_theorem_database_from_file(
        str(prover_options.path_theorem_database))

    if not theorem_db.HasField('name'):
        theorem_db.name = 'default'  # Set a dummy name for backwards compatibility

        logging.warning('Missing theorem database name is set to %s',
                        theorem_db.name)

    if config.task_list and config.tasks:
        logging.fatal('Only one of --tasks or --task_list is allowed.')

    prover_tasks = prover_util.get_task_list(config.tasks, config.task_list,
                                             config.tasks_by_fingerprint,
                                             theorem_db, splits_to_prove,
                                             library_tags)

    # TODO(szegedy): Verify tasks that they all fit the theorem database(s)

    logging.info('Number of prover tasks: %d', len(prover_tasks))

    return (prover_tasks, prover_options, config.output)
