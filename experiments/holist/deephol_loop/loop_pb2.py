# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: deepmath/deephol/deephol_loop/loop.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from experiments.holist import deephol_pb2 as deepmath_dot_deephol_dot_deephol__pb2
from experiments.holist.deephol_loop import options_pb2 as deepmath_dot_deephol_dot_deephol__loop_dot_options__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(deepmath/deephol/deephol_loop/loop.proto\x12\x10\x64\x65\x65pmath_deephol\x1a\x1e\x64\x65\x65pmath/deephol/deephol.proto\x1a+deepmath/deephol/deephol_loop/options.proto\"\x8d\x04\n\nLoopConfig\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\"\n\x1anum_prover_tasks_per_round\x18\x02 \x01(\x05\x12\x1d\n\x12num_task_per_shard\x18\x03 \x01(\x05:\x01\x31\x12\x37\n\x0eprover_options\x18\x04 \x01(\x0b\x32\x1f.deepmath_deephol.ProverOptions\x12\x19\n\x11prover_tasks_file\x18\x05 \x01(\t\x12\x1c\n\x14path_model_directory\x18\x06 \x01(\t\x12$\n\x16\x63opy_model_checkpoints\x18\x07 \x01(\x08:\x04true\x12!\n\x15\x66resh_examples_shards\x18\x08 \x01(\r:\x02\x31\x30\x12\'\n\x1ahistorical_examples_shards\x18\t \x01(\r:\x03\x31\x30\x30\x12!\n\x15\x66resh_examples_rounds\x18\n \x01(\r:\x02\x31\x30\x12\x1c\n\x14inherited_proof_logs\x18\x0b \x01(\t\x12=\n\x11\x63onvertor_options\x18\x0c \x01(\x0b\x32\".deepmath_deephol.ConvertorOptions\x12#\n\x18random_tactic_num_rounds\x18\r \x01(\x05:\x01\x30\x12%\n\x1drandom_tactic_probability_min\x18\x0e \x01(\x02\"n\n\nLoopStatus\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1f\n\x13last_finished_round\x18\x02 \x01(\x05:\x02-1\x12\x15\n\rcurrent_round\x18\x03 \x01(\x05\x12\x1a\n\x12running_controller\x18\x04 \x01(\x04\"6\n\rProofLogsMeta\x12\r\n\x05round\x18\x01 \x01(\x05\x12\x16\n\x0enumber_of_logs\x18\x02 \x01(\x05')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'deepmath.deephol.deephol_loop.loop_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _LOOPCONFIG._serialized_start=140
  _LOOPCONFIG._serialized_end=665
  _LOOPSTATUS._serialized_start=667
  _LOOPSTATUS._serialized_end=777
  _PROOFLOGSMETA._serialized_start=779
  _PROOFLOGSMETA._serialized_end=833
# @@protoc_insertion_point(module_scope)
