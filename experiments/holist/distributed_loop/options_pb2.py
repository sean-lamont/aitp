# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: experiments/holist/deephol_loop/options.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-experiments/holist/deephol_loop/options.proto\x12\x10\x64\x65\x65pmath_deephol\"\x8a\x03\n\x18TFExamplePipelineOptions\x12=\n\x11\x63onvertor_options\x18\x01 \x01(\x0b\x32\".deepmath_deephol.ConvertorOptions\x12\x12\n\nproof_logs\x18\x02 \x01(\t\x12\x18\n\x10theorem_database\x18\x0c \x01(\t\x12\x0f\n\x07out_dir\x18\x03 \x01(\t\x12\x1f\n\x10\x65xport_tfrecords\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x0f\x65xport_sstables\x18\x05 \x01(\x08:\x04true\x12\'\n\x18\x65xport_textpb_proof_logs\x18\x06 \x01(\x08:\x05\x66\x61lse\x12)\n\x1a\x65xport_recordio_proof_logs\x18\x07 \x01(\x08:\x05\x66\x61lse\x12\"\n\x14normalize_proof_logs\x18\t \x01(\x08:\x04true\x12\x1a\n\x0bstrip_paths\x18\n \x01(\x08:\x05\x66\x61lse\x12\x1c\n\x0e\x63reate_thms_ls\x18\x0b \x01(\x08:\x04true\"\xdd\x03\n\x10\x43onvertorOptions\x12\x1a\n\x0eprooflogs_path\x18\x01 \x01(\tB\x02\x18\x01\x12\x1d\n\x15theorem_database_path\x18\x02 \x01(\t\x12\x14\n\x0ctactics_path\x18\x03 \x01(\t\x12\x1f\n\x11replacements_hack\x18\x04 \x01(\x08:\x04true\x12Y\n\x10scrub_parameters\x18\x05 \x01(\x0e\x32\x36.deepmath_deephol.ConvertorOptions.ScrubParametersEnum:\x07NOTHING\x12L\n\x06\x66ormat\x18\x06 \x01(\x0e\x32\x32.deepmath_deephol.ConvertorOptions.TFExampleFormat:\x08HOLPARAM\x12\"\n\x1a\x65xtract_only_closed_proofs\x18\x07 \x01(\x08\"X\n\x13ScrubParametersEnum\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0b\n\x07NOTHING\x10\x01\x12\x0b\n\x07TESTING\x10\x02\x12\x1a\n\x16VALIDATION_AND_TESTING\x10\x03\"0\n\x0fTFExampleFormat\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x0c\n\x08HOLPARAM\x10\x01')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'experiments.holist.deephol_loop.options_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CONVERTOROPTIONS.fields_by_name['prooflogs_path']._options = None
  _CONVERTOROPTIONS.fields_by_name['prooflogs_path']._serialized_options = b'\030\001'
  _globals['_TFEXAMPLEPIPELINEOPTIONS']._serialized_start=68
  _globals['_TFEXAMPLEPIPELINEOPTIONS']._serialized_end=462
  _globals['_CONVERTOROPTIONS']._serialized_start=465
  _globals['_CONVERTOROPTIONS']._serialized_end=942
  _globals['_CONVERTOROPTIONS_SCRUBPARAMETERSENUM']._serialized_start=804
  _globals['_CONVERTOROPTIONS_SCRUBPARAMETERSENUM']._serialized_end=892
  _globals['_CONVERTOROPTIONS_TFEXAMPLEFORMAT']._serialized_start=894
  _globals['_CONVERTOROPTIONS_TFEXAMPLEFORMAT']._serialized_end=942
# @@protoc_insertion_point(module_scope)
