#!/bin/bash
wandb artifact put --name test_project/holstep_raw_data --type dataset_zip tmp/raw_data.zip
wandb artifact put --name test_project/holstep_hol_dict --type dict tmp/dicts/hol_train_dict
rm -r wandb
