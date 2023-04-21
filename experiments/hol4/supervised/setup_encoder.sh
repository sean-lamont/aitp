#!/bin/bash

# Generate expression and dependency databases

python3 gen_databases.py

# Generate data for encoder and train model

python3 gnn_rl_encoder.py

