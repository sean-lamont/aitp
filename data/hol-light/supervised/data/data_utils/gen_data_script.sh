#!/bin/bash
x=tactic-zero/test_project/holstep_raw_data:latest 
y=tactic-zero/test_project/holstep_hol_dict:latest

if [ -d ../raw_data ]; then
	echo "Raw Data found"
else
	echo "Downloading raw HOL Step data from artifact $x..."
	wandb artifact get $x && unzip -d ../raw_data ./artifacts/holstep_raw_data:*/raw_data.zip 
fi



if [ -d ../dicts ]; then
	echo "Data Dict found"
else
	echo "Downloading data dict from artifact $y..."
	mkdir ../dicts/
	wandb artifact get $y && cp ./artifacts/holstep_hol_dict:*/hol_train_dict ../dicts/hol_train_dict
fi

echo "Removing artifacts.."
rm -r ./artifacts 

echo "Generating expression graphs from raw HOL Step data..."

#mkdir ../graph_data/

python3 -m generate_hol_dataset ../raw_data/raw_data 

#python3 -m combine_data 


