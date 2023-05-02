Run gen_data.sh from the data_utils directory. Will check whether raw_data and dict is present, if not then download these from wandb. 

It will then generate data for HOLStep using data_utils/generate_hol_dataset.py which generates a dictionary mapping all expressions (conjectures and statements)
to a condensed graph representation format (one_hot indices, and adjaceny matrix in COO format). It also generates train, val, test data as per the benchmark specifications, using the raw expressions as keys. 
When it comes to training/inference, the dictionary is indexed by the relevant statements to retrieve the graph information needed. 

Data is currently stored using MongoDB.