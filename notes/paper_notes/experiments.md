## Plan
- Need:
- HOList
  - SAT, Transformer (several sweeps, say 3 each) 
  - GNN, Directed SAT rerun before end to end
  - End to end runs
- HOL4
  - Transformer + SAT E2E

[//]: # (  - GNN, Transformer, SAT, Directed SAT, vanilla...)

[//]: # (  - n-step end to end loop..)

- HOLStep
  - Directed SAT? 
- MIZAR 
  - Quick run with everything..
- Lean
  - Get data 
  - Environment?
  - Small interaction experiment
  - Supervised experiment with s expressions
- INT
  - Get Graph + Transformer running 
  - Small experiment maybe..


Have 12 days, 4090, 3090, 4x a6000

Need to 

Generate full lean set towards the end



## 31/7
- Experiments:
  - Run SAT HOL4 pretrain, and start TacticZero with it
  - Run GNN, SAT and Transformer Variants with HOList pretraining for remaining machines
  - Transformer over HOLStep expressions (i.e. run TOKEN_RE and use that as sequence)
- Paper:
  - Diagram for system? MWPToolkit inspired, showing logging, checkpointing, database, transforms, models etc.
  - Put some HOLStep graphs/tables in, and GNN vs SAT for HOList
- Code:
  - Ensemble? Try on HOLStep and HOL4 for now, with just Transformer + GNN, use extra layer to map from 2 x embed_dim to 1, then into old classifier