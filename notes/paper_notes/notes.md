## Notes

### Abstract

Recent interest in Artificial Intelligence for Theorem Proving (AITP) has given rise to a plethora of benchmarks and
methodologies,
particularly in the area of Interactive Theorem Proving (ITP).
Research in the area is fragmented, with a diverse set of approaches being spread across several ITP systems.
This presents a significant challenge to the comparison of methods, which are often complex and difficult to replicate.
To address this, we present METIS: A Modular Environment for Theorem-Proving in Interactive Systems.
By separating the learning approach from the data and ITP environment, METIS allows for the fast and efficient comparison of
approaches between systems.
METIS was designed to seamlessly incorporate new systems and benchmarks, and currently accommodates HOList, HOLStep,
MIZAR40 and HOL4's TacticZero.
We demonstrate the utility of METIS through a study of embedding architectures in the context of the above systems.

[//]: # (Through comparing the performance of approaches across a variety of ITP settings, we provide strong evidence that..)
These experiments lay the foundations for future work in evaluation of performance of formula embedding,
while simultaneously demonstrating the capability of the framework.

[//]: # (We show that the embedding approach of HOList applied to HOL4's TacticZero provides a significant improvement in)
[//]: # (performance.)
We complement these with a qualitative analysis of embeddings within and between systems, illustrating that improved
performance was associated with more semantically aware embeddings.
By streamlining the implementation and comparison of Machine Learning algorithms in the ITP context,
we anticipate METIS will be a springboard for future
research.

### Motivations
1. Current problems: Results are hard to calibrate and compare as there are many different ML techniques applied across different systems
2. Multitude of embedding approaches, large body of literature on GNNs in this area, more recently focus has been on tranformers, no thorough comparison between in ITPs
   1. Recent work in improving/combining both with SAT no study yet in ITP context

### Contributions

- METIS environment
    - Helps address lack of centralised platform for comparing results and approaches in ITP
    - Fully open source, springboard for future research
        - Speed up the testing of new ideas, without the need to reimplement data processing and supporting platform
        - Scalable, with multi GPU support with pytorch lightning and fast, efficient and streamable storage with
          MongoDB
        - Runs "out of the box", with prepackaged models, data sets and experiments.
        - Easy interface to add new systems, datasets and experiments
- Implementation and comparison of embedding methods in the context of ITP
    - Fair comparison of Transformer, GNN and SAT models across a variety of ITP data sets
    -

### METIS Overview

- Modules for Data, Models, Environments and Experiments
- Data
    - Centralised Database (using MongoDB) with a standardised format for tasks between systems
    - Contains code for preprocessing and formatting data either from environment interactions or fixed sources (proof
      logs, premise selection datasets)
    - Implemented to be scalable with large datasets. Can either load into memory or stream from the database in the
      case of limited memory or very large databases
    - Database is platform independent
    - Contains packaged Data Modules for specific learning tasks, using PyTorch Lightning
        - TacticZero Reinforcement Learning
        - HOList Training Approach
        - Premise Selection in HOLStep, HOL4, and MIZAR40
        - ***TODO*** Unsupervised learning
        - ***TODO*** HOList end to end loop
    - New data sources and systems are easily integrated and only need a "process_data" script to adapt it to the
      centralised format
    - ***TODO*** Vector/Embedding Database
        - Enables precomputing embeddings for speed as done in HOList
        - Can be used for knowledge discovery, finding e.g. similar expressions between different systems
    - ***TODO*** Stores RL traces and Log Probabilities, can be used to generate new samples and proof data
- Environments
    - Includes HOList environment and HOL4 Reinforcement Learning Environment
- Models
    - Module for model architectures from various sources
        - Embeddings
            - FormulaNet from HOLStep, GNN from HOList, Transformer
            - Structure Aware Attention and Directed Attention
        - TacticZero Networks
        - HOList Networks
- Experiments
    - Combines Data, Models and Environments to form experiments
        - Users define all of the above through experiment configurations, specifying the task, data source,
          environment (if any), model architectures, hyperparameters, checkpointing and logging details
        - Runs the experiment as specified above. Automatic logging integration with wandb, automatic checkpointing and
          resuming,
        - ***TODO***: Complex pipelines can be run for semi and unsupervised learning. E.g. Unsupervised pretraining ->
          Upstream task -> RL loop as in Lean paper

### Evaluation of contributions
- METIS
    - Metric on ease of integration (e.g. MIZAR40, version of Hypertree Equations environment) something along the lines of #lines of code
    - Custom code needed for HOList
    - Easy to run experiments
- Embeddings
  - Show seq2seq and naive transformers (in HOLStep) do clearly worse
  - Secondary characteristics of models, why to use one over the other (e.g. Transformer better for unsupervised training)
  - Qualitative results, maybe something between systems

### Immediate priorities

1. HOList Embeddings
   1. Supervised dataset with human, and human + synthetic logs
      1. SAT, GNN, Transformer, SAT + Directed with sweep
   2. Testing in end to end loop
      1. Possible reduced BFS budget, (not comparable to original paper, but better measure of performance)
    
2. TacticZero 
   1. Possible reduced number of steps to get tractable results in time
   2. Could also run faster experiment using precomputed replays, or with proof logs only (e.g. HOList Training setup)

3. Sweep over HOLStep 


### Future directions

- Comparison of proof search approaches
    - MCTS, BFS, Fringes
    - UpDown
- Adding other systems
    - LeanStep, Lean 
    - CoqGym (lower priority, not as active)
- RL 
  - Importance Sampling
  - Other setups e.g. Q learning, actor critic
  - Distributed Training/Updates 
  - Adding RL wrapper to HOList and Lean (if implemented)
  - Combining/Comparing RL to supervised 
    - E.g. Comparison of end to end RL in TacticZero with supervised logging and training loop of HOList, and vice versa
- Adding real open source ITP project (e.g. CakeML)
- Conjecturing
