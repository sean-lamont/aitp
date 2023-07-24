## Notes

### Abstract

Recent interest in Artificial Intelligence for Theorem Proving (AITP) has given rise to a plethora of benchmarks and
methodologies,
particularly in the area of Interactive Theorem Proving (ITP).
Research in the area is fragmented, with a diverse set of approaches being spread across several ITP systems.
This presents a significant challenge to the comparison of methods, which are often complex and difficult to replicate.
To address this, we present MEDIATE: A Modular Environment for Insights and Development in Automated Theorem Experiments.
By separating the learning approach from the data and ITP environment, MEDIATE allows for the fair and efficient
comparison of approaches between systems.
MEDIATE is designed to seamlessly incorporate new systems and benchmarks, and currently accommodates HOList, HOLStep,
MIZAR40, LeanStep, LeanGym and HOL4's TacticZero.
We demonstrate the utility of MEDIATE through a study of embedding architectures in the context of the above systems.

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

1. Current problems: Results are hard to calibrate and compare as there are many different ML techniques applied across
   different systems
2. Multitude of embedding approaches, large body of literature on GNNs in this area, more recently focus has been on
   tranformers, no thorough comparison between in ITPs
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
    - Database format is platform independent (separate from torch, JAX, TF etc.)
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
    - Metric on ease of integration (e.g. MIZAR40, version of Hypertree Equations environment) something along the lines
      of #lines of code
    - Custom code needed for HOList
    - Easy to run experiments
- Embeddings
    - Show seq2seq and naive transformers (in HOLStep) do clearly worse
    - Secondary characteristics of models, why to use one over the other (e.g. Transformer better for unsupervised
      training)
    - Qualitative results, maybe something between systems
    - Directed SAT Improves over original HOList GNN (TBD)

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
    - CoqGym, IsarStep? 
- RL
    - Importance Sampling
    - Other setups e.g. Q learning, actor critic
    - Distributed Training/Updates
    - Adding RL wrapper to HOList and Lean (if implemented)
    - Combining/Comparing RL to supervised
        - E.g. Comparison of end to end RL in TacticZero with supervised logging and training loop of HOList, and vice
          versa
- Adding real open source ITP project (e.g. CakeML)
- Conjecturing

# Paper skeleton

## Intro
Interactive Theorem Proving (ITP) is a key paradigm of formal verification. 
With a human providing high level proof guidance, ITP systems have been used to develop verified compilers (cite), formalise mathematical conjectures (cite) and develop provably correct microkernels (cite).
Successful proof guidance requires proficiency in both the formal system and the application domain. 
This has limited the scale and widespread adoption of formal methods, with e.g...
The nascent field of Artificial Intelligence for Theorem Proving (AITP) has the potential to address this, 
with several strong results in automating human ITP guidance.
AITP research encompasses several mathematical reasoning tasks, such as math word problems (cite) and autoformalisation (cite).
The focus of this paper is specifially AI for Interactive Theorem Proving, which we refer to as AI-ITP.

Current datasets and environments for AI-ITP are divided across several distinct ITP systems.
Although this provides a variety of tasks for benchmarking,
it poses a challenge to the fair and efficient comparison of the automation approaches.
This is further complicated by the broad range of methods which have been applied in the area,
which are generally tested only in the context of a single ITP system (cite).
(examples, HOList over the HOL Light prover, LeanStep/LeanGym over Lean,
TacticZero and TacticToe,.. with HOL4).

[//]: # (Current benchmarks have made significant progress in the generation of quality ITP proof data,)
[//]: # (still small in the context of large models and compared to NLP. )
[//]: # (Through curating these data sources into a centralised platform, we have created a large dataset  of xxx examples and xxx GB of proof data.)

A central component of AI-ITP systems is their embedding model.
Vector embeddings of the ITP logical expressions are required for the application of learning algorithms.
These are used for tasks such as premise selection, goal selection which are essential for proof guidance.

Expressions are either treated as a natural language sequence, or as a directed graph derived from their abstract syntax tree
representation. It has been argued that a graph representation is more appropriate, with a large body of work in AI-ITP
using Graph Neural Network (GNN) as the embedding model to achieve strong results in several tasks.

However, Transformer models applied to the sequence representation have also demonstrated strong performance by
leveraging large models pre-trained on NLP datasets.

Both architectures have fundamental limitations, as GNNs suffer from poor integration of global information, and vanlla
Transformers ignore the structural information of the expression.
Recent work has shown that combining both architectures leads to improved performance on several graph learning tasks.
In the context of theorem proving however, there has been no thorough comparison between approaches.

We address these two issues by ... our contributions can be summarised as ...

## Background


HOList provides a large scale dataset and interactive environment for HOL Light, with a Breadth First Search proo large
scale dataset and interactive environment for HOL Light, with a Breadth First Search proof

and LeanStep/LeanGym are the two largest and

Related work
Unified toolkit proposed for MWP, MWPToolkit.

- However, AI for ITP has several additional considerations:
    - Fundamentally different learning approaches. Range from E2E RL, direct supervised learning, pretrained LLMs ..
        - Makes it difficult to be fully modular as there are some unavoidable dependencies between e.g. environment and
          model (e.g. HOL4 environment for TacticZero)
    - Interaction with an environment
    - Several axes of variation in algorithm design. Learning paradigm, proof search, embedding arch., action selection
      arch.


AIITP benchmarks mentioned, limited in single system.
Other unified frameworks, e.g. NLP, CV, GNN (graph benchmarks), which have provided strong value to their respective areas




[//]: # (In this work, we investigate the importance of embedding architecture across several theorem proving tasks.)
[//]: # (We show a surprising improvement of 50\% over the state of the art TacticZero algorithm, with the only change being the)
[//]: # (embedding approach.)
[//]: # (We also provide comparisons of embedding architectures across the HOLStep and HOList benchmarks, and a HOL4 premise)
[//]: # (selection task.)
[//]: # (We provide a qualitative analysis of embeddings to help explain the large improvement, showing that the improved model)
[//]: # (provides more semantically aware embeddings.)

## Background


#### Provers

- Lean, metamath (mizar), HOL4, HOL-Light

#### Approaches

- Task
    - Supervised tasks, premise selection
- Proof Search
    - BFS, Fringe, MCTS
- Model Architecture
    - Embeddings
        - Transformer, LSTM, Tree-LSTM, GNN, BoW, ... (more detail in second part)
    - Tactic/arg
        - GPT, fixed tactic MLPs, premise ranking
- Learning approach
    - Pretrain, fine tune pipeline
    - RL end to end
    -
- AI-ITP component diagram of the above

#### Current benchmarks and datasets

- Datasets
    - LeanStep, mizar40, mizar60?, HOLStep,
        - Environments
            - HOList, LeanGym, CoqGym, HOL4

        - As outlined in Figure x, benchmarks consider each system in isolation, and generally focus on a small set of
          learning and proof search approaches.
          Given the many differences between ITPs, this is unsurprising, as there is significant effort required to set
          up these systems for automated proving.
          However, there is a clear bee
    -
        - Embedding architectures
            - Large background on GNN in ITP, Transformer more recently with Neural Theorem Proving and Lean(step/gym)
            - GNN
            - Transformer
            - SAT
            - Directed SAT


## AI-ITP

### Frame

### Embeddings

Intro


- Framework Overview
    - Architecture diagram
    - Case study/examples?
        - Data processing
            - s-expressions with lean
                - After hooking to output new format, can reuse sexpression scripts from HOList
                - Models are similarly adaptable
            - MIZAR
                - polished notation similar to HOL4
                - some small changes, and was in same AST format
        - Large scale training
            - Stream dataset for out of memory, no extra implementation necessary
            - Useful for Lean, HOLStep (2M)
    - Metric? e.g. lines of code to add new setup
    - Vector database?

- Embedding Experiments
    - Supervised
        - HOList
            - (Expected) Directed SAT outperforms GNN
        - HOL4
        - HOLStep
            - Naive transformer does poorly
        - Mizar40
        - Lean?
    - Ensemble?
    - E2E
        - HOList
        - TacticZero
        - Lean?

    - Qualitative study
        - Syntactic vs Semantic for TacticZero autoencoder vs fully trained
        - Comparing embeddings between different systems, i.e. closest neighbors?






- Learning Approach
    - Reinforcement Learning, Online Policy Gradient (TacticZero) (HOL4)
    - Supervised Proxy Task (HOList)
    - Self supervised pre-training + fine-tuning on upstream task (Lean)

    - Supervised (labelled data)
        - Argument selection (HOList, HOL4 (new), LeanStep, HOLStep, MIZAR40)
        - Tactic selection (HOList, HOL4 (new), LeanStep)

    - Auxillary tasks (PACT)
        - Type prediction, next_lemma, skip_tree (LeanStep)

    - Self supervised

    - Goal Selection
        - Fringes
        - BFS
        - MCTS

Figures:


Tables:
- Benchmark vs Details (size, system, interactive)
    - Highlight in bold what is currently included
    - Include HOL4 premise selection
    - HOList, LeanStep, HOLStep, MIZAR40, LeanGym, TacticZero, (CoqGym, IsarStep etc from survey)

- Interactive Approaches vs Details
  - Proof search
    - BFS, MCTS, Fringe
  - Embedding architecture
    - GNN, Transformer, Autoencoder
  - Tactic/Argument selection 
    - Generative (lean, hypertree), Fixed set (HOL4, HOList)
  - Learning approach
    - RL (TacticZero), Supervised proof logs (HOList, Lean, Hypertree?)
     
- Results
  - Embed Architecture vs performance

Diagrams
- System architecture
- AI-ITP approaches
  - Generalised loop
- Embedding plots
- RL plot?
