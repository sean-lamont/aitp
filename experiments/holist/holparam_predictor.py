"""Compute embeddings and predictions from a saved holparam checkpoint."""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from typing import List
from typing import Optional
from typing import Text

import logging

import numpy as np
import torch
from torch_geometric.data import Batch

from data.utils.graph_data_utils import to_data
from experiments.holist import predictions
from experiments.holist.utilities import process_sexp
from models.holist_models.gnn.gnn_encoder import GNNEncoder
from pymongo import MongoClient
from models.holist_models.tactic_predictor import TacticPrecdictor, CombinerNetwork
from tqdm import tqdm

GOAL_EMB_TYPE = predictions.GOAL_EMB_TYPE
THM_EMB_TYPE = predictions.THM_EMB_TYPE
STATE_ENC_TYPE = predictions.STATE_ENC_TYPE


def recommend_from_scores(scores: List[List[float]], n: int) -> List[List[int]]:
    """Return the index of the top n predicted scores.

    Args:
      scores: A list of tactic probabilities, each of length equal to the number
        of tactics.
      n: The number of recommendations requested.

    Returns:
      A list of the indices with the highest scores.
    """

    def top_idx(scores):
        return np.array(scores).argsort()[::-1][:n]

    return [top_idx(s) for s in scores]


'''

Torch Reimplementation of original TF1 HolparamPredictor

'''


# todo convert to and revert vectors from torch
class HolparamPredictor(predictions.Predictions):
    """Compute embeddings and make predictions from a save checkpoint."""

    def __init__(self,
                 ckpt: Text,
                 max_embedding_batch_size: Optional[int] = 128,
                 max_score_batch_size: Optional[int] = 128) -> None:
        """Restore from the checkpoint into the session."""
        super(HolparamPredictor, self).__init__(
            max_embedding_batch_size=max_embedding_batch_size,
            max_score_batch_size=max_score_batch_size)

        # todo load model from ckpt, link to training module

        self.embedding_model_goal = GNNEncoder(input_shape=1500,
                                               embedding_dim=128,
                                               num_iterations=12,
                                               dropout=0.2)

        self.embedding_model_premise = GNNEncoder(input_shape=1500,
                                                  embedding_dim=128,
                                                  num_iterations=12,
                                                  dropout=0.2)

        self.tac_model = TacticPrecdictor(
            embedding_dim=1024,
            num_tactics=41)

        self.combiner_model = CombinerNetwork(
            embedding_dim=1024,
            num_tactics=41,
            tac_embed_dim=128)

        self.embedding_model_goal.eval()
        self.embedding_model_premise.eval()
        self.tac_model.eval()
        self.combiner_model.eval()

        # todo configurable
        client = MongoClient()
        db = client['holist']
        expr_col = db['expression_graphs']
        vocab_col = db['vocab']
        filter = ['tokens', 'edge_index', 'edge_attr']

        vocab = {k['_id']: k['index'] for k in vocab_col.find()}

        logging.info("Loading expression dictionary..")

        self.expr_dict = {v["_id"]: to_data({x: v["data"][x] for x in filter}, data_type='graph', vocab=vocab)
                          for v in tqdm(expr_col.find({}))}

        print("testing..")
        embs = self._batch_goal_embedding([list(self.expr_dict.keys())[0]])
        print(embs)
        thms = self._batch_goal_embedding([list(self.expr_dict.keys())[1], list(self.expr_dict.keys())[2]])
        print(thms)
        scores = self._batch_tactic_scores([embs[0], embs[0]])
        print(scores)
        print(self.thm_embedding(list(self.expr_dict.keys())[1]))
        print(self._batch_thm_scores(thms, thms))

    def _goal_string_for_predictions(self, goals: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(goal) for goal in goals]

    def _thm_string_for_predictions(self, thms: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(thm) for thm in thms]

    def _batch_goal_embedding(self, goals: List[Text]) -> List[GOAL_EMB_TYPE]:
        """From a list of string goals, compute and return their embeddings."""
        # Get the first goal_net collection (second entry may be duplicated to align
        # with negative theorems)
        with torch.no_grad():
            goals = self._goal_string_for_predictions(goals)
            goal_data = Batch.from_data_list([self.expr_dict[t] for t in goals])
            embeddings = self.embedding_model_goal(goal_data)
            embeddings = embeddings.numpy()
        return embeddings

    def _batch_thm_embedding(self, thms: List[Text]) -> List[THM_EMB_TYPE]:
        """From a list of string theorems, compute and return their embeddings."""
        # The checkpoint should have exactly one value in this collection.
        with torch.no_grad():
            thms = self._thm_string_for_predictions(thms)
            # todo configure data type
            thms_data = Batch.from_data_list([self.expr_dict[t] for t in thms])
            embeddings = self.embedding_model_premise(thms_data)
            embeddings = embeddings.numpy()
        return embeddings

    def thm_embedding(self, thm: Text) -> THM_EMB_TYPE:
        """Given a theorem as a string, compute and return its embedding."""
        # Pack and unpack the thm into a batch of size one.
        # [embedding] = self.batch_thm_embedding([thm])
        embedding = self.batch_thm_embedding([thm])
        return embedding

    def proof_state_from_search(self, node) -> predictions.ProofState:
        """Convert from proof_search_tree.ProofSearchNode to ProofState."""
        return predictions.ProofState(goal=str(node.goal.conclusion))

    def proof_state_embedding(
            self, state: predictions.ProofState) -> predictions.EmbProofState:
        return predictions.EmbProofState(goal_emb=self.goal_embedding(state.goal))

    def proof_state_encoding(
            self, state: predictions.EmbProofState) -> STATE_ENC_TYPE:
        return state.goal_emb

    def _batch_tactic_scores(
            self, state_encodings: List[STATE_ENC_TYPE]) -> List[List[float]]:
        """Predict tactic probabilities for a batch of goals.

        Args:
          state_encodings: A list of n goal embeddings.

        Returns:
          A list of n tactic probabilities, each of length equal to the number of
            tactics.
        """
        # The checkpoint should have only one value in this collection.
        with torch.no_grad():
            x = torch.from_numpy(np.array(state_encodings))

            # [tactic_scores] = self.tac_model(x).tolist()
            tactic_scores = self.tac_model(x).numpy()

        return tactic_scores

    def _batch_thm_scores(self,
                          state_encodings: List[STATE_ENC_TYPE],
                          thm_embeddings: List[THM_EMB_TYPE],
                          tactic_id: Optional[int] = None) -> List[float]:
        """Predict relevance scores for goal, theorem pairs.

        Args:
          state_encodings: A proof state encoding. (effectively goal embedding)
          thm_embeddings: A list of n theorem embeddings. Theorems are paired by
            index with corresponding goals.
          tactic_id: Optionally tactic that the theorem parameters will be used in.

        Returns:
          A list of n floats, representing the pairwise score of each goal, thm.
        """
        del tactic_id  # tactic id not use to predict theorem scores.
        # The checkpoint should have only one value in this collection.
        assert len(state_encodings) == len(thm_embeddings)
        # todo use non tac dependent model

        tactic_id = 0
        with torch.no_grad():
            scores = self.combiner_model(torch.Tensor(state_encodings),
                                         torch.Tensor(thm_embeddings),
                                         torch.LongTensor(tactic_id)).numpy()

        return scores


class TacDependentPredictor(HolparamPredictor):
    """Derived class, adds dependence on tactic for computing theorem scores."""

    def __init__(self,
                 ckpt: Text,
                 max_embedding_batch_size: Optional[int] = 128,
                 max_score_batch_size: Optional[int] = 128) -> None:
        """Restore from the checkpoint into the session."""
        super(TacDependentPredictor, self).__init__(
            ckpt,
            max_embedding_batch_size=max_embedding_batch_size,
            max_score_batch_size=max_score_batch_size)
        self.selected_tactic = -1

    def _batch_thm_scores(self,
                          state_encodings: List[STATE_ENC_TYPE],
                          thm_embeddings: List[THM_EMB_TYPE],
                          tactic_id: Optional[int] = None) -> List[float]:
        """Predict relevance scores for goal, theorem pairs.

        Args:
          state_encodings: A proof state encoding.
          thm_embeddings: A list of n theorem embeddings. Theorems are paired by
            index with corresponding goals.
          tactic_id: Optionally tactic that the theorem parameters will be used in.

        Returns:
          A list of n floats, representing the pairwise score of each goal, thm.
        """
        # Check that the batch size for states and thms is the same.
        assert len(state_encodings) == len(thm_embeddings)

        # Tile the tactic to the batch size.
        # tactic_ids = np.tile(tactic_id, [len(state_encodings)])

        # The checkpoint should have only one value in this collection.
        with torch.no_grad():
            scores = self.combiner_model(torch.Tensor(state_encodings),
                                         torch.Tensor(thm_embeddings),
                                         torch.LongTensor(tactic_id)).numpy()
        return scores
