"""Compute embeddings and predictions from a saved holparam checkpoint."""
from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

from typing import List
from typing import Optional
from typing import Text

import numpy as np

from experiments.holist import predictions
from experiments.holist.utilities import process_sexp

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

    def _goal_string_for_predictions(self, goals: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(goal) for goal in goals]

    def _thm_string_for_predictions(self, thms: List[Text]) -> List[Text]:
        return [process_sexp.process_sexp(thm) for thm in thms]

    def _batch_goal_embedding(self, goals: List[Text]) -> List[GOAL_EMB_TYPE]:
        """From a list of string goals, compute and return their embeddings."""
        # Get the first goal_net collection (second entry may be duplicated to align
        # with negative theorems)
        goals = self._goal_string_for_predictions(goals)
        embeddings = self.embedding_model_goal(goals)
        return embeddings

    def _batch_thm_embedding(self, thms: List[Text]) -> List[THM_EMB_TYPE]:
        """From a list of string theorems, compute and return their embeddings."""
        # The checkpoint should have exactly one value in this collection.
        thms = self._thm_string_for_predictions(thms)
        embeddings = self.embedding_model_premises(thms)
        return embeddings

    def thm_embedding(self, thm: Text) -> THM_EMB_TYPE:
        """Given a theorem as a string, compute and return its embedding."""
        # Pack and unpack the thm into a batch of size one.
        [embedding] = self.batch_thm_embedding([thm])
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
        [tactic_scores] = self.tac_model(state_encodings)
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
        # todo convert to and revert from torch
        scores = self.combiner_model(state_encodings, thm_embeddings) #, true_tacs)

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
        tactic_ids = np.tile(tactic_id, [len(state_encodings)])
        # The checkpoint should have only one value in this collection.
        # todo convert to and revert from torch
        scores = self.combiner_model(state_encodings, thm_embeddings, tactic_ids)

        return scores
