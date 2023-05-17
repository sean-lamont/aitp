import math

import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mha_nested(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, nheads: int,
               W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_out: torch.Tensor,
               b_q: torch.Tensor = None, b_k: torch.Tensor = None, b_v: torch.Tensor = None, b_out: torch.Tensor = None,
               dropout_p: float = 0.0) -> torch.Tensor:
    """Compute multi-head attention with nested tensors.
    Args:
        query (torch.Tensor): query of shape (N, L_t, E_q)
        key (torch.Tensor): key of shape (N, L_s, E_k)
        value (torch.Tensor): value of shape (N, L_s, E_v)
        nheads (int): number of heads in multi-head attention
        W_q (torch.Tensor): Weight for query input projection of shape (E_total, E_q)
        W_k (torch.Tensor): Weight for key input projection of shape (E_total, E_k)
        W_v (torch.Tensor): Weight for value input projection of shape (E_total, E_v)
        W_out (torch.Tensor): Weight for output projection of shape (E_out, E_total)
        b_q (torch.Tensor, optional): Bias for query input projection of shape E_total. Default: None. Defaults to None.
        b_k (torch.Tensor, optional): Bias for key input projection of shape E_total. Default: None. Defaults to None.
        b_v (torch.Tensor, optional): Bias for value input projection of shape E_total. Default: None. Defaults to None.
        b_out (torch.Tensor, optional): Bias for output projection of shape E_out. Default: None. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Where:
            N is the batch size
            L_t is the target sequence length (jagged)
            L_s is the source sequence length (jagged)
            E_q is the embedding size for query
            E_k is the embedding size for key
            E_v is the embedding size for value
            E_total is the embedding size for all heads combined
            E_out is the output embedding size
    Returns:
        torch.Tensor:  Output of shape (N, L_t, E_out)
    """

    N = query.size(0)
    E_total = W_q.size(0)
    assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
    E_head = E_total // nheads

    # apply input projection
    # (N, L_t, E_q) -> (N, L_t, E_total)
    query = F.linear(query, W_q, b_q)
    # (N, L_s, E_k) -> (N, L_s, E_total)
    key = F.linear(key, W_k, b_k)
    # (N, L_s, E_v) -> (N, L_s, E_total)
    value = F.linear(value, W_v, b_v)

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
    query = query.reshape(N, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    key = key.reshape(N, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    value = value.reshape(N, -1, nheads, E_head).transpose(1, 2)

    # query matmul key^T
    # (N, nheads, L_t, E_head) x (N, nheads, L_s, E_head)^T -> (N, nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    attn_weights = torch.matmul(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights matmul value
    # (N, nheads, L_t, L_s) x (N, nheads, L_s, E_head) -> (N, nheads, L_t, E_head)
    attn_output = torch.matmul(attn_weights, value)

    # merge heads
    # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    return attn_output
#%%
def mha_padded(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, nheads: int,
               attn_mask_q: torch.Tensor, attn_mask_kv: torch.Tensor,
               W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_out: torch.Tensor,
               b_q: torch.Tensor = None, b_k: torch.Tensor = None, b_v: torch.Tensor = None, b_out: torch.Tensor = None,
               dropout_p: float = 0.0) -> torch.Tensor:
    """Compute multi-head attention for padded out dense tensors.

    Args:
        query (torch.Tensor): query of shape (N, L_t, E_q)
        key (torch.Tensor): key of shape (N, L_s, E_k)
        value (torch.Tensor): value of shape (N, L_s, E_v)
        nheads (int): number of heads in multi-head attention
        attn_mask_q (torch.Tensor): boolean mask indicating locations that should not take part in attention for query, shape (N, L_t)
        attn_mask_kv (torch.Tensor): boolean mask indicating locations that should not take part in attention for key and value, shape (N, L_s)
        W_q (torch.Tensor): Weight for query input projection of shape (E_total, E_q)
        W_k (torch.Tensor): Weight for key input projection of shape (E_total, E_k)
        W_v (torch.Tensor): Weight for value input projection of shape (E_total, E_v)
        W_out (torch.Tensor): Weight for output projection of shape (E_out, E_total)
        b_q (torch.Tensor, optional): Bias for query input projection of shape E_total.. Defaults to None.
        b_k (torch.Tensor, optional): Bias for key input projection of shape E_total.. Defaults to None.
        b_v (torch.Tensor, optional): Bias for value input projection of shape E_total.. Defaults to None.
        b_out (torch.Tensor, optional): Bias for output projection of shape E_out. Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

        Where:
            N is the batch size
            L_t is the target sequence length (padded)
            L_s is the source sequence length (padded)
            E_q is the embedding size for query
            E_k is the embedding size for key
            E_v is the embedding size for value
            E_total is the embedding size for all heads combined
            E_out is the output embedding size
    Returns:
        torch.Tensor: Output of shape (N, L_t, E_out)
    """
    N = query.size(0)
    L_t = query.size(1)
    L_s = key.size(1)
    E_total = W_q.size(0)
    assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
    assert L_t == L_s, "This implementation assumes equal query and key sequence lengths"
    E_head = E_total // nheads

    # apply input projection
    # (N, L_t, E_q) -> (N, L_t, E_total)
    query = F.linear(query, W_q, b_q)
    # (N, L_s, E_k) -> (N, L_s, E_total)
    key = F.linear(key, W_k, b_k)
    # (N, L_s, E_v) -> (N, L_s, E_total)
    value = F.linear(value, W_v, b_v)

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head) -> (N * nheads, L_t, E_head)
    query = query.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    key = key.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    value = value.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)

    # query bmm key^T
    # (N * nheads, L_t, E_head) x (N * nheads, L_s, E_head)^T -> (N * nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    attn_weights = torch.bmm(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # Have to manipulate masks in order to apply them to the attention weights
    key_padding_mask = attn_mask_q.view(N, 1, 1, L_t).expand(-1, nheads, -1, -1).reshape(N*nheads, 1, L_t).to(device=device)
    attn_mask = torch.zeros(key_padding_mask.shape, device=device, dtype=torch.float32)
    attn_mask = attn_mask.masked_fill_(key_padding_mask, float("-inf"))

    # Zero out the attention weights where the mask is True by adding -inf prior to softmax
    attn_weights.add_(attn_mask)

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1).nan_to_num_(0.0)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights bmm value
    # (N * nheads, L_t, L_s) x (N * nheads, L_s, E_head) -> (N * nheads, L_t, E_head)
    attn_output = attn_weights.bmm(value)

    # merge heads
    # (N * nheads, L_t, E_head) -> (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.reshape(N, nheads, -1, E_head).transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    # padding-specific step: remove output projection bias from padded entries
    attn_output[attn_mask_q, :] = 0.0

    return attn_output
#%%
N = 512
E_q, E_k, E_v, E_total, E_out = 512, 512, 512, 512, 512
nheads = 8
dropout_p = 0.0
#%%
import numpy as np

def zipf_sentence_lengths(alpha: float, batch_size: int) -> np.ndarray:
    # generate fake corpus by unigram Zipf distribution
    # from wikitext-2 corpus, we get rank "." = 3, "!" = 386, "?" = 858
    sentence_lengths = np.empty(batch_size, dtype=int)
    for ibatch in range(batch_size):
        sentence_lengths[ibatch] = 1
        word = np.random.zipf(alpha)
        while word != 3 and word != 386 and word != 858:
            sentence_lengths[ibatch] += 1
            word = np.random.zipf(alpha)
    return sentence_lengths

alpha = 1.2

sentence_lengths = zipf_sentence_lengths(alpha, N)
L_t = np.max(sentence_lengths)
L_s = L_t
#%%
sentence_lengths
#%%
# create parameters
W_q, b_q = torch.randn((E_total, E_q), device=device), torch.randn(E_total, device=device)
W_k, b_k = torch.randn((E_total, E_k), device=device), torch.randn(E_total, device=device)
W_v, b_v = torch.randn((E_total, E_v), device=device), torch.randn(E_total, device=device)
W_out, b_out = torch.randn((E_out, E_total), device=device), torch.randn(E_out, device=device)

# create nested input
queries = []
keys = []
values = []
for i in range(N):
    l = sentence_lengths[i]
    s = l
    queries.append(torch.randn((l, E_q), device=device))
    keys.append(torch.randn((s, E_k), device=device))
    values.append(torch.randn((s, E_v), device=device))
query = torch.nested.nested_tensor(queries)
key = torch.nested.nested_tensor(keys)
value = torch.nested.nested_tensor(values)

# pad input
padded_query = torch.nested.to_padded_tensor(query, 0.0, (N, L_t, E_q))
padded_key   = torch.nested.to_padded_tensor(key, 0.0, (N, L_s, E_k))
padded_value = torch.nested.to_padded_tensor(value, 0.0, (N, L_s, E_v))

# create attention masks
attn_mask_q = torch.zeros((N, L_t), dtype=torch.bool)
attn_mask_kv = torch.zeros((N, L_s), dtype=torch.bool)

#  We need to mask out the padding entries in the attention weights.
for i, entry_length in enumerate(sentence_lengths):
    attn_mask_q[i, entry_length:] = True
    attn_mask_kv[i, entry_length:] = True
#%%
import timeit

t0 = timeit.default_timer()
out_nested = mha_nested(
    query, key, value, nheads,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)

t1 = timeit.default_timer()
out_padded = mha_padded(
    padded_query, padded_key, padded_value, nheads,
    attn_mask_q, attn_mask_kv,
    W_q, W_k, W_v, W_out,
    b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
    dropout_p=dropout_p)
t2 = timeit.default_timer()

print("nested and padded calculations differ by", (torch.nested.to_padded_tensor(out_nested, 0.0, (N, L_t, E_out)) - out_padded).abs().max().item())
print("nestedtensor multi-head attention takes", t1 - t0, "seconds")
print("padded tensor multi-head attention takes", t2 - t1, "seconds")
