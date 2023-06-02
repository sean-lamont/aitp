from typing import Optional
import einops
import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh

'''

Code for Magnetic Laplacian (torch_geometric_signed_directed.utils.directed.get_magnetic_Laplacian)

'''


def get_magnetic_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                           normalization: Optional[str] = 'sym',
                           dtype: Optional[int] = None,
                           num_nodes: Optional[int] = None,
                           q: Optional[float] = 0.25,
                           return_eig: bool = False,
                           k=27):
    r""" Computes the magnetic Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -

            1. :obj:`None`: No normalization :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`

            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`

        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)

    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes, "add")

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym / 2

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_sym^{-1/2} A_sym D_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * \
                      edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_eig:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        # print (f" Shape {L.shape}")
        k_ = min(k, L.shape[0])
        # print (k, L.shape[0])
        eig_vals, eig_vecs = eigsh(L, k=k_- 2, which='LM', return_eigenvectors=True)
        eig_vals = torch.FloatTensor(eig_vals)
        eig_real = torch.FloatTensor(eig_vecs.real)
        eig_imag = torch.FloatTensor(eig_vecs.imag)

        if k_ < k - 2:
            eig_real = torch.nn.functional.pad(eig_real, (0, k - 2 - k_), value=0)
            eig_vals = torch.nn.functional.pad(eig_vals, (0, k - 2 - k_), value=0)
            eig_imag = torch.nn.functional.pad(eig_imag, (0, k - 2 - k_), value=0)

        # shape n,k for vecs, shape k for vals
        #todo pad out to k
        return eig_vals, (eig_real, eig_imag)
        # lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        # lambda_max = float(lambda_max.real)
        # return edge_index, edge_weight.real, edge_weight.imag, lambda_max


'''

Code for MagLapNet in Torch

'''


class MagLapNet(torch.nn.Module):
    def __init__(self,
                 eig_dim,
                 d_hidden: int = 32,
                 d_embed: int = 256,
                 num_heads: int = 4,
                 n_layers: int = 1,
                 dropout_p: float = 0.2,
                 # use_gnn = False,
                 return_real_output: bool = True,
                 consider_im_part: bool = True,
                 use_signnet: bool = True,
                 use_attention: bool = False,
                 concatenate_eigenvalues: bool = False,
                 norm=True,
                 ):

        super().__init__()

        self.concatenate_eigenvalues = concatenate_eigenvalues
        self.consider_im_part = consider_im_part
        self.use_signnet = use_signnet
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.element_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * eig_dim, d_hidden) if self.consider_im_part else torch.nn.Linear(eig_dim, d_hidden),
            torch.nn.ReLU()
        )

        # d_hidden + 1 for eig_val
        self.re_aggregate_mlp = torch.nn.Sequential(
            torch.nn.Linear(eig_dim * (d_hidden + 1), d_embed),
            torch.nn.ReLU()
        )

        self.attn = torch.nn.MultiheadAttention(embed_dim=d_embed)

        self.im_aggregate_mlp = None

        if not return_real_output and self.consider_im_part:
            self.im_aggregate_mlp = torch.nn.Sequential(
                torch.nn.Linear(eig_dim * (d_hidden + 1), d_embed),
                torch.nn.ReLU()
            )

        if norm:
            self.norm = torch.nn.LayerNorm(eig_dim * (d_hidden + 1))
        else:
            self.norm = None

    def forward(self, eigenvalues,
                eigenvectors):

        # from paper
       ###############################################################################################################
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e_real(e)

        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float()  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc)  # (Num nodes) x (Num Eigenvectors) x 2

        PosEnc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0, 1).float()  # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc)  # (Num Eigenvectors) x (Num nodes) x PE_dim

        # 1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:, :, 0])

        # remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')

        # Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)

        # Concatenate learned PE to input embedding
        h = torch.cat((h, PosEnc), 1)


       ###############################################################################################################


        mask = (eigenvalues == 0).bool()

        trans_eig = eigenvectors[0]
        trans_eig_im = eigenvectors[1]

        if self.consider_im_part:
            trans_eig = torch.cat([trans_eig, trans_eig_im], dim=-1)

        # padding_mask = (eigenvalues > 0)[..., None, :]
        # padding_mask = padding_mask.at[..., 0].set(True)
        # attn_padding_mask = padding_mask[..., None] & padding_mask[..., None, :]

        trans = self.element_mlp(trans_eig)
        if self.use_signnet:
            trans += self.element_mlp(-trans_eig)

        # if self.concatenate_eigenvalues:
        #   eigenvalues_ = jnp.broadcast_to(eigenvalues[..., None, :],
        #                                   trans.shape[:-1])
        #   trans = jnp.concatenate((eigenvalues_[..., None], trans), axis=-1)

        eigenvalues = einops.repeat(eigenvalues, "k -> k 1")

        if self.concatenate_eigenvalues:
            eigenvalues_ = einops.repeat(eigenvalues, "k 1-> n k 1", n=trans.shape[0])
            trans = torch.cat([eigenvalues_, trans], dim=-1)

        # trans = einops.rearrange(trans, "n k d -> n (k d)")

        if self.use_attention:
            if self.norm is not None:
                trans = self.norm(trans)

        # attn = MultiHeadAttention(
        #     self.num_heads,
        #     key_size=trans.shape[-1] // self.num_heads,
        #     value_size=trans.shape[-1] // self.num_heads,
        #     model_size=trans.shape[-1],
        #     w_init=None,
        #     dropout_p=self.dropout_p,
        #     with_bias=False)
        #
        # trans += attn(
        #     trans,
        #     trans,
        #     trans,
        #     mask=attn_padding_mask,
        #     is_training=call_args.is_training)

        # padding_mask = padding_mask[..., None]
        # trans = trans * padding_mask
        # trans = trans.reshape(trans.shape[:-2] + (-1,))

        # if self.dropout_p and call_args.is_training:
        #   trans = hk.dropout(hk.next_rng_key(), self.dropout_p, trans)

        output = self.re_aggregate_mlp(trans)
        if self.im_aggregate_mlp is None:
            return output

        # output_im = self.im_aggregate_mlp(trans)
        # output = output + 1j * output_im
        # return output
#
# # class MagLapNet(hk.Module):
# #   """For the Magnetic Laplacian's or Combinatorial Laplacian's eigenvectors.
# #
# #     Args:
# #       d_model_elem: Dimension to map each eigenvector.
# #       d_model_aggr: Output dimension.
# #       num_heads: Number of heads for optional attention.
# #       n_layers: Number of layers for MLP/GNN.
# #       dropout_p: Dropout for attenion as well as eigenvector embeddings.
# #       activation: Element-wise non-linearity.
# #       return_real_output: True for a real number (otherwise complex).
# #       consider_im_part: Ignore the imaginary part of the eigenvectors.
# #       use_signnet: If using the sign net idea f(v) + f(-v).
# #       use_gnn: If True use GNN in signnet, otherwise MLP.
# #       use_attention: If true apply attention between eigenvector embeddings for
# #         same node.
# #       concatenate_eigenvalues: If True also concatenate the eigenvalues.
# #       norm: Optional norm.
# #       name: Name of the layer.
# #   """
# #
# #   def __init__(self,
# #                d_model_elem: int = 32,
# #                d_model_aggr: int = 256,
# #                num_heads: int = 4,
# #                n_layers: int = 1,
# #                dropout_p: float = 0.2,
# #                activation: Callable[[Tensor], Tensor] = jax.nn.relu,
# #                return_real_output: bool = True,
# #                consider_im_part: bool = True,
# #                use_signnet: bool = True,
# #                use_gnn: bool = False,
# #                use_attention: bool = False,
# #                concatenate_eigenvalues: bool = False,
# #                norm: Optional[Any] = None,
# #                name: Optional[str] = None):
# #     super().__init__(name=name)
# #     self.concatenate_eigenvalues = concatenate_eigenvalues
# #     self.consider_im_part = consider_im_part
# #     self.use_signnet = use_signnet
# #     self.use_gnn = use_gnn
# #     self.use_attention = use_attention
# #     self.num_heads = num_heads
# #     self.dropout_p = dropout_p
# #     self.norm = norm
# #
# #     if self.use_gnn:
# #       self.element_gnn = GNN(
# #           int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
# #           gnn_type='gnn',
# #           k_hop=n_layers,
# #           mlp_layers=n_layers,
# #           activation=activation,
# #           use_edge_attr=False,
# #           concat=True,
# #           residual=False,
# #           name='re_element')
# #     else:
# #       self.element_mlp = MLP(
# #           int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
# #           n_layers=n_layers,
# #           activation=activation,
# #           with_norm=False,
# #           final_activation=True,
# #           name='re_element')
# #
# #     self.re_aggregate_mlp = MLP(
# #         d_model_aggr,
# #         n_layers=n_layers,
# #         activation=activation,
# #         with_norm=False,
# #         final_activation=True,
# #         name='re_aggregate')
# #
# #     self.im_aggregate_mlp = None
# #     if not return_real_output and self.consider_im_part:
# #       self.im_aggregate_mlp = MLP(
# #           d_model_aggr,
# #           n_layers=n_layers,
# #           activation=activation,
# #           with_norm=False,
# #           final_activation=True,
# #           name='im_aggregate')
# #
# #   def __call__(self, graph: jraph.GraphsTuple, eigenvalues: Tensor,
# #                eigenvectors: Tensor, call_args: CallArgs) -> Tensor:
# #     padding_mask = (eigenvalues > 0)[..., None, :]
# #     padding_mask = padding_mask.at[..., 0].set(True)
# #     attn_padding_mask = padding_mask[..., None] & padding_mask[..., None, :]
# #
# #     trans_eig = jnp.real(eigenvectors)[..., None]
# #
# #     if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
# #       trans_eig_im = jnp.imag(eigenvectors)[..., None]
# #       trans_eig = jnp.concatenate((trans_eig, trans_eig_im), axis=-1)
# #
# #     if self.use_gnn:
# #       trans = self.element_gnn(
# #           graph._replace(nodes=trans_eig, edges=None), call_args).nodes
# #       if self.use_signnet:
# #         trans_neg = self.element_gnn(
# #             graph._replace(nodes=-trans_eig, edges=None), call_args).nodes
# #         trans += trans_neg
# #     else:
# #       trans = self.element_mlp(trans_eig)
# #       if self.use_signnet:
# #         trans += self.element_mlp(-trans_eig)
# #
# #     if self.concatenate_eigenvalues:
# #       eigenvalues_ = jnp.broadcast_to(eigenvalues[..., None, :],
# #                                       trans.shape[:-1])
# #       trans = jnp.concatenate((eigenvalues_[..., None], trans), axis=-1)
# #
# #     if self.use_attention:
# #       if self.norm is not None:
# #         trans = self.norm()(trans)
# #       attn = MultiHeadAttention(
# #           self.num_heads,
# #           key_size=trans.shape[-1] // self.num_heads,
# #           value_size=trans.shape[-1] // self.num_heads,
# #           model_size=trans.shape[-1],
# #           w_init=None,
# #           dropout_p=self.dropout_p,
# #           with_bias=False)
# #       trans += attn(
# #           trans,
# #           trans,
# #           trans,
# #           mask=attn_padding_mask,
# #           is_training=call_args.is_training)
# #
# #     padding_mask = padding_mask[..., None]
# #     trans = trans * padding_mask
# #     trans = trans.reshape(trans.shape[:-2] + (-1,))
# #
# #     if self.dropout_p and call_args.is_training:
# #       trans = hk.dropout(hk.next_rng_key(), self.dropout_p, trans)
# #
# #     output = self.re_aggregate_mlp(trans)
# #     if self.im_aggregate_mlp is None:
# #       return output
# #
# #     output_im = self.im_aggregate_mlp(trans)
# #     output = output + 1j * output_im
# #     return output
