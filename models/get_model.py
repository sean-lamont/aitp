from models.transformer_encoder_model import TransformerEmbedding
from models import gnn_edge_labels, inner_embedding_network
from models.graph_transformers.SAT.sat.layers import AttentionRelations
from models.graph_transformers.SAT.sat.models import GraphTransformer, AMRTransformer

'''
Utility function to fetch model given a configuration dict
'''
def get_model(model_config):
    if model_config['model_type'] == 'sat':
        return GraphTransformer(in_size=model_config['vocab_size'],
                                d_model=model_config['embedding_dim'],
                                dim_feedforward=model_config['dim_feedforward'],
                                num_heads=model_config['num_heads'],
                                num_layers=model_config['num_layers'],
                                in_embed=model_config['in_embed'],
                                se=model_config['se'],
                                abs_pe=model_config['abs_pe'],
                                abs_pe_dim=model_config['abs_pe_dim'],
                                use_edge_attr=model_config['use_edge_attr'],
                                num_edge_features=model_config['num_edge_features'],
                                dropout=model_config['dropout'],
                                k_hop=model_config['gnn_layers'])

    if model_config['model_type'] == 'amr':
        return AMRTransformer(in_size=model_config['vocab_size'],
                              d_model=model_config['embedding_dim'],
                              dim_feedforward=model_config['dim_feedforward'],
                              num_heads=model_config['num_heads'],
                              num_layers=model_config['num_layers'],
                              in_embed=model_config['in_embed'],
                              abs_pe=model_config['abs_pe'],
                              abs_pe_dim=model_config['abs_pe_dim'],
                              use_edge_attr=model_config['use_edge_attr'],
                              num_edge_features=model_config['num_edge_features'],
                              dropout=model_config['dropout'],
                              layer_norm=model_config['layer_norm'],
                              global_pool=model_config['global_pool'],
                              device=model_config['device']
                              )

    elif model_config['model_type'] == 'formula-net':
        return inner_embedding_network.FormulaNet(model_config['vocab_size'],
                                                  model_config['embedding_dim'],
                                                  model_config['gnn_layers'])

    elif model_config['model_type'] == 'formula-net-edges':
        return gnn_edge_labels.message_passing_gnn_edges(model_config['vocab_size'],
                                                         model_config['embedding_dim'],
                                                         model_config['gnn_layers'])

    elif model_config['model_type'] == 'digae':
        return None

    elif model_config['model_type'] == 'classifier':
        return None

    elif model_config['model_type'] == 'transformer':
        return TransformerEmbedding(ntoken=model_config['vocab_size'],
                                    d_model=model_config['embedding_dim'],
                                    nhead=model_config['num_heads'],
                                    nlayers=model_config['num_layers'],
                                    dropout=model_config['dropout'],
                                    d_hid=model_config['dim_feedforward'])

    elif model_config['model_type'] == 'transformer_relation':
        return AttentionRelations(ntoken=model_config['vocab_size'],
                                  # global_pool=False,
                                  embed_dim=model_config['embedding_dim'])
    else:
        return None
