from models.transformer.transformer_encoder_model import TransformerEmbedding
from models.gnn.formula_net.formula_net import FormulaNet, FormulaNetEdges
from models.relation_transformer.relation_transformer_new import AttentionRelations
from models.gnn.digae.digae_model import DigaeEmbedding
from models.sat.models import GraphTransformer
from models.amr.amr import AMRTransformer


'''
Utility function to fetch model given a configuration dict
'''
def get_model(model_config):
    if model_config['model_type'] == 'sat':
        return GraphTransformer(in_size=model_config['vocab_size'],
                                num_class=2,
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
                              num_edge_features=200,#model_config['num_edge_features'],
                              dropout=model_config['dropout'],
                              layer_norm=True,#model_config['layer_norm'],
                              global_pool=True,#model_config['global_pool'],
                              )

    elif model_config['model_type'] == 'formula-net':
        return FormulaNet(model_config['vocab_size'],
                                                  model_config['embedding_dim'],
                                                  model_config['gnn_layers'])

    elif model_config['model_type'] == 'formula-net-edges':
        return FormulaNetEdges(input_shape=model_config['vocab_size'],
                                        embedding_dim=model_config['embedding_dim'],
                                               num_iterations=model_config['gnn_layers'])

    elif model_config['model_type'] == 'digae':
        return DigaeEmbedding(model_config['vocab_size'],
            model_config['embedding_dim'],
            model_config['embedding_dim'],
            model_config['embedding_dim'])


    elif model_config['model_type'] == 'transformer':
        return TransformerEmbedding(ntoken=model_config['vocab_size'],
                                    d_model=model_config['embedding_dim'],
                                    nhead=model_config['num_heads'],
                                    nlayers=model_config['num_layers'],
                                    dropout=model_config['dropout'],
                                    d_hid=model_config['dim_feedforward'])

    elif model_config['model_type'] == 'transformer_relation':
        return AttentionRelations(ntoken=model_config['vocab_size'],
                                  dropout=model_config['dropout'],
                                  num_heads=model_config['num_heads'],
                                  num_layers=model_config['num_layers'],
                                  embed_dim=model_config['embedding_dim'])

    elif model_config['model_type'] == 'classifier':
        raise NotImplementedError
    else:
        return None
