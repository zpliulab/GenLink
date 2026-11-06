"""Load retained weights strictly, accounting for explicitly inactive branches."""
from collections import OrderedDict

REMOVED_LAYER_NAMES = frozenset({'W1', 'W2', 'mix_W1', 'mix_W2', 'edge_predictor', 'edge_proj', 'edge_proj_1'})
GENE_EMBEDDING_LAYERS = frozenset({'g_emb_linear', 'x_g_emb_mix'})
PEAK_EMBEDDING_LAYERS = frozenset({'p_emb_linear', 'p_a_emb_mix'})


def load_compatible_state_dict(model, state_dict, *, use_gene_embeddings=True, use_peak_embeddings=True):
    """Allow missing embedding weights only when that entire branch is inactive.

    No model code changes are needed: inactive branches already accept None.
    All supplied tensor shapes, active keys, and unexpected keys remain strict.
    """
    filtered = OrderedDict((key, value) for key, value in state_dict.items()
                           if key.split('.', 1)[0] not in REMOVED_LAYER_NAMES)
    inactive = set()
    if not use_gene_embeddings:
        inactive.update(GENE_EMBEDDING_LAYERS)
    if not use_peak_embeddings:
        inactive.update(PEAK_EMBEDDING_LAYERS)
    for key, value in model.state_dict().items():
        if key not in filtered and key.split('.', 1)[0] in inactive:
            filtered[key] = value
    if hasattr(state_dict, '_metadata'):
        filtered._metadata = state_dict._metadata
    return model.load_state_dict(filtered, strict=True)
