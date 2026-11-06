"""Load simulation matrices or the shared Real dataset with selected embeddings."""
from functools import partial
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

GENE_EMBEDDINGS = {
    'cellfm': 'cellfm_gene_emb',
    'genecompass': 'gene_embeddings',
    'scbert': 'scbert_gene_emb',
    'scgpt': 'scgpt_gene_emb',
}


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if np.__version__.startswith('1.') and module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def normalize_sample(sample, dataset_type='case', embedding='cellfm'):
    if isinstance(sample, (list, tuple)):
        sample = dict(zip(*sample))
    keys = ('grn', 'gene_expr', 'atac_expr', 'rg', 'rtf')
    fields = (('GRN_Matrix', 'Gene_Expression', 'ATAC_Expression',
               'Region_Gene_Matrix', 'Region_TF_Matrix')
              if dataset_type == 'simulation' else keys)
    result = {}
    for key, field in zip(keys, fields):
        value = sample[field]
        result[key] = np.asarray(value.toarray() if hasattr(value, 'toarray') else value)
    result['g_emb'] = np.asarray(sample[GENE_EMBEDDINGS[embedding]]) if dataset_type == 'case' else None
    result['p_emb'] = np.asarray(sample['peak_embeddings']) if dataset_type == 'case' else None
    return result


class GRNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        keys = ('grn', 'gene_expr', 'atac_expr', 'rg', 'rtf', 'g_emb', 'p_emb')
        return tuple(torch.from_numpy(sample[key]) if sample[key] is not None else None for key in keys)


def grn_collate_fn(batch, max_n, max_r):
    batch_size = len(batch)
    gene_dim, atac_dim = batch[0][1].shape[1], batch[0][2].shape[1]
    grn_batch = torch.zeros(batch_size, max_n, max_n)
    gene_batch = torch.zeros(batch_size, max_n, gene_dim)
    atac_batch = torch.zeros(batch_size, max_r, atac_dim)
    rg_batch = torch.zeros(batch_size, max_r, max_n)
    rtf_batch = torch.zeros(batch_size, max_r, max_n)
    gemb_batch = torch.zeros(batch_size, max_n, batch[0][5].shape[1]) if batch[0][5] is not None else None
    pemb_batch = torch.zeros(batch_size, max_r, batch[0][6].shape[1]) if batch[0][6] is not None else None
    node_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    region_mask = torch.zeros(batch_size, max_r, dtype=torch.bool)
    for i, (grn, gene, atac, rg, rtf, g_emb, p_emb) in enumerate(batch):
        n, r = gene.shape[0], atac.shape[0]
        if n > max_n or r > max_r:
            raise ValueError('Sample exceeds the configured padding limits.')
        grn_batch[i, :n, :n] = grn
        gene_batch[i, :n] = gene
        atac_batch[i, :r] = atac
        rg_batch[i, :r, :n] = rg
        rtf_batch[i, :r, :n] = rtf
        if gemb_batch is not None:
            gemb_batch[i, :n] = g_emb
        if pemb_batch is not None:
            pemb_batch[i, :r] = p_emb
        node_mask[i, :n] = True
        region_mask[i, :r] = True
    # Model input order places rtf before rg.
    return grn_batch, gene_batch, atac_batch, rtf_batch, rg_batch, node_mask, region_mask, gemb_batch, pemb_batch


class GRNDataLoader:
    def __init__(self, filename, batch_size=1, max_num_nodes=None, max_num_regions=None,
                 shuffle=False, dataset_type='case', max_samples=None, num_workers=0,
                 embedding='cellfm'):
        print(f'Loading {dataset_type} dataset from {filename}...')
        with open(filename, 'rb') as handle:
            raw = _NumpyCompatibleUnpickler(handle).load()
        if isinstance(raw, dict):
            raw = [raw]
        self.total_samples = len(raw)
        if max_samples is not None:
            raw = raw[:max_samples]
        data = [normalize_sample(sample, dataset_type, embedding) for sample in raw]
        first = data[0]
        self.num_cells = first['gene_expr'].shape[1]
        self.has_gene_embeddings = first['g_emb'] is not None
        self.has_peak_embeddings = first['p_emb'] is not None
        self.gene_embedding_dim = first['g_emb'].shape[1] if self.has_gene_embeddings else 1536
        self.max_num_nodes = max_num_nodes if max_num_nodes is not None else max(s['grn'].shape[0] for s in data)
        self.max_num_regions = max_num_regions if max_num_regions is not None else max(s['atac_expr'].shape[0] for s in data)
        self.edge_percent = float(np.mean([
            sample['grn'].sum() / max(sample['grn'].shape[0] * (sample['grn'].shape[0] - 1), 1)
            for sample in data
        ]))
        self.sample_count = len(data)
        self.dataset_type = dataset_type
        self.dataloader = DataLoader(
            GRNDataset(data), batch_size=batch_size, shuffle=shuffle, drop_last=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=partial(grn_collate_fn, max_n=self.max_num_nodes, max_r=self.max_num_regions))
        print(f'Samples: {self.sample_count}/{self.total_samples}; '
              f'embedding: {embedding if self.has_gene_embeddings else "none"}; '
              f'padding: {self.max_num_nodes} genes, {self.max_num_regions} regions')

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
