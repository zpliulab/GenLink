"""Test the supplied Sim or Real data with pretrained weights."""
import argparse
import random

import numpy as np
import torch

from config import Config
from Dataset import GENE_EMBEDDINGS
from GenLink import Genlink

# 在 IDE 中直接修改这两个选项，再运行本文件。
DATASET = 'Real'          # 'Sim' 或 'Real'
EMBEDDING = 'scgpt'     # Real: cellfm / genecompass / scbert / scgpt
MAX_SAMPLES = None      # None 表示全部样本
ENSEMBLE = 1
BACKWARD_STEPS = 100
SEED = 42


def main():
    parser = argparse.ArgumentParser(description='GenLink pretrained-model testing')
    parser.add_argument('--dataset', choices=('Sim', 'Real'), default=DATASET)
    parser.add_argument('--embedding', choices=tuple(GENE_EMBEDDINGS), default=EMBEDDING)
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES)
    parser.add_argument('--ensemble', type=int, default=ENSEMBLE)
    parser.add_argument('--backward-steps', type=int, default=BACKWARD_STEPS)
    options = parser.parse_args()

    real = options.dataset == 'Real'
    args = Config('case' if real else 'simulation', embedding=options.embedding)
    args.modelCell = 'regnetwork' if real else 'simulation'
    args.max_samples = options.max_samples
    args.ensemble = options.ensemble
    args.backward_steps = options.backward_steps

    data_file = args.data_dir / ('RealTest.pkl' if real else 'SimTest.data')
    weight_file = args.pretrained_dir / (f'regnetwork_{options.embedding}.pth' if real else 'Sim.pth')
    print(f'Dataset: {options.dataset}; embedding: {options.embedding if real else "none"}')
    print(f'Data: {data_file}\nWeights: {weight_file}', flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)
    args.num_nodes = checkpoint['num_nodes']
    args.num_cells = args.Cnum = checkpoint['num_cells']
    trainer = Genlink(args)
    testdata = trainer.load_test_data(str(data_file))
    trainer.test(checkpoint, testdata)


if __name__ == '__main__':
    main()
