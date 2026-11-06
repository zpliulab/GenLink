from checkpoint import load_compatible_state_dict
from pathlib import Path
from Dataset import GRNDataLoader
from tqdm import tqdm
from datetime import datetime
import torch
from flowmatching_pytorch import FlowMatching1D
from models.GenLink import DenoisingModel
from discrete.flowmatching_utils import Evaluation
import numpy as np
import warnings
import os

class Genlink:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.num_nodes = args.num_nodes
        self.num_cells = args.num_cells
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.edge_types = args.edge_types
        self.max_region = args.max_region
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.lr = args.LR
        self.save_interval = args.save_interval
        self.forward_steps = args.forward_steps
        self.backward_steps = args.backward_steps
        self.use_PCA = int(args.use_pca) if args.use_pca.lower() != 'false' else False
        self.Cnum = args.Cnum
        self.ensemble = args.ensemble
        self.save_dir = args.save_dir
        self.use_gene_embeddings = args.dataset_type == 'case'
        self.use_peak_embeddings = args.dataset_type == 'case'
        self.setup_paths()

    def load_pre_model(self, flowmatching_pre):
        num_nodes = flowmatching_pre['num_nodes']
        input_dims = flowmatching_pre['num_cells']
        num_layers = flowmatching_pre['num_layers']
        hidden_dims = flowmatching_pre['hidden_dims']
        num_heads = flowmatching_pre['num_heads']
        self.model = DenoisingModel(num_nodes=num_nodes, num_cells=input_dims, hidden_dim=hidden_dims, num_heads=num_heads, num_layers=num_layers)
        if self.use_gene_embeddings:
            gene_dim = flowmatching_pre['model_state_dict']['g_emb_linear.weight'].shape[1]
            self.model.g_emb_linear = torch.nn.Linear(gene_dim, hidden_dims)
        self.model = self.model.to(self.device)
        for key, enabled in (('use_gene_embeddings', self.use_gene_embeddings),
                             ('use_peak_embeddings', self.use_peak_embeddings)):
            if enabled and flowmatching_pre.get(key) is False:
                raise ValueError(f'The checkpoint was trained with {key}=False, but the test data enables it.')
        load_compatible_state_dict(self.model, flowmatching_pre['model_state_dict'],
                                   use_gene_embeddings=self.use_gene_embeddings,
                                   use_peak_embeddings=self.use_peak_embeddings)
        flowmatching = FlowMatching1D(self.model, device=self.device, max_num_nodes=flowmatching_pre['num_nodes'], forward_steps=self.forward_steps, backward_steps=self.backward_steps, edge_percent=flowmatching_pre['edge_percent'])
        flowmatching = flowmatching.to(self.device)
        return flowmatching

    def setup_paths(self):
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.checkpoint_dir = str(Path(self.args.output_dir) / 'checkpoints' / self.args.save_label)

    def load_test_data(self, test_filename):
        dataloader = self._load_dataset(test_filename, batch_size=1)
        if dataloader.num_cells != self.num_cells:
            raise ValueError(f'Test expression dimension {dataloader.num_cells} does not match checkpoint dimension {self.num_cells}.')
        return dataloader

    def load_data(self, train_filename):
        dataloader = self._load_dataset(train_filename, batch_size=self.batch_size)
        self.edge_percent = dataloader.edge_percent
        self.num_cells = self.Cnum = dataloader.num_cells
        return dataloader

    def _load_dataset(self, filename, batch_size):
        dataloader = GRNDataLoader(
            filename=filename, batch_size=batch_size, max_num_nodes=self.num_nodes,
            max_num_regions=self.max_region, dataset_type=self.args.dataset_type,
            max_samples=self.args.max_samples, num_workers=self.args.num_workers,
            embedding=self.args.embedding)
        self.use_gene_embeddings = dataloader.has_gene_embeddings
        self.use_peak_embeddings = dataloader.has_peak_embeddings
        self.gene_embedding_dim = dataloader.gene_embedding_dim
        return dataloader

    def setup_model(self):
        X = self.use_PCA if self.use_PCA else self.Cnum
        self.input_dims = {'X': X, 'E': 2, 'y': 1}
        self.output_dims = {'X': X, 'E': 2, 'y': 1}
        self.model = DenoisingModel(num_nodes=self.num_nodes, num_cells=self.Cnum, hidden_dim=self.hidden_dim, num_heads=self.num_heads, num_layers=self.num_layers, dropout=self.dropout, edge_types=self.edge_types)
        if self.use_gene_embeddings:
            self.model.g_emb_linear = torch.nn.Linear(self.gene_embedding_dim, self.hidden_dim)
        self.flowmatching = FlowMatching1D(self.model, device=self.device, max_num_nodes=self.num_nodes, forward_steps=self.forward_steps, backward_steps=self.backward_steps, edge_percent=self.edge_percent)
        self.model = self.model.to(self.device)
        self.flowmatching = self.flowmatching.to(self.device)

    def save_model(self, filename):
        self.checkpoint = {'num_nodes': self.num_nodes, 'num_cells': self.num_cells, 'input_dims': self.input_dims, 'hidden_dims': self.hidden_dim, 'num_heads': self.num_heads, 'num_layers': self.num_layers, 'output_dims': self.output_dims, 'model_state_dict': self.model.state_dict(), 'edge_percent': self.edge_percent}
        self.checkpoint.update(dataset_type=self.args.dataset_type,
                               use_gene_embeddings=self.use_gene_embeddings,
                               use_peak_embeddings=self.use_peak_embeddings)
        torch.save(self.checkpoint, filename)
        print('saved model at ' + filename)

    def train(self, train_filename):
        """Train the same model on normalized case or simulation samples."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.save_dir is not None:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        dataloader = self.load_data(train_filename)
        self.setup_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        self.scaler = torch.cuda.amp.GradScaler()
        print('Start training...')
        pbar = tqdm(range(self.num_epoch), ncols=100)
        self.best_mean_AUC = 0
        self.best_mean_AUC_file = []
        for epoch in pbar:
            self.flowmatching.train()
            self.model.train()
            total_loss = []
            total_AUC = []
            for idx, batch_x in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss, Train_AUC = self.flowmatching(batch_x, epoch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss.append(loss.item())
                total_AUC.append(Train_AUC)
                pbar.set_description(f'Epoch: {epoch + 1:4.0f} - Train_AUC: {Train_AUC:.4f} - loss: {loss.item():.4f} ')
            total_loss = np.mean(total_loss)
            total_AUC = np.mean(total_AUC)
            if (epoch + 1) % 1 == 0:
                print('  --- Epoch %d average Loss: %.4f mean Train AUC: %.4f  lr: %0.6f' % (epoch + 1, total_loss, total_AUC, self.optimizer.param_groups[0]['lr']))
            if (epoch + 1) % self.save_interval == 0:
                filename = self.checkpoint_dir + f'_{self.args.modelCell}_{self.timestamp}_epoch{epoch + 1}.pth'
                self.save_model(filename)
        if self.save_dir is not None:
            filename = self.args.save_filename
            self.save_model(os.path.join(self.save_dir, filename))
        if not self.best_mean_AUC_file:
            self.best_mean_AUC_file = filename
        print(f'  --- Best AUC is {self.best_mean_AUC}')
        print(f'  --- Best best_AUC_file is {self.best_mean_AUC_file}')
        if not hasattr(self, 'printf'):
            self.printf = None
        return (self.best_mean_AUC, self.best_mean_AUC_file, self.printf)

    def test(self, flowmatching_pre, testdata):
        if hasattr(testdata, 'has_gene_embeddings'):
            self.use_gene_embeddings = testdata.has_gene_embeddings
            self.use_peak_embeddings = testdata.has_peak_embeddings
        self.flowmatching = self.load_pre_model(flowmatching_pre)
        self.flowmatching.eval()
        all_adj, performance, case_E = self.flowmatching.test_step(testdata, self.ensemble)
        return (all_adj, case_E)

def evaluation(self, adj_final, truelabel):
    performance = Evaluation(y_pred=np.array(adj_final).flatten(), y_true=truelabel.flatten())
    print(adj_final.shape)
    print(f"GenLink*-AUROC: {performance['AUROC']:.4f} AUPRC: {performance['AUPRC']:.4f} F1-score: {performance['F1-score']:.4f} precision: {performance['precision']:.4f}")
    return performance
warnings.filterwarnings('ignore')
