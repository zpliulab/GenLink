"""Dataset profiles; both profiles use the same unchanged DenoisingModel."""
from pathlib import Path


class Config:
    def __init__(self, dataset_type='simulation', embedding='cellfm'):
        if dataset_type not in ('case', 'simulation'):
            raise ValueError('dataset_type must be case or simulation.')
        self.dataset_type = dataset_type
        self.embedding = embedding
        self.project_dir = Path(__file__).resolve().parent
        self.data_dir = self.project_dir / 'Datasets'
        self.pretrained_dir = self.project_dir / 'pths'
        self.output_dir = self.project_dir / 'outputs' / dataset_type
        self.save_label = 'Simulation' if dataset_type == 'simulation' else 'CaseStudy'
        self.forward_steps = 200
        self.backward_steps = 100
        self.Cnum = 100
        self.use_pca = 'false'
        self.ensemble = 1
        self.max_region = 330 if dataset_type == 'simulation' else 2500
        self.num_epoch = 30
        self.batch_size = 1
        self.num_workers = 0
        self.max_samples = None
        self.LR = 1e-4
        self.save_interval = 10
        self.num_nodes = 99 if dataset_type == 'simulation' else 500
        self.num_cells = self.Cnum
        self.hidden_dim = 256
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.edge_types = 2
        self.set_output_dir(self.output_dir)

    def set_output_dir(self, directory):
        self.output_dir = Path(directory).resolve()
        self.save_dir = str(self.output_dir / 'models')

    def default_data_file(self, mode, cell):
        if self.dataset_type == 'simulation':
            return self.data_dir / ('Train.data' if mode == 'train' else 'SimTest.data')
        name = f'metacell_{mode}_dataset_0428_scbert.pkl'
        candidates = (self.data_dir / cell / name, self.data_dir / name)
        return next((path for path in candidates if path.is_file()), candidates[0])

    def default_checkpoint_file(self, cell):
        if self.dataset_type == 'simulation':
            return self.pretrained_dir / 'Sim.pth'
        name = f'GenLink_{cell}_CellFM_emb_Reg_GRN.pth'
        return self.pretrained_dir / name
