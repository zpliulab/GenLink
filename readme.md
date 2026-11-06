# GenLink

Evaluate pretrained models on simulated (`Sim`) or real (`Real`) data. Training code is retained in the project; `Tutorial.py` provides a testing entry point only.

## Environment Setup

Recommended environment: Python 3.10 and PyTorch 2.4.1. The commands below install PyTorch with CUDA 12.1 support, which requires a compatible NVIDIA driver:

```bash
conda create -n GenLink python=3.10 -y
conda activate GenLink
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.5.2 tqdm
```

For CPU use, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

The code automatically uses CUDA when available and otherwise runs on CPU. No torchvision, torchaudio, or additional embedding model downloads are required.

## Data and Weights

Keep the following directory structure. The selected dataset and embedding determine which data file and checkpoint are loaded:

```text
GenLink/
├── Tutorial.py
├── Datasets/
│   ├── SimTest.data
│   └── RealTest.pkl
└── pths/
    ├── Sim.pth
    ├── regnetwork_cellfm.pth
    ├── regnetwork_genecompass.pth
    ├── regnetwork_scbert.pth
    └── regnetwork_scgpt.pth
```

If the data are provided as `Datasets.rar`, extract the archive so that both data files are located directly in `Datasets/`, as shown above.

All four `Real` embedding options share `RealTest.pkl`, which already contains the required embeddings. The reference database for real data is `regnetwork`. `Sim` uses `SimTest.data` and `Sim.pth` and does not require an embedding option.

## Quickstart

Open a terminal in the project root and activate the environment.

Test simulated data:

```bash
python Tutorial.py --dataset Sim
```

Test real data with the selected embedding:

```bash
python Tutorial.py --dataset Real --embedding scgpt
```

Supported `--embedding` values: `cellfm`, `genecompass`, `scbert`, and `scgpt`.

Test only the first sample:

```bash
python Tutorial.py --dataset Real --embedding scgpt --max-samples 1
```

Optional arguments: `--ensemble` sets the number of sampling runs (default: `1`), and `--backward-steps` sets the number of sampling steps (default: `100`). All samples are tested unless `--max-samples` is specified.

To run directly from an IDE, edit the settings at the top of `Tutorial.py`:

```python
DATASET = 'Real'       # Sim / Real
EMBEDDING = 'scgpt'    # Used only for Real
MAX_SAMPLES = None    # None: all samples; 1: first sample only
ENSEMBLE = 1
BACKWARD_STEPS = 100
```

Running `python Tutorial.py` uses these settings; the current default is `Real + scgpt`. Project paths are resolved relative to the script, so preserve the directory structure when moving the project to another machine.
