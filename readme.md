# GenLink

使用预训练权重测试模拟数据（`Sim`）或真实数据（`Real`）。训练代码保留在项目中，`Tutorial.py` 仅提供测试入口。

## 环境配置

建议使用 Python 3.10、PyTorch 2.4.1。以下命令安装 CUDA 12.1 版 PyTorch，需 NVIDIA 驱动支持：

```bash
conda create -n GenLink python=3.10 -y
conda activate GenLink
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.5.2 tqdm
```

如仅使用 CPU，将上述 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

代码会自动选择可用的 CUDA GPU，否则使用 CPU。无需安装 torchvision、torchaudio 或额外下载嵌入模型。

## 数据与权重

保持以下文件结构，运行时会根据选项自动匹配数据和权重：

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

`Real` 的四种嵌入共用 `RealTest.pkl`，其中已包含所需嵌入；真实数据的参考数据库统一为 `regnetwork`。`Sim` 使用 `SimTest.data` 和 `Sim.pth`，无需选择嵌入。

## Quickstart

在项目根目录打开终端，并激活上述环境。

测试模拟数据：

```bash
python Tutorial.py --dataset Sim
```

测试真实数据，选择对应嵌入：

```bash
python Tutorial.py --dataset Real --embedding scgpt
```

`--embedding` 支持 `cellfm`、`genecompass`、`scbert`、`scgpt`。

先测试一个样本：

```bash
python Tutorial.py --dataset Real --embedding scgpt --max-samples 1
```

可选参数：`--ensemble` 为采样次数，默认 `1`；`--backward-steps` 为采样步数，默认 `100`；不指定 `--max-samples` 时测试全部样本。

也可以在 IDE 中直接修改 `Tutorial.py` 顶部的配置后运行：

```python
DATASET = 'Real'       # Sim / Real
EMBEDDING = 'scgpt'    # 仅用于 Real
MAX_SAMPLES = None    # None：全部；1：仅第一个样本
ENSEMBLE = 1
BACKWARD_STEPS = 100
```

直接执行 `python Tutorial.py` 使用文件顶部的配置；当前默认是 `Real + scgpt`。项目路径根据脚本位置自动定位，迁移服务器时保留完整目录即可。
