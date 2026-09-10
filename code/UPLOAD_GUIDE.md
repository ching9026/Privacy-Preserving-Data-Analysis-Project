# GitHub Upload / Reproduction Guide

## Why the original archive should not be uploaded

The supplied `private_final.zip` is actually a **gzip-compressed tar archive**
with an incorrect `.zip` extension. Its compressed size is about 196 MB and
the extracted project is about 679 MB.

Most of that size comes from local traffic datasets (`dataset.npy`) rather
than source code.

This cleaned package intentionally excludes:

- `dataset/*/*.npy`
- the nested `.git/` directory
- `__pycache__/` and `.pyc`
- PyCharm `.idea/`

The source code can therefore be uploaded to GitHub normally.

## Upstream project

This experiment is based on:

https://github.com/ZYuSdu/pFedCTP

The embedded Git repository in the supplied archive pointed to that upstream
repository. Keep the upstream citation/attribution in the project README.

## Experiment files added for this project

The supplied working directory contains the following additional experiment
files that were not tracked by the embedded upstream Git checkout:

- `run_pFedCTP_ES.py`
- `clients/target_client_es.py`
- `trainers/pFedCTP_ES.py`
- `utils/data_utils_es.py`
- `utils/early_stopping.py`

These files implement the target-city fine-tuning experiments, including
validation-based early stopping and several fine-tuning modes.

## Datasets

The original pFedCTP README points to the ST-GFSL project for the datasets:

https://github.com/RobinLu1209/ST-GFSL

Expected layout:

```text
dataset/
├── metr-la/
│   ├── dataset.npy
│   └── matrix.npy
├── pems-bay/
│   ├── dataset.npy
│   └── matrix.npy
├── shenzhen/
│   ├── dataset.npy
│   └── matrix.npy
└── chengdu/
    ├── dataset.npy
    └── matrix.npy
```

Do not commit these large `.npy` files into a normal Git repository.

## Environment

The upstream project documents:

- Python >= 3.8
- PyTorch 1.13.0
- NumPy 1.25.2
- tqdm
- scikit-learn
- SciPy

This project also imports `torch_geometric`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch / PyTorch Geometric installation may need to be adjusted to match
your CUDA version.

## Example experiment

From the project root, after placing the datasets in `dataset/`:

```bash
python run_pFedCTP_ES.py \
  --algo=pFedCTP \
  --batch_size=32 \
  --target_city=shenzhen \
  --num_rounds=90 \
  --local_epochs=150 \
  --target_epochs=50 \
  --gcn_layers=1 \
  --ft_mode=private_only
```

Available fine-tuning modes in `run_pFedCTP_ES.py` include:

- `full`
- `early_stopping`
- `regularized`
- `predictor_only`
- `private_only`
- `staged_reg_es`

## Recommended GitHub approach

Commit the source code and documentation normally.

For datasets, prefer documenting where to download them. If you absolutely
must distribute large files yourself, use Git LFS or a GitHub Release rather
than committing them to normal Git history.
