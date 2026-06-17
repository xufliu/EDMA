# EDMA

Code for **Energy-based Discrete Mask Approximation (EDMA)**, the method
introduced in:

> Confidence-Aware Explanations for 3D Molecular Graphs via Energy-Based Masking

This repository focuses on the QM9 experiments with PyTorch Geometric backbones:

- SchNet
- DimeNet and DimeNet++

The implementation explains pretrained QM9 models by optimizing node-level
energy masks. Edge masks are induced from node masks through source-destination
mask products, matching the dense 3D molecular graph setting used in the paper.

## Project Layout

```text
edma/
  data/                 QM9 dataset helpers
  evaluation/           top-k and paper-format evaluation utilities
  explainers/           EDMA explainers for SchNet and DimeNet/DimeNet++
  models/               pretrained PyG QM9 model loaders
scripts/
  qm9_schnet_energy.py    SchNet QM9 evaluation
  qm9_dimenet_energy.py   DimeNet/DimeNet++ QM9 evaluation
```

Legacy root-level imports such as `energy_explainer_instance_g.py` can be kept as
thin compatibility wrappers, but the maintained implementation lives under
`edma/`.

## Installation

Create an environment with a PyTorch/PyTorch Geometric stack that matches your
CUDA version. For example:

```bash
conda create -n edma python=3.10
conda activate edma
```

Install PyTorch and PyTorch Geometric using their official instructions, then
install this package in editable mode from the repository root:

```bash
pip install -e .
```

The lightweight Python requirements are listed in `requirements.txt`. PyG may
also require compiled packages such as `pyg-lib`, `torch-scatter`,
`torch-sparse`, `torch-cluster`, and `torch-spline-conv`, depending on your
platform and PyTorch version.

## Data

Scripts use the QM9 dataset loader from PyTorch Geometric. By default, data is
stored at:

```text
../data/QM9
```

Pass `--data-path` to use a different location.

## Usage

Run a small SchNet smoke test:

```bash
python scripts/qm9_schnet_energy.py \
  --test-size 2 \
  --batch-size 2 \
  --epochs 1 \
  --node-feat-size 1 \
  --node-feat-ent 0.0005 \
  --output /tmp/edma_schnet_smoke.pkl
```

Run SchNet on QM9 property `G`/free-energy style targets by selecting the
corresponding PyG `target_attr`:

```bash
python scripts/qm9_schnet_energy.py \
  --gpu 0 \
  --target-attr 10 \
  --test-size 1024 \
  --batch-size 128 \
  --epochs 300 \
  --node-feat-size 500 \
  --node-feat-ent 10 \
  --output schnet_g_energy.pkl
```

Run DimeNet++:

```bash
python scripts/qm9_dimenet_energy.py \
  --gpu 0 \
  --dimenet plusplus \
  --target-attr 0 \
  --test-size 1024 \
  --batch-size 128 \
  --epochs 300 \
  --node-feat-size 1.0 \
  --node-feat-ent 0.0005 \
  --output dimenet_u_energy.pkl
```

Use DimeNet instead of DimeNet++ with:

```bash
python scripts/qm9_dimenet_energy.py --dimenet base
```

## Output Format

The evaluation scripts write a pickle file with:

- `best_mae`
- `best_parameters`
- `all_results`
- `all_closeness`
- `all_node_ranks`
- `all_node_masks`

This keeps compatibility with the original experiment scripts.

## Importing EDMA

```python
from edma.explainers.schnet import SchNetEnergyInstanceExplainer
from edma.explainers.dimenet import DimeNetEnergyInstanceExplainer
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{edma2026,
  title = {Confidence-Aware Explanations for 3D Molecular Graphs via Energy-Based Masking},
  author = {Xufeng Liu, Wenhan Gao, Yi Liu},
  journal = {Transactions on Machine Learning Research},
  year = {2026}
}
```

Update the BibTeX entry after the final TMLR metadata is available.

## License

This project is released under the MIT License. See `LICENSE`.
