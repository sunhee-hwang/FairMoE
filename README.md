# FAMoE: Fairness-Aware Mixture-of-Experts

Official implementation of **"Fairness-Aware Mixture-of-Experts via Subgroup Reweighting and Gate Regularization."**

FAMoE mitigates demographic bias in visual classification by jointly addressing
**subgroup data imbalance** and **skewed expert routing** in a Mixture-of-Experts (MoE) model.
It combines two lightweight components on top of a standard MoE:

- **Subgroup reweighting** — sample weights inversely proportional to the size of each
  joint subgroup `(target y, sensitive s)`, correcting data imbalance.
- **Gate entropy regularization** — an entropy term on the routing distribution that
  encourages balanced expert utilization and prevents routing from correlating with
  sensitive attributes.

The model is trained end-to-end and selects the best checkpoint by **validation FATS**
(Fairness–Accuracy Trade-off Score).

## Method Overview

| Component | Description | Paper |
|---|---|---|
| MoE head | 4 experts + a softmax gating network over a shared ResNet-18 backbone | §3.1–3.2 |
| Subgroup reweighting | weight `w_{y,s} ∝ 1 / n_{y,s}` in a weighted BCE loss | §3.3 |
| Gate entropy regularization | `L = L_cls − λ_ent · H(g)` | §3.4 |

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.9+ and PyTorch 2.x. A CUDA-capable GPU is recommended.

## Dataset

This code uses the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
Download and arrange it as follows:

```
celeba/
├── img_align_celeba/        # all aligned face images (*.jpg)
├── list_attr_celeba.txt     # attribute annotations
└── list_eval_partition.txt  # official train/val/test split
```

Place the `celeba/` folder in the project root (or edit the paths in `main()` of `famoe.py`).

- **Target attributes:** `Attractive`, `Big_Nose`
- **Sensitive attributes:** `Male`, `Young`

By default the script runs the `Attractive` target against both sensitive attributes.
To reproduce the full Table 1 setting, edit `main()`:

```python
target_attrs = ['Attractive', 'Big_Nose']
sensitive_attrs = ["Male", "Young"]
```

## Usage

```bash
python famoe.py
```

This trains FAMoE for 30 epochs (Adam, lr `1e-4`, weight decay `1e-4`, batch size 256),
saves the best checkpoint per run under `experiments/`, and prints test
Accuracy / EO / FATS. The random seed is fixed to `0` for reproducibility.

## Evaluation Metrics

- **Accuracy (ACC)** — overall binary classification accuracy.
- **Equalized Odds (EO)** — mean absolute difference in per-class accuracy across
  subgroups defined by `(y, s)`. Lower is better.
- **FATS** — `EO + α · (1 − ACC)` with `α = 0.5`. Lower indicates a more favorable
  fairness–accuracy trade-off; used for model selection.

## Output

```
experiments/
└── FAMoE_<target>_<sensitive>/
    └── seed0/
        ├── best.pt
        └── log_seed0.txt
```

## Citation

```bibtex
@inproceedings{hwang2026famoe,
  title     = {Fairness-Aware Mixture-of-Experts via Subgroup Reweighting and Gate Regularization},
  author    = {Hwang, Sunhee},
  booktitle = {Proceedings of the IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS)},
  address   = {Lecce, Italy},
  year      = {2026}
}
```
