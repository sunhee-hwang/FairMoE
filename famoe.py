# =========================
# FAMoE: Fairness-Aware Mixture-of-Experts
# via Subgroup Reweighting and Gate Entropy Regularization
# =========================

import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# SEED FIX
# =========================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# DATASET
# =========================
class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, part_path, transform=None,
                 target_attr="Attractive", sensitive_attr="Male"):
        self.img_dir = img_dir
        self.transform = transform
        self.target_attr = target_attr
        self.sensitive_attr = sensitive_attr

        with open(attr_path, 'r') as f:
            lines = f.readlines()

        columns = lines[1].strip().split()
        data_lines = lines[2:]

        attr_dict = {}
        for line in data_lines:
            parts = line.strip().split()
            img = parts[0]
            values = list(map(int, parts[1:]))
            attr_dict[img] = dict(zip(columns, values))

        attr_df = pd.DataFrame.from_dict(attr_dict, orient='index')

        part_df = pd.read_csv(part_path, sep=r'\s+', header=None)
        part_df.columns = ["image", "partition"]
        part_df.set_index("image", inplace=True)

        df = attr_df.join(part_df, how="inner")

        self.df = df
        self.img_names = df.index.tolist()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        row = self.df.loc[img_name]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = 1 if row[self.target_attr] == 1 else 0
        s = 1 if row[self.sensitive_attr] == 1 else 0

        return img, torch.tensor(y).float(), torch.tensor(s).long()


def compute_subgroup_weights(y, s, eps=1e-6):
    """
    Subgroup reweighting based on the joint distribution of (target y, sensitive s).
    Weight is inversely proportional to subgroup size.

    y: [B, 1] float tensor with values {0, 1}
    s: [B]    long  tensor with values {0, 1}
    returns:
        weights: [B, 1] float tensor
    """
    y_flat = y.view(-1).long()
    s_flat = s.view(-1).long()

    weights = torch.zeros_like(y_flat, dtype=torch.float)

    subgroup_masks = [
        (y_flat == 0) & (s_flat == 0),
        (y_flat == 0) & (s_flat == 1),
        (y_flat == 1) & (s_flat == 0),
        (y_flat == 1) & (s_flat == 1),
    ]

    counts = [mask.sum().float() for mask in subgroup_masks]
    total = y_flat.numel()
    num_groups = 4

    for mask, count in zip(subgroup_masks, counts):
        if count > 0:
            weights[mask] = total / (num_groups * count + eps)

    return weights.unsqueeze(1)


# =========================
# MODEL: Fairness-Aware MoE
# =========================
class FAMoE(nn.Module):
    def __init__(self, num_experts=4, expert_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        dim = backbone.fc.in_features

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, expert_dim)
            ) for _ in range(num_experts)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(expert_dim, 1) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)

        gate = F.softmax(self.gate(feat), dim=1)

        expert_feats = [e(feat) for e in self.experts]
        logits = torch.stack(
            [c(f) for c, f in zip(self.classifiers, expert_feats)], dim=1)

        out = torch.sum(gate.unsqueeze(2) * logits, dim=1)
        return out, gate


# =========================
# METRICS
# =========================
def compute_metrics(pred, y, s):
    pred = pred.view(-1)
    y = y.view(-1).long()
    s = s.view(-1).long()

    prob = torch.sigmoid(pred)
    pred_bin = (prob > 0.5).long()

    acc = (pred_bin == y).float().mean().item()

    def class_acc(val):
        mask = (y == val)
        if mask.sum() == 0:
            return 0.0
        return (pred_bin[mask] == y[mask]).float().mean().item()

    bal_acc = (class_acc(0) + class_acc(1)) / 2

    def group_acc(y_val, s_val):
        mask = (y == y_val) & (s == s_val)
        if mask.sum() == 0:
            return 0.0
        return (pred_bin[mask] == y[mask]).float().mean().item()

    eo = (
        abs(group_acc(0, 0) - group_acc(0, 1)) +
        abs(group_acc(1, 0) - group_acc(1, 1))
    ) / 2

    return acc, bal_acc, eo


# =========================
# TRAIN
# =========================
def run_experiment(config, dataset_base, device, transforms_dict,
                   target_attr, sensitive_attr, seed=0):

    name = config["name"]
    print(f"\n===== {name} | Target: {target_attr} | Sensitive: {sensitive_attr} =====")

    idx_map = {n: i for i, n in enumerate(dataset_base.img_names)}

    train_idx = [idx_map[i] for i in dataset_base.df[dataset_base.df['partition'] == 0].index]
    val_idx   = [idx_map[i] for i in dataset_base.df[dataset_base.df['partition'] == 1].index]
    test_idx  = [idx_map[i] for i in dataset_base.df[dataset_base.df['partition'] == 2].index]

    # separate dataset per split (different transforms)
    train_dataset = CelebADataset(img_dir, attr_path, part_path, transform=transforms_dict["train"],
                                  target_attr=target_attr, sensitive_attr=sensitive_attr)
    val_dataset   = CelebADataset(img_dir, attr_path, part_path, transform=transforms_dict["val"],
                                  target_attr=target_attr, sensitive_attr=sensitive_attr)
    test_dataset  = CelebADataset(img_dir, attr_path, part_path, transform=transforms_dict["val"],
                                  target_attr=target_attr, sensitive_attr=sensitive_attr)

    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(val_dataset, val_idx)
    test_ds  = Subset(test_dataset, test_idx)

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=256)
    test_dl  = DataLoader(test_ds, batch_size=256)

    model = FAMoE(num_experts=4, expert_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_score = 1e9
    alpha = config.get("alpha", 0.5)
    lambda_gate = config.get("lambda_gate", 0.01)
    num_epochs = 30

    save_dir = f"experiments/{name}_{target_attr}_{sensitive_attr}/seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best.pt")

    log_path = os.path.join(save_dir, f"log_seed{seed}.txt")
    log_file = open(log_path, "w")

    for epoch in range(num_epochs):

        model.train()
        for x, y, s in train_dl:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            s = s.to(device)

            pred, gate = model(x)

            # weighted BCE (subgroup reweighting)
            bce = F.binary_cross_entropy_with_logits(pred, y, reduction='none')
            weights = compute_subgroup_weights(y, s).to(device)
            loss = (weights * bce).mean()

            # gate entropy regularization
            gate_entropy = -(gate * torch.log(gate + 1e-8)).sum(dim=1).mean()
            loss -= lambda_gate * gate_entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ===== VALID =====
        model.eval()
        preds, ys, ss = [], [], []
        with torch.no_grad():
            for x, y, s in val_dl:
                x = x.to(device)
                pred, _ = model(x)
                preds.append(pred.cpu())
                ys.append(y)
                ss.append(s)

        preds = torch.cat(preds)
        ys = torch.cat(ys)
        ss = torch.cat(ss)

        acc, bal_acc, eo = compute_metrics(preds, ys, ss)
        score = eo + alpha * (1 - acc)  # validation FATS

        log_line = (f"{name} | Epoch {epoch+1} | Acc {acc:.4f} | "
                    f"BalAcc {bal_acc:.4f} | EO {eo:.4f} | Score {score:.4f}")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), best_model_path)

    # ===== TEST =====
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    preds, ys, ss = [], [], []
    with torch.no_grad():
        for x, y, s in test_dl:
            x = x.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu())
            ys.append(y)
            ss.append(s)

    preds = torch.cat(preds)
    ys = torch.cat(ys)
    ss = torch.cat(ss)

    acc, bal_acc, eo = compute_metrics(preds, ys, ss)
    fats = eo * 100 + alpha * (1 - acc) * 100

    test_line = (f"TEST | {name} | Acc {acc:.4f} | BalAcc {bal_acc:.4f} | "
                 f"EO {eo:.4f} | FATS {fats:.2f}")
    print(test_line)
    log_file.write(test_line + "\n")
    log_file.close()

    return acc, eo


# =========================
# MAIN
# =========================
def main():

    global img_dir, attr_path, part_path

    img_dir = "./celeba/img_align_celeba"
    attr_path = "./celeba/list_attr_celeba.txt"
    part_path = "./celeba/list_eval_partition.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    image_size = 128

    transforms_dict = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    }

    dataset = CelebADataset(img_dir, attr_path, part_path)

    # FAMoE (Ours): MoE + subgroup reweighting + gate entropy regularization
    config = {
        "name": "FAMoE",
        "lambda_gate": 0.01,
        "alpha": 0.5,
    }

    results = []
    target_attrs = ['Attractive']
    sensitive_attrs = ["Male", "Young"]

    print(f"Target: {target_attrs}")
    print(f"Sensitive: {sensitive_attrs}")

    seed = 0
    set_seed(seed)

    for target_attr in target_attrs:
        for sensitive_attr in sensitive_attrs:
            acc, eo = run_experiment(config, dataset, device, transforms_dict,
                                     target_attr, sensitive_attr, seed)
            results.append((target_attr, sensitive_attr, acc, eo))

    print("\n===== RESULT =====")
    print("Target\t\tSensitive\tAcc\tEO")
    for t, s, acc, eo in results:
        print(f"{t:12s}\t{s:8s}\t{acc:.4f}\t{eo:.4f}")


if __name__ == "__main__":
    main()
