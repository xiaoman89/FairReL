"""CSV-driven deepfake dataset with demographic subgroup labels.

Expected CSV columns (see docs/DATASETS.md):
    path      image path, absolute or relative to ``data_root``
    label     0 = real, 1 = fake
    subgroup  one of M-W, M-A, M-B, F-W, F-A, F-B
              (alternatively two columns ``gender`` in {M, F} and ``race`` in
              {White, Asian, Black}, which are combined automatically)

Samples annotated as "Others" should be excluded when building the CSVs,
matching the six-subgroup protocol of the paper.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .subgroups import SUBGROUPS, SUBGROUP_TO_ID

# OpenAI CLIP normalisation.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_RACE_TO_CODE = {"white": "W", "asian": "A", "black": "B"}


def build_transform(input_size: int = 224, train: bool = False):
    ops = [transforms.Resize((input_size, input_size),
                             interpolation=transforms.InterpolationMode.BICUBIC)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    return transforms.Compose(ops)


def _subgroup_column(df: pd.DataFrame) -> pd.Series:
    if "subgroup" in df.columns:
        return df["subgroup"].astype(str).str.strip()
    if {"gender", "race"}.issubset(df.columns):
        gender = df["gender"].astype(str).str.strip().str.upper().str[0]
        race = df["race"].astype(str).str.strip().str.lower().map(_RACE_TO_CODE)
        if race.isna().any():
            bad = df.loc[race.isna(), "race"].unique()
            raise ValueError(f"unrecognised race values {bad}; expected White/Asian/Black")
        return gender + "-" + race
    raise ValueError("CSV must contain either a 'subgroup' column or 'gender' + 'race'")


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str = "", input_size: int = 224,
                 train: bool = False):
        df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain 'path' and 'label' columns")
        subgroup = _subgroup_column(df)
        unknown = set(subgroup.unique()) - set(SUBGROUPS)
        if unknown:
            raise ValueError(f"unknown subgroup labels {sorted(unknown)}; expected {SUBGROUPS}")

        self.paths = df["path"].tolist()
        self.labels = torch.tensor(df["label"].astype(int).values)
        self.groups = torch.tensor([SUBGROUP_TO_ID[s] for s in subgroup])
        self.data_root = data_root
        self.transform = build_transform(input_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_root, self.paths[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[idx], self.groups[idx]
