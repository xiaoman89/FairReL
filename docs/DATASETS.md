# Dataset Preparation

FairReL is trained on FF++ (c23) and evaluated on FF++, Celeb-DF, DFD and
DFDC under a cross-dataset fairness protocol. This document describes how to
obtain the data and build the CSV files consumed by `train.py` / `test.py`.

## 1. Datasets and demographic annotations

Our experiments use four datasets:
[FF++ (c23)](https://github.com/ondyari/FaceForensics) as the source domain,
and [Celeb-DF (v2)](https://github.com/yuezunli/celeb-deepfakeforensics),
[DFD](https://github.com/ondyari/FaceForensics) and
[DFDC](https://ai.meta.com/datasets/dfdc/) as unseen targets.
Each dataset requires agreeing to its terms of use before downloading;
DFD is distributed through the FaceForensics++ download script
(dataset name `DeepFakeDetection`).

The demographic annotations follow
[Lin et al. (CVPR 2024)](https://github.com/Purdue-M2/Fairness-Generalization),
whose repository provides the annotated versions of all four datasets.
Fairness is evaluated over the intersectional gender-race partition with six
subgroups: Male-White (M-W), Male-Asian (M-A), Male-Black (M-B),
Female-White (F-W), Female-Asian (F-A), Female-Black (F-B).
**Samples annotated as "Others" are excluded.** Subgroups missing from a
dataset's annotations (e.g. Asian subgroups in Celeb-DF and DFD) are simply
absent from that dataset's CSV; the metrics handle this automatically.

## 2. Preprocessing

We use frame-level evaluation on face crops:

1. Extract frames from each video.
2. Detect, align and crop the face region, and save each crop as an image
   (any standard pipeline works; we follow the common practice of prior
   fairness work — e.g. the
   [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) preprocessing
   tools).
3. Resize is handled inside the dataloader (bicubic to 224 x 224, CLIP
   normalisation), so crops can be stored at their native resolution.

## 3. Build the CSVs

One CSV per split with columns:

```csv
path,label,subgroup
ffpp/real/000/frame_000.png,0,M-W
ffpp/fake/Deepfakes/000_003/frame_000.png,1,F-A
```

- `path` — image path, absolute or relative to `--data-root`
- `label` — 0 = real, 1 = fake
- `subgroup` — one of `M-W, M-A, M-B, F-W, F-A, F-B`
  (alternatively provide `gender` in {M, F} and `race` in
  {White, Asian, Black} as two columns; they are combined automatically)

Required files:

- `ffpp_train.csv`, `ffpp_val.csv`, `ffpp_test.csv` — FF++ is split
  60/20/20 into train/validation/test **at the video level with no identity
  overlap between splits**, following the split protocol of
  [Lin et al.](https://github.com/Purdue-M2/Fairness-Generalization);
  if you use their released annotated data, keep their split assignment.
  Never split frames randomly: frames of the same video (or the same
  identity) appearing in both train and test inflate every metric.
- `celebdf_test.csv`, `dfd_test.csv`, `dfdc_test.csv` — full test sets

## 4. Expected sample counts

After excluding "Others", the splits used in the paper contain:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| FF++ | 45,377 | 15,126 | 15,126 |
| Celeb-DF | — | — | 87,213 |
| DFD | — | — | 20,032 |
| DFDC | — | — | 63,225 |

Per-subgroup counts are listed in Table S2 of the supplementary material.
