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

## 4. Dataset splits and sample counts

The FF++ (c23) train/validation/test assignment follows the annotated data
release of [Lin et al.](https://github.com/Purdue-M2/Fairness-Generalization):
60/20/20 at the video level with no identity overlap between splits.
Celeb-DF, DFD and DFDC are used in full as unseen test sets.
After excluding "Others", the splits used in the paper contain:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| FF++ | 45,377 | 15,126 | 15,126 |
| Celeb-DF | — | — | 87,213 |
| DFD | — | — | 20,032 |
| DFDC | — | — | 63,225 |

Per-subgroup counts (for FF++, totals across train/val/test; "—" means the
subgroup does not exist in that dataset's annotations):

| Dataset | M-A | M-W | M-B | F-A | F-W | F-B | Total |
|---------|-----|-----|-----|-----|-----|-----|-------|
| FF++ | 2,475 | 31,281 | 1,468 | 8,013 | 31,281 | 1,111 | 75,629 |
| Celeb-DF | — | 81,194 | 600 | — | 5,389 | 30 | 87,213 |
| DFD | — | 7,784 | 6,482 | — | 4,127 | 1,639 | 20,032 |
| DFDC | 2,144 | 21,755 | 9,603 | 1,915 | 18,502 | 9,306 | 63,225 |

For comparisons at the fixed operating point, remember that all FPR-based
metrics use a single threshold calibrated on the FF++ validation split at
TPR = 0.90 and reused unchanged on every test set (Sec. 4.1 of the paper).
