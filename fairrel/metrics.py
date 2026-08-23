"""Detection and fairness metrics (paper Sec. 4.1 and supplementary Sec. B).

All FPR-based fairness metrics are computed at a single operating point tau
fixed on the FF++ validation split at source-domain TPR = 0.90 and reused on
every dataset, so FPR gaps cannot be shrunk by moving the threshold.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data.subgroups import SUBGROUPS


def threshold_at_tpr(scores, labels, target_tpr: float = 0.90) -> float:
    """Smallest threshold whose TPR is >= target_tpr on the given split."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = tpr >= target_tpr
    if not valid.any():
        return float(thresholds[-1])
    return float(thresholds[valid][0])


def _rate(pred, mask):
    n = int(mask.sum())
    return float(pred[mask].mean()) if n > 0 else float("nan")


def fairness_report(scores, labels, groups, threshold: float) -> dict:
    """Full metric set for one dataset at a fixed threshold.

    Returns AUC, AUC_avg, AUC gap, per-subgroup AUC/FPR, F_FPR (Equal FPR,
    sum of |FPR_g - FPR_overall|) and F_MEO (Max Equalized Odds, the larger
    of the worst-case FPR gap and worst-case TPR gap).  Percentages.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups).astype(int)
    # ">=" matches sklearn.roc_curve's threshold semantics, so the realised
    # TPR on the split used by threshold_at_tpr is guaranteed >= the target.
    pred = (scores >= threshold).astype(int)

    report = {"threshold": float(threshold),
              "auc": float(100.0 * roc_auc_score(labels, scores))}

    per_auc, per_fpr, per_tpr = {}, {}, {}
    for gid, name in enumerate(SUBGROUPS):
        mask = groups == gid
        if not mask.any():
            continue  # subgroup absent from this dataset's annotations
        if len(np.unique(labels[mask])) == 2:
            per_auc[name] = float(100.0 * roc_auc_score(labels[mask], scores[mask]))
        real, fake = mask & (labels == 0), mask & (labels == 1)
        if real.any():
            per_fpr[name] = 100.0 * _rate(pred, real)
        if fake.any():
            per_tpr[name] = 100.0 * _rate(pred, fake)

    aucs = list(per_auc.values())
    report["auc_avg"] = float(np.mean(aucs)) if aucs else float("nan")
    report["auc_gap"] = float(np.max(aucs) - np.min(aucs)) if len(aucs) > 1 else float("nan")
    report["per_subgroup_auc"] = per_auc
    report["per_subgroup_fpr"] = per_fpr

    overall_fpr = 100.0 * _rate(pred, labels == 0)
    report["fpr_overall"] = overall_fpr
    report["tpr_overall"] = 100.0 * _rate(pred, labels == 1)
    fprs = list(per_fpr.values())
    tprs = list(per_tpr.values())
    report["f_fpr"] = float(sum(abs(f - overall_fpr) for f in fprs)) if fprs else float("nan")
    gaps = []
    if len(fprs) > 1:
        gaps.append(max(fprs) - min(fprs))
    if len(tprs) > 1:
        gaps.append(max(tprs) - min(tprs))
    report["f_meo"] = float(max(gaps)) if gaps else float("nan")
    return report


def format_report(name: str, r: dict) -> str:
    return (f"{name:>10s} | AUC {r['auc']:6.2f}  AUC_avg {r['auc_avg']:6.2f}  "
            f"AUC_gap {r['auc_gap']:5.2f} | F_FPR {r['f_fpr']:6.2f}  F_MEO {r['f_meo']:6.2f}")
