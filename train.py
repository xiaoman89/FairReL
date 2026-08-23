"""Train FairReL on FF++ (c23) with demographic subgroup labels.

Example:
    python train.py --config configs/fairrel_clip_vitl14.yaml \
        --train-csv data/ffpp_train.csv --val-csv data/ffpp_val.csv \
        --data-root /path/to/images --output-dir outputs/fairrel

Objective (Eq. 9):  L = L_CE + lambda_orth * L_orth + lambda_G * L_GCWD
                        + lambda_S * L_SLMA
Only the SVD residual factors and the classifier are updated.

Model selection follows the protocol in the supplementary material: the
checkpoint maximising (val AUC - val F_FPR) on the FF++ validation split at
the TPR = 0.90 operating point.  Target-domain data is never used.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairrel.data import DeepfakeDataset, GroupBalancedBatchSampler, SUBGROUPS
from fairrel.losses import GCWDLoss, SLMALoss
from fairrel.metrics import fairness_report, format_report, threshold_at_tpr
from fairrel.models import FairReLDetector
from fairrel.utils import AverageMeter, load_config, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train FairReL")
    p.add_argument("--config", default="configs/fairrel_clip_vitl14.yaml")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--output-dir", default="outputs/fairrel")
    p.add_argument("--seed", type=int, default=None, help="overrides config seed")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, tpr_anchor):
    model.eval()
    scores, labels, groups = [], [], []
    for img, y, g in tqdm(loader, desc="val", leave=False):
        p = model.predict(img.to(device, non_blocking=True))
        scores.extend(p.float().cpu().tolist())
        labels.extend(y.tolist())
        groups.extend(g.tolist())
    thr = threshold_at_tpr(scores, labels, tpr_anchor)
    return fairness_report(scores, labels, groups, thr)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tcfg, mcfg, lcfg = cfg["train"], cfg["model"], cfg["loss"]

    train_set = DeepfakeDataset(args.train_csv, args.data_root,
                                tcfg["input_size"], train=True)
    val_set = DeepfakeDataset(args.val_csv, args.data_root,
                              tcfg["input_size"], train=False)
    sampler = GroupBalancedBatchSampler(train_set.groups, tcfg["batch_size"], seed=seed)
    train_loader = DataLoader(train_set, batch_sampler=sampler,
                              num_workers=tcfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"], pin_memory=True)

    model = FairReLDetector(mcfg["backbone"], mcfg["pretrained"], mcfg["svd_rank"],
                            mcfg.get("pooling", "cls")).to(device)
    gcwd = GCWDLoss(num_groups=len(SUBGROUPS), levels=lcfg["gcwd"]["levels"],
                    temperature=lcfg["gcwd"]["temperature"],
                    normalize_energy=lcfg["gcwd"]["normalize_energy"]).to(device)
    slma = SLMALoss(num_groups=len(SUBGROUPS), feat_dim=model.backbone.feat_dim,
                    ema_decay=lcfg["slma"]["ema_decay"],
                    min_count=lcfg["slma"]["min_count"]).to(device)

    params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_trainable:,} "
          f"(SVD residual factors + classifier only)")
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    use_amp = tcfg["amp"] and device.type == "cuda"
    best_score, history = float("-inf"), []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        meters = {k: AverageMeter() for k in ("loss", "ce", "orth", "gcwd", "slma")}
        for img, y, g in tqdm(train_loader, desc=f"epoch {epoch}"):
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                feat_map, _, logit = model(img)              # standard path
                z_delta = model.forward_residual(img)        # residual-only path
                loss_ce = F.binary_cross_entropy_with_logits(logit.float(), y.float())
                loss_gcwd = gcwd(feat_map.float(), g)
                loss_slma = slma(z_delta.float(), y, g)
            loss_orth = model.orthogonality_loss()           # fp32 factors
            loss = (loss_ce + lcfg["lambda_orth"] * loss_orth
                    + lcfg["lambda_gcwd"] * loss_gcwd
                    + lcfg["lambda_slma"] * loss_slma)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = img.size(0)
            for k, v in (("loss", loss), ("ce", loss_ce), ("orth", loss_orth),
                         ("gcwd", loss_gcwd), ("slma", loss_slma)):
                meters[k].update(v.item(), n)

        slma_fallback = slma.fallback_fraction()
        slma.reset_stats()
        report = evaluate(model, val_loader, device, cfg["eval"]["tpr_anchor"])
        score = report["auc"] - report["f_fpr"]
        print(f"epoch {epoch}: " + " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
              + f" slma_ema_fallback={100.0 * slma_fallback:.2f}%")
        print(format_report("FF++ val", report)
              + f" | realised TPR {report['tpr_overall']:.1f}"
              + f" | selection score {score:.2f}")
        history.append({"epoch": epoch, "val": report, "score": score,
                        "slma_ema_fallback": slma_fallback,
                        **{k: m.avg for k, m in meters.items()}})
        save_json(history, os.path.join(args.output_dir, "history.json"))

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch,
                 "val_report": report, "seed": seed}
        torch.save(state, os.path.join(args.output_dir, "last.pt"))
        if score > best_score:
            best_score = score
            torch.save(state, os.path.join(args.output_dir, "best.pt"))
            print(f"  new best checkpoint (score {score:.2f})")

    print(f"done. best selection score: {best_score:.2f}")


if __name__ == "__main__":
    main()
