"""Evaluate a trained FairReL checkpoint under the paper's fixed-threshold protocol.

The decision threshold is fixed on the FF++ validation split at source-domain
TPR = 0.90 and reused unchanged on every test dataset (Sec. 4.1).

Example:
    python test.py --config configs/fairrel_clip_vitl14.yaml \
        --checkpoint outputs/fairrel/best.pt \
        --val-csv data/ffpp_val.csv --data-root /path/to/images \
        --test ffpp=data/ffpp_test.csv celebdf=data/celebdf_test.csv \
               dfd=data/dfd_test.csv dfdc=data/dfdc_test.csv
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairrel.data import DeepfakeDataset
from fairrel.metrics import fairness_report, format_report, threshold_at_tpr
from fairrel.models import FairReLDetector
from fairrel.utils import load_config, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FairReL")
    p.add_argument("--config", default="configs/fairrel_clip_vitl14.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-csv", required=True,
                   help="FF++ validation CSV used to fix the operating point")
    p.add_argument("--test", nargs="+", required=True, metavar="NAME=CSV")
    p.add_argument("--data-root", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output", default=None, help="optional JSON output path")
    return p.parse_args()


@torch.no_grad()
def collect_scores(model, csv_path, args, device, input_size):
    dataset = DeepfakeDataset(csv_path, args.data_root, input_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    scores, labels, groups = [], [], []
    for img, y, g in tqdm(loader, desc=os.path.basename(csv_path), leave=False):
        p = model.predict(img.to(device, non_blocking=True))
        scores.extend(p.float().cpu().tolist())
        labels.extend(y.tolist())
        groups.extend(g.tolist())
    return scores, labels, groups


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = cfg["train"]["input_size"]
    mcfg = cfg["model"]

    model = FairReLDetector(mcfg["backbone"], mcfg["pretrained"], mcfg["svd_rank"],
                            mcfg.get("pooling", "cls")).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    scores, labels, _ = collect_scores(model, args.val_csv, args, device, input_size)
    threshold = threshold_at_tpr(scores, labels, cfg["eval"]["tpr_anchor"])
    print(f"operating point: threshold {threshold:.4f} "
          f"(FF++ val TPR = {cfg['eval']['tpr_anchor']:.2f})")

    results = {"threshold": threshold}
    for spec in args.test:
        name, _, csv_path = spec.partition("=")
        if not csv_path:
            raise ValueError(f"--test entries must be NAME=CSV, got '{spec}'")
        s, y, g = collect_scores(model, csv_path, args, device, input_size)
        report = fairness_report(s, y, g, threshold)
        results[name] = report
        print(format_report(name, report))

    if args.output:
        save_json(results, args.output)
        print(f"saved results to {args.output}")


if __name__ == "__main__":
    main()
