#!/usr/bin/env python3
"""Build mergeweight_final.pth from a final_mergeweight CSV file."""
import argparse
import csv
import os

import numpy as np
import torch


def load_mergeweight_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = [f"weight_col{i}" for i in range(8)]
        for row in reader:
            rows.append([float(row[c]) for c in cols])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape != (294, 8):
        raise ValueError(f"expected (294, 8), got {arr.shape} from {csv_path}")
    return torch.from_numpy(arr)


def save_ckpt(tensor, out_path, source_csv):
    payload = {
        "mergeweight": tensor.clone(),
        "mergeweight_clamped": torch.clamp(tensor, min=0.0, max=1.0).clone(),
        "shape": tuple(tensor.shape),
        "meta": {
            "source_csv": os.path.abspath(source_csv),
            "stage": "from_csv",
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {out_path} shape={tuple(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    w = load_mergeweight_csv(args.csv)
    save_ckpt(w, args.out, args.csv)


if __name__ == "__main__":
    main()
