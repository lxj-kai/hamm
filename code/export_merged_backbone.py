#!/usr/bin/env python3
"""Export the merged RETFound backbone from a HAMM mergeweight checkpoint."""
import argparse
import json
import os
import sys

import torch

_SUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(_SUB_DIR, "code")
ADA_ROOT = os.environ.get("ADA_ROOT", os.path.abspath(os.path.join(_SUB_DIR, "..")))
_PATH_PLACEHOLDER = "/path/to/HAMM_resources"

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from eval_from_ckpt import build_model, load_train_module


def resolve_config_paths(cfg, ada_root=None):
    root = os.path.abspath(ada_root or ADA_ROOT)
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str) and _PATH_PLACEHOLDER in value:
            resolved[key] = value.replace(_PATH_PLACEHOLDER, root.rstrip("/"))
        else:
            resolved[key] = value
    return resolved


def load_args_from_config(cfg_path):
    with open(cfg_path) as f:
        cfg = resolve_config_paths(json.load(f))
    sys.argv = ["export_merged_backbone.py"]
    module = load_train_module()
    args = module.getargs()
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return module, args


def compute_merged_backbone_state_dict(model):
    """Apply the same merge rule as Adamerge_model.forward and return a backbone state_dict."""
    lambdas = model.lambdas()
    merged = {}
    n_sources = len(model.paramslist)
    for j, params in enumerate(zip(*model.paramslist)):
        is_pretrained = True
        total = 0
        name = None
        for (param_name, param_value), lambda_i in zip(params, lambdas[j].cpu()):
            name = param_name
            if any(keyword in param_name for keyword in model.update_layers) and param_value.dim() in [1, 2]:
                total = total + param_value * lambda_i
            else:
                total = total + param_value * ((n_sources - 1) if is_pretrained else 1) / (n_sources - 1)
                is_pretrained = False
        merged[name] = total.detach().cpu().clone()
    return merged


def main():
    parser = argparse.ArgumentParser(description="export a merged RETFound backbone")
    parser.add_argument(
        "--config",
        default=os.path.join(_SUB_DIR, "config/run_args.json"),
        help="path configuration JSON",
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_SUB_DIR, "model/mergeweight_final.pth"),
        help="mergeweight checkpoint",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_SUB_DIR, "model/merged_backbone_final.pth"),
        help="output path for the merged backbone checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="device used to construct the merged backbone; cpu is recommended for export",
    )
    cli = parser.parse_args()

    if cli.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not os.path.isfile(cli.ckpt):
        raise FileNotFoundError(f"mergeweight checkpoint not found: {cli.ckpt}")

    module, args = load_args_from_config(cli.config)
    args.device = "cuda:0" if cli.device == "cuda" and torch.cuda.is_available() else "cpu"
    exam_datasets = args.exam_datasets

    print(f"mergeweight={cli.ckpt}")
    print(f"device={args.device}")
    print(f"out={cli.out}")

    model = build_model(module, args, exam_datasets)
    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    model.mergeweight.data.copy_(checkpoint["mergeweight"].to(args.device))
    print(f"loaded mergeweight shape={tuple(model.mergeweight.shape)}")

    merged = compute_merged_backbone_state_dict(model)
    num_params = sum(value.numel() for value in merged.values())
    num_bytes = sum(value.numel() * value.element_size() for value in merged.values())
    payload = {
        "model": merged,
        "meta": {
            "format": "RETFound_mae_backbone_nohead",
            "source_mergeweight": os.path.abspath(cli.ckpt),
            "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
            "checkpoints_path": args.checkpoints_path,
            "exam_datasets": exam_datasets,
            "merge_rule": "qkv: pretrained + sum(lambda_i * task_vector_i); other: mean(pretrained + task_vectors)",
            "num_tensors": len(merged),
            "num_params": num_params,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    torch.save(payload, cli.out)
    file_size = os.path.getsize(cli.out)
    print(f"saved {cli.out}")
    print(f"  tensors={len(merged)} params={num_params:,} raw={num_bytes/1e9:.2f}GB file={file_size/1e9:.2f}GB")
    print("Dataset-specific classifier heads are not stored in this file.")


if __name__ == "__main__":
    main()
