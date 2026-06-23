#!/usr/bin/env python3
"""Evaluate a merged RETFound backbone checkpoint with dataset-specific classifier heads."""
import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

_SUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(_SUB_DIR, "code")
ADA_ROOT = os.environ.get("ADA_ROOT", os.path.abspath(os.path.join(_SUB_DIR, "..")))
_PATH_PLACEHOLDER = "/path/to/HAMM_resources"
TRAIN_SCRIPT = os.path.join(CODE_DIR, "train_merge.py")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_train_module():
    spec = importlib.util.spec_from_file_location("merge_train", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    sys.argv = ["eval_from_merged_backbone.py"]
    module = load_train_module()
    args = module.getargs()
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return module, args


class MergedBackboneEvalModel(torch.nn.Module):
    def __init__(self, backbone, exam_datasets, head_path, classification_head_cls):
        super().__init__()
        self.backbone = backbone
        self.exam_datasets = exam_datasets
        for dataset_name in exam_datasets:
            head_file = os.path.join(head_path, f"head_{dataset_name}.pt")
            if not os.path.isfile(head_file):
                raise FileNotFoundError(f"classification head not found: {head_file}")
            head_ckpt = torch.load(head_file, map_location="cpu", weights_only=False)
            weight = head_ckpt["model.head.weight"]
            bias = head_ckpt["model.head.bias"]
            head = classification_head_cls(weight, bias)
            setattr(self, f"classifier_{dataset_name}", head)

    def forward(self, images, dataset_name):
        features = self.backbone(images)
        head = getattr(self, f"classifier_{dataset_name}")
        return head(features)


def build_model(module, args, exam_datasets, backbone_path):
    payload = torch.load(backbone_path, map_location="cpu", weights_only=False)
    merged_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    backbone = module.RETFound_mae(num_classes=0, global_pool=True, drop_path_rate=0.2)
    backbone.load_state_dict(merged_dict, strict=False)
    wrapper = module.ModelWrapper(backbone)
    model = MergedBackboneEvalModel(
        wrapper, exam_datasets, args.head_path, module.ClassificationHead
    )
    model.to(args.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="evaluate a merged backbone checkpoint")
    parser.add_argument(
        "--config",
        default=os.path.join(_SUB_DIR, "config/run_args.json"),
        help="evaluation config JSON",
    )
    parser.add_argument(
        "--backbone",
        default=os.path.join(_SUB_DIR, "model/merged_backbone_final.pth"),
        help="merged backbone checkpoint",
    )
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--tag", default="merged_backbone", help="metrics output suffix")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--metrics_out", default="", help="metrics CSV output path")
    cli = parser.parse_args()

    if not os.path.isfile(cli.backbone):
        raise FileNotFoundError(f"backbone checkpoint not found: {cli.backbone}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    module, args = load_args_from_config(cli.config)
    set_seed(cli.seed)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.num_workers = cli.num_workers
    args.batch_size = cli.batch_size

    out_dir = os.path.join(_SUB_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{cli.tag}" if cli.tag else ""
    log_file = cli.metrics_out or os.path.join(out_dir, f"metrics{tag}.csv")
    eval_log = os.path.join(_SUB_DIR, "logs", f"eval{tag}.log")
    os.makedirs(os.path.dirname(eval_log), exist_ok=True)

    exam_datasets = args.exam_datasets
    print(f"backbone={cli.backbone}")
    print(f"gpu={cli.gpu} bs={args.batch_size} seed={cli.seed} metrics={log_file}")

    model = build_model(module, args, exam_datasets, cli.backbone)

    epoch_aucs = []
    with open(eval_log, "w") as flog:
        flog.write(f"backbone={cli.backbone}\n")
        flog.write(f"seed={cli.seed} batch_size={args.batch_size}\n")
        generator = torch.Generator()
        generator.manual_seed(cli.seed)
        for dataset_name in exam_datasets:
            dataset_test = module.build_dataset(is_train="test", args=args, dataset_name=dataset_name)
            loader = torch.utils.data.DataLoader(
                dataset_test,
                sampler=torch.utils.data.SequentialSampler(dataset_test),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                worker_init_fn=seed_worker if args.num_workers > 0 else None,
                generator=generator,
            )
            metrics = module.evaluate(loader, model, args, dataset_name=dataset_name)
            module.write_metrics_to_csv(log_file, cli.epoch, dataset_name, 0, metrics)
            epoch_aucs.append(metrics["roc_auc"])
            line = (
                f"{dataset_name} acc={metrics['accuracy']:.4f} "
                f"auc={metrics['roc_auc']:.6f}\n"
            )
            print(line.strip())
            flog.write(line)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_auc = sum(epoch_aucs) / len(epoch_aucs)
        summary = f"epoch{cli.epoch}_mean_roc_auc={mean_auc:.6f}\n"
        print(summary.strip())
        flog.write(summary)

    print(f"done -> {log_file}")


if __name__ == "__main__":
    main()
