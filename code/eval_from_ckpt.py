#!/usr/bin/env python3
"""Load mergeweight_final.pth and run Epoch2 eval only."""
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_config_paths(cfg, ada_root=None):
    root = os.path.abspath(ada_root or ADA_ROOT)
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str) and _PATH_PLACEHOLDER in value:
            resolved[key] = value.replace(_PATH_PLACEHOLDER, root.rstrip("/"))
        else:
            resolved[key] = value
    return resolved


def build_model(m, args, exam_datasets):
    pretrained_model_dict = torch.load(args.pretrained_checkpoint_path, map_location="cpu")["model"]
    task_vectors = [
        m.TaskVector(
            args.pretrained_checkpoint_path,
            args.checkpoints_path + f"/RETFound_mae_meh-{ds}/checkpoint-nohead.pth",
        )
        for ds in exam_datasets
    ]
    pretrained_model = m.RETFound_mae(num_classes=0, global_pool=True, drop_path_rate=0.2)
    pretrained_model.load_state_dict(pretrained_model_dict, strict=False)
    wrapper = m.ModelWrapper(pretrained_model, exam_datasets).to("cpu")
    _, names = m.make_functional(wrapper)
    paramslist = [tuple((k, v.detach().cpu()) for k, v in pretrained_model_dict.items())]
    paramslist += [tuple((k, v.detach().cpu()) for k, v in tv.vector.items()) for tv in task_vectors]
    del task_vectors
    model = m.Adamerge_model(paramslist, exam_datasets, args.head_path, names=names, model=wrapper)
    model.to(args.device)
    model.isfirst = False
    for name, param in model.named_parameters():
        if name != "mergeweight":
            param.requires_grad = False
    return model


def load_args_from_config(cfg_path):
    with open(cfg_path) as f:
        cfg = resolve_config_paths(json.load(f))
    sys.argv = ["eval_from_ckpt.py"]
    m = load_train_module()
    args = m.getargs()
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return m, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(_SUB_DIR, "config/run_args.json"),
        help="evaluation config JSON",
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_SUB_DIR, "model/mergeweight_final.pth"),
        help="mergeweight checkpoint path",
    )
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--tag", default="submission", help="metrics output suffix")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--metrics_out", default="", help="metrics CSV output path; defaults to results/metrics_{tag}.csv")
    cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    m, args = load_args_from_config(cli.config)
    set_seed(cli.seed)
    args.device = "cuda:0"
    args.num_workers = cli.num_workers
    args.batch_size = cli.batch_size

    out_dir = os.path.join(_SUB_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{cli.tag}" if cli.tag else ""
    log_file = cli.metrics_out or os.path.join(out_dir, f"metrics{tag}.csv")
    eval_log = os.path.join(_SUB_DIR, "logs", f"eval{tag}.log")
    os.makedirs(os.path.dirname(eval_log), exist_ok=True)

    ckpt_path = cli.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    exam_datasets = args.exam_datasets
    print(f"ckpt={ckpt_path}")
    print(f"gpu={cli.gpu} bs={args.batch_size} seed={cli.seed} metrics={log_file}")

    model = build_model(m, args, exam_datasets)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.mergeweight.data.copy_(ck["mergeweight"].to(args.device))
    print(f"loaded mergeweight shape={tuple(model.mergeweight.shape)}")

    epoch2_aucs = []
    with open(eval_log, "w") as flog:
        flog.write(f"ckpt={ckpt_path}\n")
        flog.write(f"seed={cli.seed} batch_size={args.batch_size}\n")
        g = torch.Generator()
        g.manual_seed(cli.seed)
        for dataset_name in exam_datasets:
            dataset_test = m.build_dataset(is_train="test", args=args, dataset_name=dataset_name)
            loader = torch.utils.data.DataLoader(
                dataset_test,
                sampler=torch.utils.data.SequentialSampler(dataset_test),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                worker_init_fn=seed_worker if args.num_workers > 0 else None,
                generator=g,
            )
            model.isfirst = False
            metrics = m.evaluate(loader, model, args, dataset_name=dataset_name)
            m.write_metrics_to_csv(log_file, cli.epoch, dataset_name, 0, metrics)
            epoch2_aucs.append(metrics["roc_auc"])
            line = (
                f"{dataset_name} acc={metrics['accuracy']:.4f} "
                f"auc={metrics['roc_auc']:.6f}\n"
            )
            print(line.strip())
            flog.write(line)
            torch.cuda.empty_cache()

        mean_auc = sum(epoch2_aucs) / len(epoch2_aucs)
        summary = f"epoch{cli.epoch}_mean_roc_auc={mean_auc:.6f}\n"
        print(summary.strip())
        flog.write(summary)

    print(f"done -> {log_file}")


if __name__ == "__main__":
    main()
