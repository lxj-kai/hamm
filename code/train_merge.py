import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from tqdm import tqdm  
import torch

_SUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ADA_ROOT = os.environ.get("ADA_ROOT", os.path.abspath(os.path.join(_SUB_DIR, "..")))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from task_vectors import TaskVector
from model_vit import RETFound_mae
from retfound_dataset import build_dataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, jaccard_score, precision_score, recall_score, average_precision_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
import csv
from third_party import aug
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch.nn.functional as F
def watch_grad(grad_tensor,
              mergeweight_tensor,
              log_dir=os.path.join(ADA_ROOT, "temp_grad_logs"),
              weight_log_dir=None,
              batch_idx=0,
              dataset_name="OCTID"):
    """
    Save the gradient matrix and mergeweight matrix to CSV files.
    
    Args:
        grad_tensor: gradient tensor; must have shape (294, 8).
        mergeweight_tensor: mergeweight tensor; must have shape (294, 8).
        log_dir: directory for gradient logs.
        weight_log_dir: directory for mergeweight logs. Inferred from log_dir if omitted.
        batch_idx: batch index used in the output filename.
        dataset_name: dataset name used in the output filename.
    """
    assert grad_tensor.shape == (294, 8), "grad_tensor must have shape (294, 8)"
    assert mergeweight_tensor.shape == (294, 8), "mergeweight_tensor must have shape (294, 8)"
    
    if weight_log_dir is None:
        weight_log_dir = log_dir.replace("temp_grad_logs", "temp_weight_logs")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_log_dir, exist_ok=True)

    def process_tensor(tensor, save_dir, prefix, batch_idx, dataset_name):
        matrix = tensor.detach().cpu().numpy()
        csv_path = os.path.join(save_dir, f"{dataset_name}_{prefix}_batch_{batch_idx:06d}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f"{prefix}_col{i}" for i in range(8)] + ["mean", "variance"]
            writer.writerow(header)
            
            for row in matrix:
                mean = np.mean(row)
                var = np.var(row)
                
                formatted_row = [
                    "{:.8e}".format(val) for val in row
                ] + [
                    "{:.8e}".format(mean),
                    "{:.8e}".format(var)
                ]
                writer.writerow(formatted_row)

    process_tensor(grad_tensor, log_dir, "grad", batch_idx, dataset_name)
    
    process_tensor(mergeweight_tensor, weight_log_dir, "weight", batch_idx, dataset_name)

def save_mergeweight_ckpt(model, path, meta=None):
    payload = {
        "mergeweight": model.mergeweight.detach().cpu().clone(),
        "mergeweight_clamped": torch.clamp(model.mergeweight, min=0.0, max=1.0).detach().cpu().clone(),
        "shape": tuple(model.mergeweight.shape),
    }
    if meta:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path

def save_last_batch_grad(grad_tensor, path, meta=None):
    payload = {"grad": grad_tensor.detach().cpu().clone(), "shape": tuple(grad_tensor.shape)}
    if meta:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path

def setup_run_dir(args):
    os.makedirs(args.run_root, exist_ok=True)
    if not args.run_name:
        args.run_name = datetime.now().strftime("max_%Y%m%d_%H%M%S")
    args.run_dir = os.path.join(args.run_root, args.run_name)
    for sub in [
        "code",
        "config",
        "results",
        "logs/temp_grad_logs",
        "weights/temp_weight_logs",
        "ckpt",
    ]:
        os.makedirs(os.path.join(args.run_dir, sub), exist_ok=True)
    args.logspath = os.path.join(args.run_dir, "results/metrics.txt")
    args.grad_log_dir = os.path.join(args.run_dir, "logs/temp_grad_logs")
    args.weight_log_dir = os.path.join(args.run_dir, "weights/temp_weight_logs")
    args.ckpt_dir = os.path.join(args.run_dir, "ckpt")
    args.savepath = os.path.join(args.ckpt_dir, "mergeweight_final.pth")
    print(f"RUN_DIR={args.run_dir}")
    return args

def snapshot_code_and_config(args, script_path):
    code_dir = os.path.join(args.run_dir, "code")
    script_dir = os.path.dirname(os.path.abspath(script_path))
    src_dir = script_dir if os.path.isfile(os.path.join(script_dir, "task_vectors.py")) else _CODE_DIR
    exec_dir = script_dir
    exec_script = os.path.abspath(script_path)
    for name in [
        os.path.basename(script_path),
        "task_vectors.py",
        "model_vit.py",
        "retfound_dataset.py",
        "third_party.py",
    ]:
        src = os.path.join(src_dir, name) if name != os.path.basename(script_path) else script_path
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(code_dir, os.path.basename(name) if name == os.path.basename(script_path) else name))
    percentiles_file = os.path.join(_SUB_DIR, "config/percentiles_output.txt")
    if os.path.isfile(percentiles_file):
        shutil.copy2(percentiles_file, os.path.join(args.run_dir, "config/percentiles_output.txt"))
    with open(os.path.join(args.run_dir, "config/run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(args.run_dir, "run_command.sh"), "w") as f:
        f.write(
            "#!/bin/bash\n"
            f"export CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '6')}\n"
            f"cd {exec_dir}\n"
            f"nohup python \"{exec_script}\" "
            f"--percentile_threshold \"{args.percentile_threshold}\" "
            f"--run_root \"{args.run_root}\" "
            f"--run_name \"{args.run_name}\" "
            f"> \"{args.run_dir}/logs/nohup_output.log\" 2>&1 &\n"
        )
    os.chmod(os.path.join(args.run_dir, "run_command.sh"), 0o755)

def write_run_readme(args, last_step, mean_auc=None):
    lines = [
        "max merge run bundle",
        f"run_dir: {args.run_dir}",
        f"run_name: {args.run_name}",
        f"percentile_threshold: {args.percentile_threshold}",
        f"last_train_step: dataset={last_step.get('dataset')} batch={last_step.get('batch_idx')} loss={last_step.get('loss')}",
        "ckpt_policy: save only final mergeweight_final.pth; input pretrained/task-vector/head files are not copied",
        f"input_pretrained: {args.pretrained_checkpoint_path}  (recorded in config/run_args.json)",
        f"input_checkpoints: {args.checkpoints_path}",
        f"input_heads: {args.head_path}",
        f"final_merged_ckpt: {args.savepath}",
        f"metrics_csv: {args.logspath.replace('.txt', '.csv')}",
    ]
    if mean_auc is not None:
        lines.append(f"epoch2_mean_roc_auc: {mean_auc:.6f}")
    with open(os.path.join(args.run_dir, "README.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

def getargs():
    parser = argparse.ArgumentParser(description='merging')

    # paths
    parser.add_argument('--data_path', default=os.path.join(ADA_ROOT, "data/"),help='path to dataset')
    parser.add_argument('--pretrained_checkpoint_path', default=os.path.join(ADA_ROOT, "checkpoint_pretrained_fc_norm.pth"), type=str,help='path to pretrained checkpoint')
    parser.add_argument('--checkpoints_path',default=os.path.join(ADA_ROOT, "c_meh"), type=str)
 
    parser.add_argument('--savepath', default=os.path.join(ADA_ROOT, "mergedmodel"), type=str,help='path to save checkpoint')
    parser.add_argument('--logspath', default=os.path.join(_SUB_DIR, "results/metrics.txt"), type=str,help='path to save logs')
    parser.add_argument('--model', default='ViT-L/16', type=str,help='model name')
    parser.add_argument('--head_path', default=os.path.join(ADA_ROOT, "new/ViT-L-16"), type=str,help='path to checkpointhead')
    parser.add_argument('--percentile_threshold', default="50%", type=str, help='percentile threshold (e.g., "50%")')
    parser.add_argument('--run_root', default=os.path.join(_SUB_DIR, "runs"), type=str, help='root directory for archived runs')
    parser.add_argument('--run_name', default='', type=str, help='run subdirectory name; defaults to timestamp max_YYYYMMDD_HHMMSS')
    parser.add_argument('--grad_log_dir', default='', type=str, help='gradient log directory; defaults to run_dir/logs/temp_grad_logs')
    parser.add_argument('--weight_log_dir', default='', type=str, help='mergeweight log directory; defaults to run_dir/weights/temp_weight_logs')
    # training
    parser.add_argument('--epochs', default=1, type=int,help='number of epochs to train')
    parser.add_argument('--batch_size', default=1, type=int,help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
    parser.add_argument('--weight_decay', default=0.0001, type=float,help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
    parser.add_argument('--seed', default=42, type=int,help='random seed')
    parser.add_argument('--exam_datasets', nargs='+', default=['APTOS2019', 'OCTID','Glaucoma_fundus', 'IDRID', 'JSIEC', 'MESSIDOR2', 'PAPILA', 'Retina'], 
                        help="List of datasets to evaluate. Options: 'APTOS2019', 'OCTID','Glaucoma_fundus', 'IDRID', 'JSIEC', 'MESSIDOR2', 'PAPILA', 'Retina', etc.")

    # device
    parser.add_argument('--device', default='cuda:0', type=str, help='device to use for training (e.g., "cuda:0" or "cpu")')

    # The following arguments are kept aligned with RETFound.
    parser.add_argument('--input_size', default=224, type=int,help='images input size')
    # retfound:Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # retfound:Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    return parser.parse_args()

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)
import torchvision.transforms as T
def topil(tensor):
    """
    Convert a batch of PyTorch tensors to PIL images.
    Args:
        tensor (torch.Tensor): input tensor with shape B x C x H x W.

    Returns:
        list: PIL images.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor.cpu() * std + mean
    
    tensor = tensor.cpu()                                
    tensor = (tensor * 255).type(torch.uint8)                     

    pil_images = []
    to_pil = T.ToPILImage()
    for img_tensor in tensor:
        pil_images.append(to_pil(img_tensor))
    
    return pil_images

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def consisitecy_loss(x, x_aug, model):
    logits = model(x)
    logits_aug = model(x_aug)
    # 1. MEMO
    # logits.shape = [bs, num classes]  logits_aug.shape = [bs, num classes]
    # [bs, 1 + num_augs, num_classes]
    # Compute entropy after averaging.
    # logits_avg = torch.mean((logits, logits_aug), )

    # Pull logits and augmented logits closer.
    # loss = F.mse_loss(logits, logits_aug)

def evaluate(data_loader, model, args, dataset_name):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            images = batch[0].to(args.device, non_blocking=True)
            labels = batch[1].to(args.device, non_blocking=True)

            outputs = model(images, dataset_name)
            probabilities = outputs.softmax(dim=1)
            _, predictions = outputs.max(1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    n_classes = all_predictions.shape[1]
    all_labels_bin = label_binarize(all_labels, classes=np.arange(n_classes))

    accuracy = total_correct / total_samples * 100
    f1 = f1_score(all_labels, all_predictions.argmax(axis=1), average='weighted')
    roc_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr', average='weighted')
    hamming = hamming_loss(all_labels, all_predictions.argmax(axis=1))  
    jaccard = jaccard_score(all_labels, all_predictions.argmax(axis=1), average='weighted')
    precision = precision_score(all_labels, all_predictions.argmax(axis=1), average='weighted')
    recall = recall_score(all_labels, all_predictions.argmax(axis=1), average='weighted')
    average_precision = average_precision_score(all_labels_bin, all_predictions, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_predictions.argmax(axis=1))

    print(f"Dataset: {dataset_name}, Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f}")
    print(f"Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "hamming": hamming,  
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "kappa": kappa
    }

def write_metrics_to_csv(log_file, epoch, dataset_name, epoch_loss, metrics):
    """
    Append evaluation metrics to a CSV file and create the header if needed.
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as log:
        csv_writer = csv.writer(log)
        if not file_exists:
            csv_writer.writerow([
                "Epoch", "Dataset", "Epoch Loss", "Accuracy", "F1", "ROC AUC", "Hamming",
                "Jaccard", "Precision", "Recall", "Average Precision", "Kappa"
            ])
        csv_writer.writerow([
            epoch, dataset_name, epoch_loss,
            metrics["accuracy"], metrics["f1"], metrics["roc_auc"], metrics["hamming"],
            metrics["jaccard"], metrics["precision"], metrics["recall"],
            metrics["average_precision"], metrics["kappa"]
        ])
        log.flush()

def save_mergeweight_to_csv(model, filename='mergeweight2.csv', header_written=False):
    mergeweight = model.mergeweight.detach().cpu().numpy().flatten().tolist()
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not header_written:
            header = [f'weight_{i}' for i in range(len(mergeweight))]
            writer.writerow(header)
            header_written = True
        writer.writerow(mergeweight)
    return header_written

def train_and_log(model, optimizer, args, exam_datasets, thresholds):
    """
    Train the model and write logs.

    Args:
        model: model to train.
        optimizer: optimizer.
        args: parsed arguments.
        exam_datasets: list of datasets.
    """
    log_file = args.logspath.replace(".txt", ".csv")
    last_step = {}
    grad_log_dir = args.grad_log_dir or os.path.join(ADA_ROOT, "temp_grad_logs")
    weight_log_dir = args.weight_log_dir or os.path.join(ADA_ROOT, "temp_weight_logs")
    for epoch in range(args.epochs):
        for dataset_name in exam_datasets:
            dataset_test = build_dataset(is_train='test', args=args, dataset_name=dataset_name)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
                
            optimizer.zero_grad()
            total_loss = 0.0
            header_written=False

            for batch_idx, batch in enumerate(tqdm(data_loader_test, desc=f"Epoch {epoch+1}/{args.epochs}, Dataset: {dataset_name}")):
                original_images = batch[0].to(args.device, non_blocking=True)

                # Forward augmented images one by one to reduce memory usage.
                pil_image = topil(original_images)[0]
                original_out = model(original_images, dataset_name)
                original_probs = F.softmax(original_out[0], dim=-1).detach()
                del original_out

                aug_log_probs_list = []
                for _ in range(7):
                    aug_img = aug(pil_image).unsqueeze(0).to(args.device, non_blocking=True)
                    aug_out = model(aug_img, dataset_name)
                    aug_log_probs_list.append(F.log_softmax(aug_out[0], dim=-1))
                    del aug_img, aug_out

                augmented_log_probs = torch.stack(aug_log_probs_list, dim=0)
                original_probs_expanded = original_probs.unsqueeze(0).expand_as(augmented_log_probs)

                loss = F.kl_div(
                    input = augmented_log_probs,
                    target = original_probs_expanded,
                    reduction = 'batchmean',
                    log_target = False
                )                
                if loss<thresholds[dataset_name] or True:
                    print(f"entropy={loss.item():.4f}, threshold={thresholds[dataset_name]:.4f}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_snapshot = model.mergeweight.grad.detach().cpu().clone()
                    optimizer.step()
                    watch_grad(
                        grad_snapshot,
                        mergeweight_tensor=model.mergeweight.data.detach().cpu(),
                        log_dir=grad_log_dir,
                        weight_log_dir=weight_log_dir,
                        batch_idx=batch_idx,
                        dataset_name=dataset_name,
                    )
                    last_step = {
                        "dataset": dataset_name,
                        "batch_idx": batch_idx,
                        "grad": grad_snapshot,
                        "loss": float(loss.item()),
                        "epoch": epoch + 1,
                    }
                    total_loss += loss.item()

                del original_images, original_probs, augmented_log_probs, aug_log_probs_list, original_probs_expanded, loss
                torch.cuda.empty_cache()


            epoch_loss = total_loss / len(data_loader_test)
            print(f"Epoch Loss: {epoch_loss:.4f}")

            metrics = evaluate(data_loader_test, model, args, dataset_name=dataset_name)

            write_metrics_to_csv(log_file, epoch + 1, dataset_name, epoch_loss, metrics)
    return last_step

# layerwise
class Adamerge_model(torch.nn.Module):                                                                                                                  #"fc", "proj",
    def __init__(self, paramslist, exam_datasets, head_path=os.path.join(ADA_ROOT, "new/ViT-L-16"),names=None,model=None,update_layers= [ "qkv"]):
        super(Adamerge_model, self).__init__()
        self.task_count = len(paramslist) - 1
        self.paramslist = paramslist
        self.head_path = head_path
        self.update_layers = update_layers
        self.model=model
        self.names= names
        self.exam_datasets = exam_datasets
        self.classifier = self.get_classification_head(self.exam_datasets, self.head_path)  

        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.125

        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        self.mergeweight = torch.nn.Parameter(rlambdas) 

        self.isfirst = True

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        # params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        # for j, p in enumerate(zip(*self.paramslist)):
        #     # parameter lists for each task
        #     for (n, pi), lambdasi in zip(p, alph[j].cpu()):
        #         if n in self.update_layers:
        #             params[j][n] = pi * lambdasi
        #         else:
        #             params[j][n]  1/n

        params_list = []
        for j, p in enumerate(zip(*self.paramslist)):
            flag=True
            total = 0
            for (n, pi), lambdasi in zip(p, alph[j].cpu()):
                # update_layers= ["fc", "proj", "qkv"]
                if any(kw in n for kw in self.update_layers) and (pi.dim() in [1, 2]):
                    total += pi * lambdasi
                    if self.isfirst:
                        print(f"dynamically updated parameter: {n}")
                else:
                    total += pi * ((len(self.paramslist)-1) if flag else 1) / (len(self.paramslist)-1)
                    
                    if flag and self.isfirst:print("pretrained parameter group")
                    flag=False
                    if self.isfirst:
                        print(f"averaged parameter: {n}")
            params_list.append(total)
        self.isfirst = False
        params = tuple(params_list)

        device = inp.device

        params = tuple(p.to(device) for p in params)


        load_weights(self.model,self.names, params)

        feature = self.model(inp)  
        classification_head = getattr(self, f'classifier_{dataset_name}')
        out = classification_head(feature)
        return out

    def lambdas(self):
        task_lambdas = torch.clamp(self.mergeweight, min=0.0, max=1.0)
        device = self.mergeweight.device
        pretrain_lambdas = self.pretrain_lambdas.to(device)
        lambdass = torch.cat((pretrain_lambdas, task_lambdas), 1)
        return lambdass.requires_grad_()

    def collect_trainable_params(self):
        return [self.mergeweight]

    def get_classification_head(self, dataset_name, head_path):
        classification_heads = {}
        for dataset_name in self.exam_datasets:
            head_file = os.path.join(head_path, f"head_{dataset_name}.pt")
            if not os.path.exists(head_file):
                raise FileNotFoundError(f"classification head file not found: {head_file}")
            
            classification_head_dict = torch.load(head_file)
            print(f"classification head type: {type(classification_head_dict)}")
            
            weight = classification_head_dict['model.head.weight']
            bias = classification_head_dict['model.head.bias']
            
            classification_head = ClassificationHead(weight, bias)

            classification_heads[f"classifier_{dataset_name}"] = classification_head
            setattr(self, f'classifier_{dataset_name}', classification_head)
            print(f"loaded classification head for {dataset_name}: {head_file}")
        
        return classification_heads

class ClassificationHead(torch.nn.Module):
    def __init__(self, weight, bias):
        super(ClassificationHead, self).__init__()
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

def load_thresholds(percentiles_file, percentile_threshold):
    """
    Load thresholds from the percentile file.

    Args:
        percentiles_file (str): path to the percentile file.
        percentile_threshold (str): percentile key, e.g. "50%".

    Returns:
        dict: dataset name to threshold.
    """
    thresholds = {}
    with open(percentiles_file, "r") as file:
        current_dataset = None
        for line in file:
            if line.startswith("Dataset:"):
                current_dataset = line.split(":")[1].strip()
            elif line.strip().startswith(percentile_threshold):
                thresholds[current_dataset] = float(line.split(":")[1].strip())
    return thresholds


def set_seed(seed, deterministic=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = getargs()
    if args.seed < 0:
        args.seed = int(time.time()) % (2**31 - 1)
    set_seed(args.seed)
    print(f"training seed={args.seed}")
    args = setup_run_dir(args)
    snapshot_code_and_config(args, __file__)
    exam_datasets = args.exam_datasets
    percentiles_file = os.path.join(_SUB_DIR, "config/percentiles_output.txt")
    if not os.path.isfile(percentiles_file):
        percentiles_file = os.path.join(ADA_ROOT, "percentiles_output.txt")
    thresholds = load_thresholds(percentiles_file, args.percentile_threshold)

    pretrained_model_dict = torch.load(args.pretrained_checkpoint_path)['model']
    taskVector = [TaskVector(args.pretrained_checkpoint_path, args.checkpoints_path + '/RETFound_mae_meh-' + dataset_name + '/checkpoint-nohead.pth') for dataset_name in exam_datasets]

    pretrained_model = RETFound_mae(num_classes=0, global_pool=True, drop_path_rate=0.2)
    pretrained_model.load_state_dict(pretrained_model_dict, strict=False)  

    model = ModelWrapper(pretrained_model, exam_datasets)
    model = model.to('cpu')
    _, names = make_functional(model)

    # Build parameter lists.
    paramslist = []
    paramslist += [tuple((_,v.detach().requires_grad_().cpu()) for _, v in pretrained_model_dict.items())] 
    paramslist += [tuple((_, v.detach().requires_grad_().cpu()) for _, v in tv.vector.items()) for i, tv in enumerate(taskVector)]  # task vectors
    
    del taskVector
    torch.cuda.empty_cache()


    # Initialize the merge model.
    model = Adamerge_model(paramslist, exam_datasets, args.head_path, names=names, model=model)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.collect_trainable_params(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))  

    # test(pretrained_pth=args.pretrained_checkpoint_path, fine_tuned_pth=args.checkpoints_path + '/RETFound_mae_meh-' + exam_datasets[0] + '/checkpoint-nohead.pth', model=model)
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")
        if name != "mergeweight":
            param.requires_grad = False

    # print("Trainable parameters in the model:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Name: {name}, Shape: {param.shape}")
    #         print("+_+")

    log_file = args.logspath.replace(".txt", ".csv")
    # for dataset_name in exam_datasets:
    #     dataset_test = build_dataset(is_train='test', args=args, dataset_name=dataset_name)
    #     sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test, sampler=sampler_test,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )
    #     metrics = evaluate(data_loader_test, model, args, dataset_name=dataset_name)
    #     # Record initial evaluation results as epoch 0.
    #     write_metrics_to_csv(log_file, 0, dataset_name, 0, metrics)

    last_step = train_and_log(model, optimizer, args, exam_datasets, thresholds)

    final_meta = {
        "stage": "after_full_train_before_epoch2_eval",
        "last_dataset": last_step.get("dataset"),
        "last_batch_idx": last_step.get("batch_idx"),
        "last_loss": last_step.get("loss"),
        "percentile_threshold": args.percentile_threshold,
        "run_name": args.run_name,
    }
    save_mergeweight_ckpt(model, args.savepath, meta=final_meta)
    print(f"saved final merged ckpt: {args.savepath}")
    print("input checkpoints were not copied; paths are recorded in config/run_args.json")

    epoch2_aucs = []
    for dataset_name in exam_datasets:
        dataset_test = build_dataset(is_train='test', args=args, dataset_name=dataset_name)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        metrics = evaluate(data_loader_test, model, args, dataset_name=dataset_name)
        epoch2_aucs.append(metrics["roc_auc"])
        # Record evaluation results as epoch 2.
        write_metrics_to_csv(log_file, 2, dataset_name, 0, metrics)

    mean_auc = sum(epoch2_aucs) / len(epoch2_aucs) if epoch2_aucs else None
    write_run_readme(args, last_step, mean_auc=mean_auc)
    print(f"run archived at: {args.run_dir}")
    if mean_auc is not None:
        print(f"Epoch2 mean ROC AUC: {mean_auc:.6f}")
