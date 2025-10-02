import os
import sys
os.chdir(sys.path[0])
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore", message=".*MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid.*indexing")
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,)
from mmcv.utils import DictAction
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score

from model import digit_version
from thop import profile
from model.apis import single_gpu_test, set_random_seed
from model.datasets import build_dataloader, build_dataset
from model.models import build_segmentor
from model.utils import build_difusionseg, get_device,PrintModelInfo,count_params
"""please use RTX4090 to fork the results"""
GPU=0
CONFIG='./configs/IOMSG_config.py'
CHECKPOINT='./exps/GAMMA_rgb_oct_cls_DSC_CMFA/iter_100000.pth'
OUT="./out/"
METRIC='mDice' # mIoU, mDice

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config',default=CONFIG,
                        help='test config file path')
    parser.add_argument('--checkpoint',default=CHECKPOINT,
                        help='checkpoint file')
    parser.add_argument('--work-dir',
        help=('if specified, the evaluation metric results will be dumped into the directory as json'))
    parser.add_argument('--aug-test', action='store_true',
                        help='Use Flip and Multi scale aug')
    parser.add_argument('--out', 
                        help='output result file in pickle format')
    parser.add_argument('--format-only',action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument('--eval',type=str,nargs='+',default=METRIC,
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU" for generic datasets')
    parser.add_argument('--show', action='store_true', 
                        help='show results')
    parser.add_argument('--show-dir', default=OUT,
                        help='directory where painted images will be saved')
    parser.add_argument('--gpu-collect',action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument('--gpu-id',type=int,default=GPU,
        help='id of gpu to use ''(only applicable to non-distributed testing)')
    parser.add_argument('--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument('--options',nargs='+',action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--eval-options',nargs='+',action=DictAction,
        help='custom options for evaluation')
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],default='none',
        help='job launcher')
    parser.add_argument('--opacity',type=float,default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--seed',type=int,default=2002,
        help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args

def evaluate_predictions(pred_label, label_file, average='macro'):
 # 1️⃣ 读取真实标签
    true_label_dict = {}
    with open(label_file, "r") as f:
        for line in f:
            key, val = line.strip().split()
            true_label_dict[key] = int(val)

    # 2️⃣ 只计算预测过的样本
    common_keys = [k for k in true_label_dict if k in pred_label]
    y_true = [true_label_dict[k] for k in common_keys]
    y_pred = [pred_label[k] for k in common_keys]

    # 3️⃣ 计算总体指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}\n")

    # 4️⃣ 计算每类指标
    labels = sorted(list(set(y_true + y_pred)))  # 取出现过的类别
    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    print("Per-Class Metrics:")
    print("Class | Precision | Recall | F1")
    for i, cls in enumerate(labels):
        print(f"{cls:5} | {precision_per_class[i]:9.4f} | {recall_per_class[i]:6.4f} | {f1_per_class[i]:5.4f}")

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Per-Class": {
            "labels": labels,
            "precision": precision_per_class,
            "recall": recall_per_class,
            "f1": f1_per_class
        }
    }

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    set_random_seed(args.seed)

    """set cudnn_benchmark"""
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]
    """init distributed env first, since logger depends on the dist info."""
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    """build the dataloader"""
    dataset = build_dataset(cfg.data.test)
    """The default loader config"""
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    """The overall dataloader settings"""
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader']
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    """build the dataloader"""
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    """build the model and load checkpoint"""
    cfg.model.train_cfg = None
    cfg.device = get_device()
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')).to(cfg.device)
    PrintModelInfo(model)
    count_params(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    """clean gpu memory when starting a new evaluation."""
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    tmpdir = None
    cfg.device = get_device()
    #if not distributed:
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    model = revert_sync_batchnorm(model)
    model = build_difusionseg(model, cfg.device, device_ids=cfg.gpu_ids)
    results,pred_label = single_gpu_test(
        model,
        data_loader,
        args.show,
        args.show_dir,
        False,
        args.opacity,
        pre_eval=args.eval is not None and not False,
        format_only=args.format_only or False,
        format_args=eval_kwargs)

    eval_kwargs.update(metric=args.eval)
    metric = dataset.evaluate(results, **eval_kwargs)
    evaluate_predictions(pred_label, '/home/BlueDisk/Dataset/HKU/GAMMA/seg/class_label.txt')
    
if __name__ == '__main__':
    main()
