import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
from PIL import Image
from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn.functional as F
from CTAC.model_attn_aff import WeTr
from torch.utils.data import DataLoader
from datasets import brats as brats
import logging
from texttable import Texttable
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="test", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path", default=r"....pth", type=str, help="model_path")

parser.add_argument("--backbone", default='mit_b1', type=str)
parser.add_argument("--data_folder", default=r'', type=str, help="dataset folder")
parser.add_argument("--list_folder", default=r'', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=4, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
parser.add_argument("--log_step", type=int, default=1, help="gpu ids, example: 0,1")
parser.add_argument("--num_workers", type=int, default=4, help="gpu ids, example: 0,1")
parser.add_argument("--num_channels", default=3, type=int, help="aux_layer")
parser.add_argument("--stride", default=[4, 2, 2, 1], help="random rescale in training")



def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def format_tabs_metric(scores, name_list, cat_list=None):
    _keys_iou = list(scores[0]['iou'].keys())
    _values_iou = []
    _keys_Dice = list(scores[0]['Dice'].keys())
    _values_Dice = []
    _keys_HD95 = list(scores[0]['HD95'].keys())
    _values_HD95 = []


    for i in range(len(scores)):
        _values_iou.append(list(scores[i]['iou'].values()))
        _values_Dice.append(list(scores[i]['Dice'].values()))
        _values_HD95.append(list(scores[i]['HD95'].values()))

    _values_iou = np.array(_values_iou) * 100
    _values_Dice = np.array(_values_Dice) * 100
    _values_HD95 = np.array(_values_HD95)

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys_iou)):
        if i == 0:
            t.add_row([cat_list[i]] + list(_values_iou[:, i]) + list(_values_Dice[:, i]) + ['None'])
        else:
            t.add_row([cat_list[i]] + list(_values_iou[:, i]) + list(_values_Dice[:, i]) + list(_values_HD95[:, i-1]) )

    t.add_row(["mean"] + list(_values_iou.mean(1)) + list(_values_Dice.mean(1)) + list(_values_HD95.mean(1)) )

    return t.draw()

def validate(args=None):

    model = WeTr(
        backbone=args.backbone,
        num_classes=args.num_classes,
        embedding_dim=256,
        stride=args.stride,
        pretrained=False,
        pooling=args.pooling,
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.to(device)


    val_dataset = brats.brats2021SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split='test',
        stage='test',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            )

    # 在 CPU 内存中累积结果
    all_preds = []
    all_gts = []
    model.eval()
    with torch.no_grad(), torch.cuda.device(0):

        for idx, (names, inputs, labels,cls_label) in enumerate(val_loader):
            inputs = inputs.float().to(device)

            labels = labels.to(device)
            img_name = [item + ".png" for item in names]
            cls, segs, _, attn_pred = model(inputs, )

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            pred = torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.uint8)
            _masks = labels.detach().cpu().numpy().astype(np.uint8)

            # 转换为one-hot格式（MONAI要求）
            # 累积结果
            cmpute1 = torch.argmax(resized_segs, dim=1).cpu().numpy()
            cmpute2 = labels.cpu().numpy()
            all_preds.append(cmpute1)  # 转为 numpy 数组
            all_gts.append(cmpute2)

            pred[pred == 3] = 4
            for item, name in zip(pred, img_name):
                item = Image.fromarray(item, mode="L").convert("P")
                item.putpalette(palette)
                item.save(os.path.join(preds_save_dir, name))

        # 最终指标计算（批量处理）
        # 合并所有结果
        all_preds = np.concatenate(all_preds, axis=0)  # [N, H, W]
        all_gts = np.concatenate(all_gts, axis=0)  # [N, H, W]
        # 转换为 one-hot 格式（MONAI 要求）
        num_classes = args.num_classes
        pred_onehot = np.eye(num_classes)[all_preds].transpose(0, 3, 1, 2)  # [N, C, H, W]
        gt_onehot = np.eye(num_classes)[all_gts].transpose(0, 3, 1, 2)
        # 计算指标（CPU 版）
        dice = compute_dice(torch.from_numpy(pred_onehot), torch.from_numpy(gt_onehot), include_background=True,
                            ignore_empty=False)
        hd95 = compute_hausdorff_distance(torch.from_numpy(pred_onehot), torch.from_numpy(gt_onehot), percentile=95,
                                          include_background=True)
        iou = compute_iou(torch.from_numpy(pred_onehot), torch.from_numpy(gt_onehot), include_background=True,
                          ignore_empty=False)

        # 转换为类别字典
        results = {
            "Dice": {brats.class_list[i]: torch.nanmean(dice, dim=0)[i].item() for i in range(0, num_classes)},
            "HD95": {brats.class_list[i]: torch.nanmean(hd95, dim=0)[i].item() for i in range(1, num_classes)},
            "iou": {brats.class_list[i]: torch.nanmean(iou, dim=0)[i].item() for i in range(0, num_classes)}
        }
        tab_results = format_tabs_metric([results], ["IOU", "DICE", "HD95"],
                                                 cat_list=brats.class_list)
        logging.info("\n" + tab_results)

    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    model_name = args.model_path.split("checkpoints")[1].split(".")[0]


    preds_save_dir = os.path.join(base_dir, "test/preds/")
    logs_dir = os.path.join(base_dir, "test/logs/")
    log_path = os.path.join(base_dir, "test/logs/test.log")



    colormap_path = os.path.join(args.list_folder, "grayscale_colormap.txt")
    with open(colormap_path, "r", encoding="utf-8") as f:
        palette = eval(f.read().strip())  # convert to list object

    for _ in (preds_save_dir,logs_dir):
        os.makedirs(_, exist_ok=True)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    print(args)

    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f"cuda:{args.gpus}")
    else:
        device = torch.device("cpu")
    fix_seed(1)
    validate(args=args)
