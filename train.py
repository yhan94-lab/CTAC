import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from datasets import brats as brats
import utils.losses as lossall
from CTAC.PAR import PAR
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.optimizer import PolyWarmupAdamW
from CTAC.model_attn_aff import WeTr
import utils.pyutils as pyutils
import utils.camutils as camutils


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=r'**.yaml',
                    type=str,
                    help="config")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(rank,filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if rank == 0:
        fHandler = logging.FileHandler(filename, mode='w')
        fHandler.setFormatter(logFormatter)
        logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    # time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def train(rank, world_size,cfg):
    print('*********************************************************************************************************')
    print(rank, world_size)
    dist.init_process_group(backend=cfg.train.backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    setup_logger(filename=os.path.join(cfg.work_dir.tb_logger_dir, f'train_{rank}.log'), rank=rank)
    logging.info('Pytorch version: %s' % torch.__version__)
    logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
    logging.info('\nconfigs: %s' % cfg)
    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), cfg.train.samples_per_gpu))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = brats.brats2021ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )


    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              # shuffle=True,
                              num_workers=cfg.train.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)


    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=cfg.backbone.pooling, )
    logging.info('\nNetwork config: \n%s' % (wetr))
    param_groups = wetr.get_param_groups()
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    wetr.to(device)
    par.to(device)


    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    wetr = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wetr)
    wetr = DistributedDataParallel(wetr, device_ids=[rank], find_unused_parameters=True)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))

    loss_layer = lossall.DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))
    list_meter = pyutils.ListMeter()
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, mask, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, mask, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)

        cams, aff_mat = camutils.multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)

        cls, segs, attns, attn_pred = wetr(inputs, seg_detach=cfg.train.seg_detach)

	# MSACAM
        sm_x = cams.clone().detach() + aff_mat
        sm_x = camutils.norm_map_cam(sm_x)
        fused_x = sm_x * cams


        refined_pseudo_label = camutils.refine_cams_with_bkg_v2(par, inputs_denorm, cams=fused_x, cls_labels=cls_labels,
                                                                cfg=cfg,
                                                                img_box=img_box)
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        seg_loss = lossall.get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)


        cls_loss = torch.where(torch.isnan(cls_loss), torch.tensor(0.0, device=cls_loss.device, requires_grad=True), cls_loss)
        seg_loss = torch.where(torch.isnan(seg_loss), torch.tensor(0.0, device=seg_loss.device, requires_grad=True), seg_loss)



        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss + 0.0 * seg_loss
        else:
            loss = 1.0 * cls_loss + 0.1 * seg_loss

        avg_meter.add({'loss': loss.item(), 'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item(),
                       })
        acc, recall = pyutils.multi_label_accuracy(cls, cls_labels)
        avg_meter.add({'acc': acc, 'recall': recall})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs, dim=1, ).cpu().numpy().astype(np.int8)
            gts = refined_pseudo_label.cpu().numpy().astype(np.int8)

            seg_mAcc = (preds == gts).sum() / preds.size

            logging.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; loss: %.4f, cls_loss: %.4f, seg_loss: %.4f, pseudo_seg_mAcc: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, avg_meter.get('loss'), avg_meter.get('cls_loss'), avg_meter.get('seg_loss'),seg_mAcc))

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth" % (n_iter + 1))
            torch.save(wetr.state_dict(), ckpt_name)
            return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir,timestamp, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.tb_logger_dir)
    cfg.work_dir.history = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.history)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.history, exist_ok=True)
    ## fix random seed
    setup_seed(1)
    torch.multiprocessing.spawn(train, args=(cfg.dist.world_size, cfg), nprocs=cfg.dist.world_size, join=True)
