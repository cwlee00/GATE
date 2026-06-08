import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import sys
from copy import deepcopy

import math
import numpy as np
import wandb
import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
from collections import defaultdict

from config.basic import ConfigBasic
from utils.util import write_log, get_current_time, to_np, make_dir, log_configs, save_ckpt, set_wandb
from utils.util import AverageMeter, extract_embs, evaluate_metric, find_kNN
from utils.loss_util import compute_center_loss, heteroscedastic_nll, RnCLoss
from networks.util import prepare_model
from data.get_datasets_tr_OLbasic_val_NN import get_datasets_AGE

import shutil
import argparse


def parse_args():
    ############ cmd process ##############
    parser = argparse.ArgumentParser(description='SOL')
    parser.add_argument('--dataset', type=str, default='clap',
                        help='dataset')
    parser.add_argument('--fold', type=str, default='eval_on_test',
                        help="train on which cuda device")
    parser.add_argument('--seed', type=int, default=999,
                        help="train on which cuda device")
    parser.add_argument('--gpu', type=int, default=0,
                        help="train on which cuda device")
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='dataset fold number')
    parser.add_argument('--metric', type=str, default='L2',
                        help='dataset')
    parser.add_argument('--label_diff', type=str, default='l2',
                        help='dataset')
    parser.add_argument('--version', type=str, default='try1',
                        help='dataset')
    parser.add_argument('--similarity_type', type=str, default='L2',
                        help='dataset')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='dataset fold number')
    parser.add_argument('--temp', type=float, default=2.0,
                        help='dataset fold number')
    parser.add_argument('--lambda_sigma', type=float, default=0.1,
                        help='dataset fold number')
    parser.add_argument('--query_update_layers', type=list, default=[6,7,8],
                        help='dataset fold number')

    parser.add_argument('--num_layers', type=int, default=1,
                        help="train on which cuda device")
    parser.add_argument('--wd', type=float, default=0.05,
                        help='dataset fold number')
    parser.add_argument('--nll', type=float, default=1.0,
                        help='dataset fold number')
    parser.add_argument('--center', type=float, default=0.1,
                        help='dataset fold number')
    parser.add_argument('--lr_decay_rate', type=float, default=0.0001,
                        help='dataset fold number')
    parser.add_argument('--epochs', type=int, default=100,
                        help="train on which cuda device")
    ########## cmd end ############

    args = parser.parse_args()

    return args


def set_local_config(cfg, args):
    # Dataset
    cfg.seed = args.seed
    random_seed = cfg.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    cfg.norm_age = False
    cfg.dataset = args.dataset
    cfg.logscale = False
    cfg.fold = args.fold
    cfg.set_dataset()
    cfg.tau = 0

    # Model
    cfg.model = 'GATE'
    cfg.backbone = 'vitB16'
    cfg.gauss_mode = True

    if cfg.dataset == 'morph':
        cfg.k = 2
    elif cfg.dataset == 'clap':
        cfg.k = 42
    elif cfg.dataset == 'ageDB':
        cfg.k = 18
    elif cfg.dataset == 'utk':
        cfg.k = 58
    elif cfg.dataset == 'cacd':
        cfg.k = 58
    elif cfg.dataset == 'adience':
        cfg.k = 58
    else:
        cfg.k = np.arange(2,60,2)
    cfg.epochs = args.epochs
    cfg.scheduler = 'cosine'
    cfg.lr_decay_epochs = [10, 30]
    cfg.lr_decay_rate = args.lr_decay_rate
    cfg.period = 3

    cfg.margin = 0.05
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60  # 60 Fold1, 58 Fold0 setting D // 56 setting c // 58 setting B // 55 setting A
    cfg.drct_wieght = 1
    cfg.start_norm = False
    #############################################
    cfg.inference = 'knn'
    cfg.tr_loader_version = 'ContrastiveOrder'
    cfg.learning_rate = args.lr
    cfg.temp = args.temp
    cfg.metric = args.metric
    cfg.label_diff = args.label_diff
    cfg.similarity_type = args.similarity_type
    cfg.version = args.version

    if cfg.version == 'MAE':
        cfg.label_diff = 'l1'

    cfg.epsilon = args.epsilon
    cfg.test_batch_size = 100

    cfg.save_feats = False
    cfg.augmentation = True


    if cfg.augmentation:
        if cfg.dataset == 'utk':
            cfg.batch_size = 16
        else:
            cfg.batch_size = 64                    
    else:
        if cfg.dataset == 'utk':
            cfg.batch_size = 32
        else:
            cfg.batch_size = 128

    cfg.num_heads = 8
    cfg.lambda_nll = args.nll
    cfg.lambda_center = args.center
    cfg.lambda_conord = 1.0
    cfg.num_layers = args.num_layers
    cfg.weight_decay = args.wd
    cfg.lambda_sigma = args.lambda_sigma
    cfg.query_update_layers = args.query_update_layers
    ##################################################
    # Log
    cfg.wandb = False
    cfg.experiment_name = f'GATE-s'
    cfg.save_folder = f'../GATE_results/{cfg.dataset}/results_{cfg.fold}_{cfg.model}/{cfg.experiment_name}/Epochs{cfg.epochs}_PREFIX_{cfg.margin}_tau{cfg.tau}_{cfg.model}_{cfg.backbone}_lr{cfg.learning_rate}_{get_current_time()}'
    make_dir(cfg.save_folder)
                    
    cfg.n_gpu = torch.cuda.device_count()
    cfg.num_workers = 8
    return cfg


def main():
    
    cfg = ConfigBasic()
    args = parse_args()
    cfg = set_local_config(cfg, args)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    # dataloader
    loader_dict = get_datasets_AGE(cfg)
    if cfg.norm_age:
        cfg.n_ranks = loader_dict['train'].dataset.ranks.max() + 1
    else:
        if cfg.dataset == 'ageDB':
            cfg.n_ranks = 102
        else:
            cfg.n_ranks = 101
    print(f'[*] {cfg.n_ranks} ranks exist. ')
    ####################################################
    cfg.ref_point_num = cfg.n_ranks
    cfg.fiducial_point_num = cfg.n_ranks

    cfg.sigma_min = 0.1
    cfg.sigma_max = cfg.n_ranks
    #####################################################

    # model
    model = prepare_model(cfg)
    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    # ====== PARAM GROUP 설정 ======
    param_groups = []
    for key, value in dict(model.named_children()).items():
        if key == 'encoder':  # Backbone
            param_groups.append({"params": value.parameters(), "lr": cfg.learning_rate})
        else:
            param_groups.append({"params": value.parameters(), "lr": cfg.learning_rate*10})

    param_groups += [{"params": model.ref_points, "lr": cfg.learning_rate*10}]
    # ====== OPTIMIZER  ======
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs,
                                                         eta_min=cfg.learning_rate * cfg.lr_decay_rate)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)
    else:
        scheduler = None

    criterion = RnCLoss(temperature=cfg.temp)

    if torch.cuda.is_available():
        model = model.cuda()

    val_mae_best = np.inf
    val_cs_best = 0.0
    log_dict = dict()
    # init loss matrix
    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]

    for epoch in range(cfg.epochs):
        print("==> training...")
        if cfg.scheduler:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch}] Current LR: {current_lr}")

        time1 = time.time()
        train_loss, loss_record = train(epoch, loader_dict['train'], model, optimizer, criterion, cfg,
                                        prev_loss_record=loss_record)

        if cfg.scheduler:
            scheduler.step()

        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        print('==> validation...')
        if cfg.dataset == 'clap':
            val_mae, val_eps, val_cs= validate_AGE(loader_dict, model, cfg)
            if val_mae < val_mae_best:
                val_mae_best = val_mae
                save_ckpt(cfg, model, f'ep_{epoch}_MAE_{val_mae:.3f}_EPS{val_eps:.4f}_CS_{val_cs:.4f}.pth')
        else:             
            val_mae, val_cs = validate_AGE(loader_dict, model, cfg)
        
            if val_mae < val_mae_best:
                val_mae_best = val_mae
                save_ckpt(cfg, model, f'ep_{epoch}_MAE_{val_mae:.3f}_CS_{val_cs:.4f}.pth')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['Train Loss'] = train_loss
            log_dict['Val Mae'] = val_mae
            log_dict['LR'] = scheduler.get_lr()[0] if scheduler else cfg.learning_rate
            wandb.log(log_dict)

    print('[*] Training ends')



def train(epoch, train_loader, model, optimizer, criterion, cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dist_losses = AverageMeter()
    nll_losses = AverageMeter()
    center_losses = AverageMeter()

    loss_record = deepcopy(prev_loss_record)
    end = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for idx, (images, _, ranks, _) in enumerate(train_loader):
        if cfg.augmentation:
            images = torch.cat(images)
            bsz = ranks.shape[0]
        else:
            bsz = ranks.shape[0] // 2

        if torch.cuda.is_available():
            images = images.cuda()
            ranks = ranks.cuda()

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():
            # ===================forward=====================
            H, age_query = model.forward_features(images)
            mu, sigma = model.predict_mu_sigma(H)
            features = model.forward_transformer(H, age_query, mu, sigma)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # =====================loss======================
            # compute loss
            dist_loss = criterion(features, ranks.unsqueeze(1), cfg)

            # heteroscedastic NLL (if you want, alternatively keep your existing dist_loss)
            if cfg.augmentation:
                nll_loss = heteroscedastic_nll(mu, sigma, ranks.repeat(2), cfg)
                center_loss = compute_center_loss(torch.cat(torch.unbind(features, dim=1), dim=0), ranks.repeat(2),
                                                    model.ref_points, cfg)
            else:
                nll_loss = heteroscedastic_nll(mu, sigma, ranks, cfg)
                center_loss = compute_center_loss(torch.cat(torch.unbind(features, dim=1), dim=0), ranks,
                                                    model.ref_points, cfg)
            total_loss = dist_loss + cfg.lambda_nll * nll_loss + cfg.lambda_center * center_loss

        losses.update(total_loss.item(), ranks.size(0))
        dist_losses.update(dist_loss.item(), ranks.size(0))
        nll_losses.update(nll_loss.item() * cfg.lambda_nll, ranks.size(0))
        center_losses.update(center_loss.item() * cfg.lambda_center, ranks.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        scaler.scale(total_loss).backward() 
        scaler.step(optimizer)
        scaler.update()
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % cfg.print_freq == 0:
            mu_mean = mu.mean().item()
            mu_std = mu.std().item()
            sigma_mean = sigma.mean().item()
            sigma_std = sigma.std().item()

            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Dist-Loss {dist_losses.val:.4f}\t'
                      f'NLL-Loss {nll_losses.val:.4f}\t'
                      f'Center-Loss {center_losses.val:.4f}\t'
                      f'mu: mean={mu_mean:.3f}, std={mu_std:.3f}\t'
                      f'sigma: mean={sigma_mean:.3f}, std={sigma_std:.3f}\t'
                      )
            sys.stdout.flush()

    return losses.avg, loss_record

def validate_AGE(loader_dict, model, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train = extract_embs(model, loader_dict['train_for_val'])
    embs_train = embs_train.cuda()

    embs_test = extract_embs(model, loader_dict['val'])
    embs_test = embs_test.cuda()

    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.test_batch_size))

    test_labels = loader_dict['val'].dataset.labels
    train_labels = loader_dict['train_for_val'].dataset.labels

    preds_all = defaultdict(list)
    k = cfg.k
    with torch.no_grad():
        end = time.time()
        for idx in range(n_batch):
            data_time.update(time.time() - end)
            i_st = idx * cfg.test_batch_size
            i_end = min(i_st + cfg.test_batch_size, n_test)

            # ===================meters=====================
            vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=k,
                                    metric=cfg.metric)
            inds = np.squeeze(to_np(inds), 0)


            nn_labels = train_labels[inds[:, :k]]
            pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
            preds_all[k].append(pred_mean)
    preds_all[k] = np.concatenate(preds_all[k])
    pred_age = preds_all[k]

    p_age = pred_age
    t_labels = test_labels
    mae, cs, acc = evaluate_metric(p_age, t_labels)
   
    write_log(cfg.logfile, f'MAE: {mae:.3f}, CS:{cs:.4f}, Acc : {acc * 100:.2f}')

    return mae, cs

if __name__ == "__main__":
    main()
