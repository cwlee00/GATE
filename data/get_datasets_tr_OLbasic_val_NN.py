import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import basic
from utils.util import TwoCropTransform


def get_datasets_AGE(cfg):
    tr_std = None
    te_std = None
    if 'morph' in cfg.dataset:
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
        tr_ages = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
        te_ages = te_list[:, cfg.lb_idx]

    elif cfg.dataset =='clap':
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_ages = tr_list[:, cfg.lb_idx]
        tr_imgs = [f'{img_root}/{tr_list[i, 3]}/{tr_list[i, cfg.img_idx]}' for i in range(len(tr_list))]
        tr_std = tr_list[:, 2]
        #
        # # debug for n_ranks and margin relation
        # idx = np.argwhere(tr_ages < 60).flatten()
        # tr_ages = tr_ages[idx]
        # tr_imgs = np.array(tr_imgs)[idx]
        # tr_std = tr_std[idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{te_list[i, 3]}/{te_list[i, cfg.img_idx]}' for i in range(len(te_list))]
        te_ages = te_list[:, cfg.lb_idx]
        te_std = te_list[:, 2]
        #
        # # debug for n_ranks and margin relation
        # idx = np.argwhere(te_ages < 60).flatten()
        # te_ages = te_ages[idx]
        # te_imgs = np.array(te_imgs)[idx]
        # te_std = te_std[idx]

    elif cfg.dataset == 'ageDB':
        img_root = cfg.img_root
        split = pd.read_csv(cfg.data_file)['split']

        tr_list = pd.read_csv(cfg.data_file)[split == 'train']
        tr_ages = tr_list['age'].to_numpy()
        tr_imgs = [os.path.join(img_root, tr_list['path'].to_numpy()[i]) for i in range(len(tr_list))]

        te_list = pd.read_csv(cfg.data_file)[split == 'test']
        te_ages = te_list['age'].to_numpy()
        te_imgs = [os.path.join(img_root, te_list['path'].to_numpy()[i]) for i in range(len(te_list))]

    elif cfg.dataset == 'utk':
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file)
        tr_ages = tr_list['age'].to_numpy()
        tr_imgs = [os.path.join(img_root, tr_list['filename'].to_numpy()[i]) for i in range(len(tr_list))]

        te_list =pd.read_csv(cfg.test_file)
        te_ages = te_list['age'].to_numpy()
        te_imgs = [os.path.join(img_root, te_list['filename'].to_numpy()[i]) for i in range(len(te_list))]

    elif cfg.dataset == 'cacd':
        img_root = cfg.img_root
        split = pd.read_csv(cfg.data_file)['fold']
        tr_list = pd.read_csv(cfg.data_file)[split == 'train']
        tr_ages = tr_list['age'].to_numpy()
        tr_imgs = [os.path.join(img_root, tr_list['filename'].to_numpy()[i]) for i in range(len(tr_list))]

        te_list = pd.read_csv(cfg.data_file)[split == 'test']
        te_ages = te_list['age'].to_numpy()
        te_imgs = [os.path.join(img_root, te_list['filename'].to_numpy()[i]) for i in range(len(te_list))]
   
    elif cfg.dataset =='adience':
        label_map = np.array([1, 5, 10.5, 17.5, 34, 40.5, 50.5, 80.5])

        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            # tr_imgs = data['data']
            tr_imgs = data['img_path']
            
            tr_ages = data['age']
            tr_ages_scaled = np.round(label_map[data['age'].astype('long')]).astype('long')  # age를 매핑된 값으로 변환

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            # te_imgs = data['data']
            te_imgs = data['img_path']
            te_ages = data['age']
    elif cfg.dataset == 'FGNet':
        tr_list = pd.read_csv(cfg.data_file)
        tr_ages = tr_list['age'].to_numpy()
        tr_imgs = np.array([t.replace('/media/cwlee/HDD', '/hdd1/cwlee') for t in tr_list['path']])
        
        te_list = pd.read_csv(cfg.data_file)
        te_ages = te_list['age'].to_numpy()
        te_imgs = np.array([t.replace('/media/cwlee/HDD', '/hdd1/cwlee') for t in te_list['path']])
    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']


    if cfg.augmentation:
        if cfg.dataset == 'adience':
            trainset = basic.Basic(tr_imgs, tr_ages_scaled, TwoCropTransform(cfg.transform_tr), is_filelist=cfg.is_filelist, return_ranks=True,norm_age=cfg.norm_age)
        else:
            trainset = basic.Basic(tr_imgs, tr_ages, TwoCropTransform(cfg.transform_tr), is_filelist=cfg.is_filelist, return_ranks=True,norm_age=cfg.norm_age)
    else:
        trainset = basic.Basic(tr_imgs, tr_ages, cfg.transform_tr, is_filelist=cfg.is_filelist, return_ranks=True, norm_age=cfg.norm_age)
    loader_dict = dict()
    if cfg.dataset == 'utk':
        loader_dict['train'] = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    else:
        loader_dict['train'] = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_te, is_filelist=cfg.is_filelist, norm_age=False),
                                    batch_size=cfg.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=cfg.num_workers)

    loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False),
                                     batch_size=cfg.test_batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict







