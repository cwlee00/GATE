import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from utils.util import TwoCropTransform
from data.multi_datasets import basic

import random
from torch.utils.data.sampler import Sampler

import random
from torch.utils.data import DataLoader
import json

import numpy as np
import random
from torch.utils.data import Sampler



class BalancedDatasetBatchSampler(Sampler):
    """
    Ensures each batch contains approximately the same number of samples
    from each dataset (useful when dataset sizes are highly imbalanced).
    """
    def __init__(self, dataset, batch_size, dataset_names, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_names = np.array(dataset_names)
        self.shuffle = shuffle

        # Identify dataset groups
        self.unique_datasets = np.unique(self.dataset_names)
        self.num_datasets = len(self.unique_datasets)

        # Per-dataset sample indices
        self.indices_per_dataset = {
            d: np.where(self.dataset_names == d)[0].tolist()
            for d in self.unique_datasets
        }

        # How many samples per dataset per batch
        self.samples_per_dataset = batch_size // self.num_datasets

        # Compute how many batches fit the smallest dataset
        self.min_len = min(len(v) for v in self.indices_per_dataset.values())
        self.num_batches = self.min_len // self.samples_per_dataset

    def __iter__(self):
        # Shuffle each dataset individually
        if self.shuffle:
            for d in self.unique_datasets:
                random.shuffle(self.indices_per_dataset[d])

        for b in range(self.num_batches):
            batch_indices = []
            for d in self.unique_datasets:
                start = b * self.samples_per_dataset
                end = start + self.samples_per_dataset
                indices = self.indices_per_dataset[d][start:end]
                batch_indices.extend(indices)

            if self.shuffle:
                random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


def get_datasets_AGE(cfg):
    tr_std = None
    te_std = None

    cfg.is_filelist = dict()
    cfg.d_nums = dict()
    tr_imgs, tr_ages = [], []
    te_imgs, te_ages = [], []
    tr_orig_ages = []
    tr_dataset_names, te_dataset_names = [], []

    if 'morph' in cfg.dataset:
        morph_img_root = cfg.morph_img_root
        morph_tr_list = pd.read_csv(cfg.morph_train_file, sep=cfg.morph_delimeter)
        morph_tr_list = np.array(morph_tr_list)
        morph_tr_imgs = [f'{morph_img_root}/{i_path}' for i_path in morph_tr_list[:, cfg.morph_img_idx]]
        morph_tr_ages = morph_tr_list[:, cfg.morph_lb_idx]

        morph_te_list = pd.read_csv(cfg.morph_test_file, sep=cfg.morph_delimeter)
        morph_te_list = np.array(morph_te_list)
        morph_te_imgs = [f'{morph_img_root}/{i_path}' for i_path in morph_te_list[:, cfg.morph_img_idx]]
        morph_te_ages = morph_te_list[:, cfg.morph_lb_idx]

        tr_imgs += morph_tr_imgs
        te_imgs += morph_te_imgs

        tr_ages += morph_tr_ages.tolist()
        tr_orig_ages += morph_tr_ages.tolist()
        te_ages += morph_te_ages.tolist()
    
        tr_dataset_names += ['morph'] * len(morph_tr_imgs)
        te_dataset_names += ['morph'] * len(morph_te_imgs)
        cfg.d_nums['morph'] = len(morph_tr_imgs)
        cfg.is_filelist['morph'] = cfg.morph_is_filelist

    if 'clap' in cfg.dataset:
        clap_img_root = cfg.clap_img_root
        clap_tr_list = pd.read_csv(cfg.clap_train_file, sep=cfg.clap_delimeter)
        clap_tr_list = np.array(clap_tr_list)
        clap_tr_ages = clap_tr_list[:, cfg.clap_lb_idx]
        clap_tr_imgs = [f'{clap_img_root}/{clap_tr_list[i, 3]}/{clap_tr_list[i, cfg.clap_img_idx]}' for i in range(len(clap_tr_list))]
        clap_tr_std = clap_tr_list[:, 2]
    

        clap_te_list = pd.read_csv(cfg.clap_test_file, sep=cfg.clap_delimeter)
        clap_te_list = np.array(clap_te_list)
        clap_te_imgs = [f'{clap_img_root}/{clap_te_list[i, 3]}/{clap_te_list[i, cfg.clap_img_idx]}' for i in range(len(clap_te_list))]
        clap_te_ages = clap_te_list[:, cfg.clap_lb_idx]
        clap_te_std = clap_te_list[:, 2]

        tr_imgs += clap_tr_imgs
        te_imgs += clap_te_imgs

        tr_ages += clap_tr_ages.tolist()
        tr_orig_ages += clap_tr_ages.tolist()
        te_ages += clap_te_ages.tolist()

        tr_dataset_names += ['clap'] * len(clap_tr_imgs)
        te_dataset_names +=  ['clap'] * len(clap_te_imgs)

        cfg.d_nums['clap'] = len(clap_tr_imgs)
        cfg.is_filelist['clap'] = cfg.clap_is_filelist

    if 'adience' in cfg.dataset:
        label_map = np.array([1, 5, 10.5, 17.5, 34, 40.5, 50.5, 80.5])

        with open(cfg.adience_train_file, 'rb') as f:
            adience_data = pickle.load(f)
            adience_tr_imgs = adience_data['img_path']
            adience_tr_orig_ages = adience_data['age']
            adience_tr_ages = label_map[adience_data['age'].astype('int')].astype('long')

        with open(cfg.adience_test_file, 'rb') as f:
            adience_data = pickle.load(f)
            adience_te_imgs = adience_data['img_path']
            # adience_te_imgs = adience_data['img_path'].tolist()
            adience_te_ages = adience_data['age']

        tr_imgs += adience_tr_imgs
        te_imgs += adience_te_imgs

        tr_ages += adience_tr_ages.tolist()
        tr_orig_ages += adience_tr_orig_ages.tolist()
        te_ages += adience_te_ages.tolist()

        tr_dataset_names += ['adience'] * len(adience_tr_ages)
        te_dataset_names += ['adience'] * len(adience_te_ages)

        cfg.d_nums['adience'] = len(adience_tr_ages)
        cfg.is_filelist['adience'] = cfg.adience_is_filelist

    if 'ageDB' in cfg.dataset:
        ageDB_img_root = cfg.ageDB_img_root
        ageDB_split = pd.read_csv(cfg.ageDB_data_file)['split']

        ageDB_tr_list = pd.read_csv(cfg.ageDB_data_file)[ageDB_split == 'train']
        ageDB_tr_ages = ageDB_tr_list['age'].to_numpy()
        ageDB_tr_imgs = [os.path.join(ageDB_img_root, ageDB_tr_list['path'].to_numpy()[i]) for i in range(len(ageDB_tr_list))]

        ageDB_te_list = pd.read_csv(cfg.ageDB_data_file)[ageDB_split == 'test']
        ageDB_te_ages = ageDB_te_list['age'].to_numpy()
        ageDB_te_imgs = [os.path.join(ageDB_img_root, ageDB_te_list['path'].to_numpy()[i]) for i in range(len(ageDB_te_list))]

        tr_imgs += ageDB_tr_imgs
        te_imgs += ageDB_te_imgs

        tr_ages += ageDB_tr_ages.tolist()
        tr_orig_ages += ageDB_tr_ages.tolist()
        te_ages += ageDB_te_ages.tolist()

        tr_dataset_names += ['ageDB'] * len(ageDB_tr_imgs)
        te_dataset_names += ['ageDB'] * len(ageDB_te_imgs)

        cfg.d_nums['ageDB'] = len(ageDB_tr_imgs)
        cfg.is_filelist['ageDB'] = cfg.ageDB_is_filelist

    if 'utk' in cfg.dataset:
        utk_img_root = cfg.utk_img_root
        utk_tr_list = pd.read_csv(cfg.utk_train_file)
        utk_tr_ages = utk_tr_list['age'].to_numpy()
        utk_tr_imgs = [os.path.join(utk_img_root, utk_tr_list['filename'].to_numpy()[i]) for i in range(len(utk_tr_list))]

        utk_te_list = pd.read_csv(cfg.utk_test_file)
        utk_te_ages = utk_te_list['age'].to_numpy()
        utk_te_imgs = [os.path.join(utk_img_root, utk_te_list['filename'].to_numpy()[i]) for i in range(len(utk_te_list))]

        tr_imgs += utk_tr_imgs
        te_imgs += utk_te_imgs

        tr_ages += utk_tr_ages.tolist()
        tr_orig_ages += utk_tr_ages.tolist()
        te_ages += utk_te_ages.tolist()

        tr_dataset_names += ['utk'] * len(utk_tr_imgs)
        te_dataset_names += ['utk'] * len(utk_te_imgs)

        cfg.d_nums['utk'] = len(utk_tr_imgs)
        cfg.is_filelist['utk'] = cfg.utk_is_filelist

    if 'cacd' in cfg.dataset:
        cacd_img_root = cfg.cacd_img_root
        cacd_split = pd.read_csv(cfg.cacd_data_file)['fold']

        cacd_tr_list = pd.read_csv(cfg.cacd_data_file)[cacd_split == 'train']
        cacd_tr_ages = cacd_tr_list['age'].to_numpy()
        cacd_tr_imgs = [os.path.join(cacd_img_root, cacd_tr_list['filename'].to_numpy()[i]) for i in
                         range(len(cacd_tr_list))]

        cacd_te_list = pd.read_csv(cfg.cacd_data_file)[cacd_split == 'test']
        cacd_te_ages = cacd_te_list['age'].to_numpy()
        cacd_te_imgs = [os.path.join(cacd_img_root, cacd_te_list['filename'].to_numpy()[i]) for i in
                         range(len(cacd_te_list))]

        tr_imgs += cacd_tr_imgs
        te_imgs += cacd_te_imgs

        tr_ages += cacd_tr_ages.tolist()
        tr_orig_ages += cacd_tr_ages.tolist()
        te_ages += cacd_te_ages.tolist()

        tr_dataset_names += ['cacd'] * len(cacd_tr_imgs)
        te_dataset_names += ['cacd'] * len(cacd_te_imgs)

        cfg.d_nums['cacd'] = len(cacd_tr_imgs)
        cfg.is_filelist['cacd'] = cfg.cacd_is_filelist

    tr_imgs, tr_ages, tr_dataset_names = np.asarray(tr_imgs), np.asarray(tr_ages), np.asarray(tr_dataset_names)
    te_imgs, te_ages, te_dataset_names = np.asarray(te_imgs), np.asarray(te_ages), np.asarray(te_dataset_names)
    tr_orig_ages = np.asarray(tr_orig_ages)
    cfg.d_nums['total'] = len(tr_ages)
    #########################################################################################
    if cfg.augmentation:
        trainset = basic.Basic(tr_imgs, tr_orig_ages, tr_dataset_names, TwoCropTransform(cfg.transform_tr), is_filelist=cfg.is_filelist, return_ranks=True, norm_age=cfg.norm_age)
    else:
        trainset = basic.Basic(tr_imgs, tr_orig_ages, tr_dataset_names, cfg.transform_tr, is_filelist=cfg.is_filelist, return_ranks=True, norm_age=cfg.norm_age)
    
    cb_sampler = BalancedDatasetBatchSampler(
        dataset=trainset,
        batch_size=cfg.batch_size,
        dataset_names=tr_dataset_names,
        shuffle=True
    )

    ############################################################################
    loader_dict = dict()
    loader_dict['train'] = DataLoader(trainset, batch_sampler=cb_sampler, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_orig_ages, tr_dataset_names, cfg.transform_te, is_filelist=cfg.is_filelist, norm_age=False),
                                    batch_size=cfg.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=cfg.num_workers)

    # loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, te_dataset_names, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False, morph_races=morph_te_race, morph_genders=morph_te_gender, utk_races=utk_te_race, utk_genders=utk_te_gender),
    #                                  batch_size=cfg.test_batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, te_dataset_names, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False),
                                     batch_size=cfg.test_batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict





