import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image

class Basic(Dataset):
    def __init__(self, imgs, labels, datasets,transform, norm_age=True, is_filelist=False, return_ranks=False, std=None, morph_genders=None, morph_races=None, clap_genders=None, clap_races=None, ageDB_genders=None, ageDB_races=None, utk_genders=None, utk_races=None, cacd_genders=None, cacd_races=None,adience_genders=None, adience_races=None, genders=None, races=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.labels = labels
        self.datasets = datasets
        #######################################
        self.morph_genders=morph_genders
        self.morph_races=morph_races

        self.clap_genders=clap_genders
        self.clap_races=clap_races

        self.ageDB_genders=ageDB_genders
        self.ageDB_races=ageDB_races

        self.utk_genders=utk_genders
        self.utk_races=utk_races

        self.cacd_genders=cacd_genders
        self.cacd_races=cacd_races

        self.adience_genders=adience_genders
        self.adience_races=adience_races

        self.genders = genders 
        self.races = races
        #######################################

        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if norm_age:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks
        self.std = std

        # rank = 0
        # self.mapping = dict()
        # for cls in np.unique(self.labels):
        #     self.mapping[cls] = rank
        #     rank += 1
        # self.ranks = np.array([self.mapping[l] for l in self.labels])
        self.ranks = self.labels

        d_idx = 0
        self.dataset_mapping = dict()
        for d in np.unique(self.datasets):
            self.dataset_mapping[d] = d_idx
            d_idx += 1
        self.d_idxs = np.array([self.dataset_mapping[k] for k in self.datasets])


    def __getitem__(self, item):
        img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        img = self.transform(img)

        d_idx = self.d_idxs[item]

        if self.return_ranks:
            return img, self.labels[item], self.ranks[item], item, d_idx
        else:
            return img, self.labels[item], item, d_idx

    def __len__(self):
        return len(self.imgs)
