import torch
import torch.nn as nn
import numpy as np

from utils.util import to_np


def heteroscedastic_nll(mu, sigma, target, cfg):
    # mu: (B,), logvar: (B,), target: (B,)
    # use softplus for stability to get sigma^2
    sigma2 = sigma**2
    loss = 0.5 * ( (target - mu) ** 2 / (sigma2+cfg.epsilon) + cfg.lambda_sigma * torch.log(sigma2+cfg.epsilon))
    return loss.mean()


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l2':
            return torch.square(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l3':
            return torch.pow(torch.abs(labels[:, None, :] - labels[None, :, :]), 3).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='L2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'L2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cosine':
            # return (features)
            return
        else:
            raise ValueError(self.similarity_type)

class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='L2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels, cfg):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        if cfg.augmentation:
            labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0] # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss



def compute_center_loss(embs, rank_labels, fdc_points, cfg, record=False):
    if cfg.start_norm:
        fdc_points = nn.functional.normalize(fdc_points, dim=-1)
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (cfg.fiducial_point_num - 1)) * i for i in range(cfg.fiducial_point_num)])

    def get_pos_neg_idxs(ranks, fdc_ranks, cfg):
        adaptive_margin = cfg.n_ranks != cfg.fiducial_point_num
        if adaptive_margin:
            nn_idxs = []
            margins = []
            emb_idxs = []
            emb_idx = 0
            for r in ranks:
                abs_diff = np.abs(fdc_ranks - r)
                min_val = abs_diff.min()
                nn = np.argwhere(abs_diff == min_val).flatten()
                nn_idxs.append(nn)

                margin_val = min_val * cfg.margin / (max(cfg.tau, 1))
                margins.append([margin_val] * len(nn))
                emb_idxs.append([emb_idx] * len(nn))
                emb_idx += 1
            nn_idxs = np.concatenate(nn_idxs)
            margins = np.concatenate(margins)
            emb_idxs = np.concatenate(emb_idxs)
        else:
            nn_idxs = ranks
            margins = np.array([0.5 * cfg.margin / (max(cfg.tau, 1))] * len(nn_idxs))
            emb_idxs = np.arange(len(nn_idxs))

        return nn_idxs, emb_idxs, margins

    nn_idxs, emb_idxs, margins = get_pos_neg_idxs(rank_labels, fdc_point_ranks, cfg)

    if cfg.metric == 'L2':
        dists = torch.cdist(fdc_points, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(fdc_points, embs.transpose(1, 0))

    loss = dists[nn_idxs, emb_idxs]

    # loss = nn.functional.relu(violation)
    # loss = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
    if record:
        return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7), to_np(loss)
    # return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7)
    return torch.sum(loss) / embs.shape[0]

