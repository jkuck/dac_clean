import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import benchmarks_path, results_path
from utils.misc import add_args
from utils.plots import scatter, draw_ellipse, scatter_mog

from data.mvn import MultivariateNormalDiag
from data.mog import sample_mog, sample_mog_FP

from neural.attention import StackedISAB, PMA, MAB

from models.base import ModelTemplate

parser = argparse.ArgumentParser()

# for training
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

parser.add_argument('--run_name', type=str, default='trial')

# for visualization
parser.add_argument('--vB', type=int, default=10)
parser.add_argument('--vN', type=int, default=1000)
parser.add_argument('--vK', type=int, default=4)

sub_args, _ = parser.parse_known_args()

save_dir = os.path.join(results_path, 'mog', sub_args.run_name)

class FindCluster(nn.Module):
    def __init__(self, mvn, dim_hids=128, num_inds=32):
        super().__init__()
        self.mvn = mvn

        self.isab1 = StackedISAB(mvn.dim, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, mvn.dim_params)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None, predict_from_clustering=False):
        H_enc = self.isab1(X, mask=mask)
        Z = self.pma(H_enc, mask=mask)
        params = self.mvn.transform(self.fc1(Z))
        ll = self.mvn.log_prob(X, params)

        H_dec = self.mab(H_enc, Z)
        logits = self.fc2(self.isab2(H_dec, mask=mask))

        if predict_from_clustering:
            # print("params.shape:", params.shape)
            # sleep(temps)
            logit_threshold = 0
            if mask is not None:
                # print("torch.sum(mask[0]):", torch.sum(mask[0]))
                clustered_TP_indices = torch.where((logits.repeat(1,1,2)>logit_threshold) * #logical and
                                                   (mask == 0))
            else:
                clustered_TP_indices = torch.where(logits.repeat(1,1,2)>logit_threshold)
            clusters = X[clustered_TP_indices]

            for b_idx in range(params.shape[0]):
                if torch.where(clustered_TP_indices[0] == b_idx)[0].size()[0] > 0:
                    # print("params[b_idx,0,:2].shape:", params[b_idx,0,:2].shape)
                    # print("torch.mean(clusters[torch.where(clustered_TP_indices[0] == b_idx)[0]].reshape(-1,2), dim=0).shape:", torch.mean(clusters[torch.where(clustered_TP_indices[0] == b_idx)[0]].reshape(-1,2), dim=0).shape)
                    params[b_idx,0,:2] = torch.mean(clusters[torch.where(clustered_TP_indices[0] == b_idx)[0]].reshape(-1,2), dim=0)

        return params, ll, logits

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)          
        self.testfile = os.path.join(save_dir,
                'mog_10_1000_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(save_dir,
                # 'mog_10_3000_12.tar' if self.clusterfile is None else self.clusterfile)
                'mog_10_100_16.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster(MultivariateNormalDiag(2))

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, self.N, 4,
                    rand_N=True, rand_K=True, return_ll=True))
            torch.save(bench, self.testfile)
        if not os.path.isfile(self.clusterfile) or force:
            print('generating benchmark {}...'.format(self.clusterfile))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, self.N*4, 16,
#                 bench.append(sample_mog(10, 600, 12,
                    rand_N=True, rand_K=True, return_ll=True))
 #                bench.append(sample_mog_FP(B=10, N=-1, K=12, sample_K=False, det_per_cluster=4, dim=2,
 # onehot=True, add_false_positives=False, FP_count=64, meas_std=.1))
            torch.save(bench, self.clusterfile)

    def sample(self, B, N, K, **kwargs):
#         print("kwargs:", kwargs)
#         sleep(temp)
#         return sample_mog(B, N, K, device=torch.device('cuda'), **kwargs)
        return sample_mog_varNumFP(B, N, K, device=torch.device('cuda'), **kwargs)


    def sample_mog_FP(self, B, N, K, **kwargs):
        return sample_mog_FP(B, N, K, **kwargs)


    def plot_clustering(self, X, params, labels):
        B = X.shape[0]
        mu, cov = self.net.mvn.stats(torch.cat(params, 1))
        if B == 1:
            scatter_mog(X[0], labels[0], mu[0], cov[0])
        else:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                scatter_mog(X[b], labels[b], mu[b], cov[b], ax=ax)

    def plot_step(self, X):
        B = X.shape[0]
        self.net.eval()
        params, _, logits = self.net(X)
        mu, cov = self.net.mvn.stats(params)
        labels = (logits > 0.0).int().squeeze(-1)
        if B == 1:
            scatter(X[0], labels=labels[0])
            draw_ellipse(mu[0][0], cov[0][0])
        else:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                scatter(X[b], labels=labels[b], ax=ax)
                draw_ellipse(mu[b][0], cov[b][0], ax=ax)

def load(args):
    add_args(args, sub_args)
    return Model(args)
