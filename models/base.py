import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.tensor import to_numpy
import numpy as np

def compute_filter_loss(ll, logits, labels, lamb=1.0):
    B, K = labels.shape[0], labels.shape[-1]
    bcent = F.binary_cross_entropy_with_logits(
            logits.repeat(1, 1, K),
            labels, reduction='none').mean(1)
    ll = (ll * labels).sum(1) / (labels.sum(1) + 1e-8)
    loss = lamb * bcent - ll
    loss[ll==0] = float('inf')
    loss, idx = loss.min(1)
    bidx = loss != float('inf')

    loss = loss[bidx].mean()
    ll = ll[bidx, idx[bidx]].mean()
    bcent = bcent[bidx, idx[bidx]].mean()
    return loss, ll, bcent

def compute_cluster_loss(ll, logits, labels):
    B, K = labels.shape[0], labels.shape[-1]
    bcent = F.binary_cross_entropy_with_logits(
            logits.repeat(1, 1, K),
            labels, reduction='none').mean(1)
    loss = bcent
    loss, idx = loss.min(1)
    bidx = loss != float('inf')

    loss = loss[bidx].mean()
    ll = -1
    bcent = bcent[bidx, idx[bidx]].mean()
    return loss, ll, bcent

def compute_FP_removal_loss(FP_logits, FP_labels, det_mask, weight=.2):
    #BCE on classifying TP/FP
    B, K = FP_labels.shape[0], FP_labels.shape[-1]
    FP_logits = FP_logits.squeeze()
    # print("FP_logits.shape:", FP_logits.shape)
    # print("FP_labels.shape:", FP_labels.shape)
    # print("FP_logits:", FP_logits)
    # print("FP_labels:", FP_labels)
    # print()
    bcent_loss = F.binary_cross_entropy_with_logits(
            FP_logits, FP_labels, reduction='none',
            pos_weight=torch.tensor(weight))
    # print("bcent_loss:", bcent_loss)
    bcent_loss = (bcent_loss * det_mask.bitwise_not().float())
    # print('-'*80)
    # print("masked bcent_loss:", bcent_loss)
    # print("torch.mean(bcent_loss):", torch.mean(bcent_loss))
    # print("det_mask:", det_mask)
    # print("torch.sum(det_mask.bitwise_not()):", torch.sum(det_mask.bitwise_not()))
    # print('mean unmasked bce:', torch.sum(bcent_loss)/torch.sum(det_mask.bitwise_not()))
    # sleep(temp)
    bcent_loss = torch.sum(bcent_loss)/torch.sum(det_mask.bitwise_not())

    return bcent_loss, bcent_loss

def compute_filter_loss_distance(logits, labels, pred_cluster, gt_objects,\
                                 gt_count_per_img, det_mask, weight=None, lamb=1.0, verbose=False):
    '''
    directly regress ground truth object positions using a distance loss
    instead of log likelihood
    '''
    verboseA = False
    if verbose:
        print("labels.shape:", labels.shape)
        print("labels:", labels)
        print("logits.shape:", logits.shape)
        print("logits", logits)
    B, K = labels.shape[0], labels.shape[-1]

    if verboseA:
        print("logits.shape:", logits.shape)
        print("labels.shape:", labels.shape)

    if weight is None:
        bcent = F.binary_cross_entropy_with_logits(
                logits.repeat(1, 1, K),
                labels, reduction='none')#.mean(1)
    else:
        bcent = F.binary_cross_entropy_with_logits(
                logits.repeat(1, 1, K),
                labels, reduction='none', pos_weight=torch.tensor(weight))#.mean(1)

    if verboseA:
        print("a bcent.shape:", bcent.shape)   

    det_mask = det_mask.unsqueeze(dim=2).repeat(1,1,bcent.shape[2])
    not_masked = det_mask.bitwise_not().float()
    bcent = (bcent * not_masked)
    bcent = torch.sum(bcent, dim=1)/torch.sum(not_masked, dim=1)

    if verboseA:
        print("b bcent.shape:", bcent.shape)
        print("gt_count_per_img.shape:", gt_count_per_img.shape)
        print("gt_count_per_img:", gt_count_per_img)
        print("det_mask.shape:", det_mask.shape)
    # sleep(asldfkj)
    if verbose:
        print("bcent.shape:", bcent.shape)
    assert(len(pred_cluster.shape) == 3), (pred_cluster.shape)
    batch_size, dummy_one, dim = pred_cluster.shape
    assert(batch_size == B)
    assert(dummy_one == 1)
    assert(len(gt_objects.shape) == 3), (gt_objects.shape, pred_cluster.shape)
    batch_size1, gt_obj_count, dim1 = gt_objects.shape
    assert(batch_size1 == B)
    assert(dim == dim1)
    distances = torch.cdist(gt_objects.contiguous(), pred_cluster.contiguous())
    distances = distances.squeeze(dim=2)

    assert((batch_size, gt_obj_count) == distances.shape), (batch_size, gt_obj_count, distances.shape)

    indices = torch.tensor([[c for c in range(bcent.shape[1])] for r in range(bcent.shape[0])], device=gt_count_per_img.device)
    repeated_gt_count_per_img = gt_count_per_img.unsqueeze(dim=1).repeat(1,bcent.shape[1])
    assert(indices.shape == repeated_gt_count_per_img.shape), (indices.shape, repeated_gt_count_per_img.shape)
    assert(bcent.shape == repeated_gt_count_per_img.shape), (bcent.shape, repeated_gt_count_per_img.shape)
    if verboseA:
        print(repeated_gt_count_per_img.device)
    bcent[torch.where(indices >= repeated_gt_count_per_img)] = np.inf

    loss = lamb * bcent + distances
    

    if verbose:
        print("loss.shape:", loss.shape)
    loss, idx = loss.min(1)
    # print("loss:", loss[0])
    if verbose:
        print("loss idx.shape:", idx.shape)
    bidx = loss != float('inf')
    # print("loss[bidx]:", loss[bidx])

    loss = loss[bidx].mean()
    if verbose:
        print("post mean loss.shape:", loss.shape)
    bcent = bcent[bidx, idx[bidx]].mean()
    if verbose:
        print("post mean bcent.shape:", bcent.shape)
    # sleep(checklabelshape)
    return loss, bcent

class ModelTemplate(object):
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.net = None
        self.metrics = ['ll', 'bcent']

    def load_from_ckpt(self):
        pass

    def sample(self, B, N, K):
        raise NotImplementedError

    def get_train_loader(self):
#         print("self.N:", self.N)
#         sleep(temp)
        for _ in range(self.num_steps):
            yield self.sample(self.B, self.N, self.K)

    def gen_benchmarks(self, force=False):
        pass

    def get_test_loader(self, filename=None):
        filename = self.testfile if filename is None else filename
        return torch.load(filename)

    def build_optimizer(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.num_steps)
        return optimizer, scheduler

    # compute training loss
    def loss_fn(self, batch, train=True, lamb=1.0, mode='FP_removal'):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda().float()
        det_mask = batch['det_mask'].cuda()#.float()
        pred_bbox, logits = self.net(X, mask=det_mask)
        gt_mask = batch['gt_mask'].cuda()
        # print()
        # print("labels.shape:", labels.shape)
        # print("X.shape:", X.shape)
#         loss, ll, bcent = compute_filter_loss(ll, logits, labels, lamb=lamb)
#         loss, ll, bcent = compute_cluster_loss(ll, logits, labels)
    
        if mode == 'FP_removal':
            FP_labels = batch['FP_labels'].cuda().float()
            loss, bcent = compute_FP_removal_loss(logits, FP_labels, det_mask)
        elif mode == 'clustering':    
            gt_objects = batch['gt_objects'].cuda()
            onehot_labels = batch['onehot_labels'].cuda().float()
            gt_count_per_img = batch['gt_count_per_img'].cuda()#.float()
            loss, bcent = compute_filter_loss_distance(logits, onehot_labels, pred_bbox, gt_objects, gt_count_per_img, det_mask) 
    
        else:
            assert(False), mode
        if train:
            return loss
        else:
            return loss, bcent

    def cluster(self, X, max_iter=50, verbose=True, check=False, mask=None):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            pred_bbox, logits = self.net(X)
            bboxes = [pred_bbox]

            labels = torch.zeros_like(logits).squeeze(-1).int()
            if mask is None:
                mask = (logits > 0.0)
                done = mask.sum((1,2)) == N
            else:
                ind = logits > 0.0
                labels[mask.squeeze(-1)] = max_iter+1
                # labels[(ind*mask.bitwise_not()).squeeze(-1)] = -1
                mask[ind] = True                
                done = mask.sum((1,2)) == N

            for i in range(1, max_iter):
                pred_bbox, logits = self.net(X, mask=mask)

                bboxes.append(pred_bbox)

                ind = logits > 0.0
                labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
                mask[ind] = True

                num_processed = mask.sum((1,2))
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

            fail = done.sum() < B



            if check:
                return bboxes, labels, fail
            else:
                return bboxes, labels,

    def plot_clustering(self, X, params, labels):
        raise NotImplementedError

    def plot_step(self, X):
        raise NotImplementedError
