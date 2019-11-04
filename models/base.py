import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.tensor import to_numpy

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

def compute_filter_loss_distance(logits, labels, pred_cluster, gt_objects, weight=None, lamb=1.0, verbose=False):
    '''
    directly regress ground truth object positions using a distance loss
    instead of log likelihood
    '''
    if verbose:
        print("labels.shape:", labels.shape)
        print("labels:", labels)
        print("logits.shape:", logits.shape)
        print("logits", logits)
    B, K = labels.shape[0], labels.shape[-1]

    if weight is None:
        bcent = F.binary_cross_entropy_with_logits(
                logits.repeat(1, 1, K),
                labels, reduction='none').mean(1)
    else:
        bcent = F.binary_cross_entropy_with_logits(
                logits.repeat(1, 1, K),
                labels, reduction='none', pos_weight=torch.tensor(weight)).mean(1)
   
    
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


#     loss = lamb * bcent[:,:-1] + distances #exclude FP class
    loss = lamb * bcent + distances
    # print("loss:", loss[0])
    

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
    def loss_fn(self, batch, train=True, lamb=1.0):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda().float()
        params, ll, logits = self.net(X)
#         loss, ll, bcent = compute_filter_loss(ll, logits, labels, lamb=lamb)
        loss, ll, bcent = compute_cluster_loss(ll, logits, labels)
    
        gt_objects = batch['gt_objects'].cuda()
        pred_cluster = params[:,:,:2]
        compute_filter_loss_distance(logits, labels, pred_cluster, gt_objects) 
        if train:
            return loss
        else:
            return ll, bcent

    def cluster(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            params, ll, logits = self.net(X)
            params = [params]

            labels = torch.zeros_like(logits).squeeze(-1).int()
            mask = (logits > 0.0)
            done = mask.sum((1,2)) == N
            for i in range(1, max_iter):
                params_, ll_, logits = self.net(X, mask=mask)

                ll = torch.cat([ll, ll_], -1)
                params.append(params_)

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

            # ML estimate of mixing proportion pi
            pi = F.one_hot(labels.long(), len(params)).float()
            pi = pi.sum(1, keepdim=True) / pi.shape[1]
            ll = ll + (pi + 1e-10).log()
            ll = ll.logsumexp(-1).mean()

            if check:
                return params, labels, ll, fail
            else:
                return params, labels, ll

    def plot_clustering(self, X, params, labels):
        raise NotImplementedError

    def plot_step(self, X):
        raise NotImplementedError
