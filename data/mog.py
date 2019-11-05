import torch
import torch.nn.functional as F
from data.cluster import sample_labels
from data.mvn import MultivariateNormalDiag
import math

def sample_mog(B, N, K,
        mvn=None, return_ll=False,
        alpha=1.0, onehot=True,
        rand_N=True, rand_K=True,
        device='cpu', add_false_positives=True,
        FP_count=64):

    mvn = MultivariateNormalDiag(2) if mvn is None else mvn
    N = torch.randint(int(0.3*N), N, [1], dtype=torch.long).item() \
            if rand_N else N
    labels = sample_labels(B, N, K, alpha=alpha, rand_K=rand_K, device=device)
    params = mvn.sample_params([B, K], device=device)
#     params = mvn.sample_params_jdk([B, K], device=device)
    gathered_params = torch.gather(params, 1,
            labels.unsqueeze(-1).repeat(1, 1, params.shape[-1]))
    X = mvn.sample(gathered_params)
    
    if add_false_positives:
#         print("labels:", labels)
#         print("labels.shape:", labels.shape)
#         print("X.shape", X.shape)
        B = X.shape[0]
        dim = X.shape[2]
#         false_positives = -4 + 8*torch.rand(B, FP_count, dim).to(device)
        false_positives = 3.0 * torch.randn(B, FP_count, dim).to(device)
        X = torch.cat([X, false_positives], dim=1)
        labels_for_FP =  K * torch.ones((X.shape[0], FP_count), dtype=torch.long).to(device)
        labels = torch.cat([labels, labels_for_FP], dim=1)

        FP_labels = torch.cat([torch.zeros((X.shape[0], N), dtype=torch.long),
                            torch.ones((X.shape[0], FP_count), dtype=torch.long)], dim=1).to(device)
#         print("labels:", labels)
#         print("labels.shape:", labels.shape)
    

    if onehot:
        if add_false_positives:
            # print("labels:", labels)
            labels = F.one_hot(labels, K+1)
        else:        
            labels = F.one_hot(labels, K) 
            
    gt_objects = params[:, :, :2]
    dataset = {'X':X, 'labels':labels, "gt_objects":gt_objects, 'FP_labels': FP_labels}
    if return_ll:
        if not onehot:
            labels = F.one_hot(labels, K)
        # recover pi from labels
        if add_false_positives:
#             print("X[:, :-FP_count, :].shape:", X[:, :-FP_count, :].shape)
#             print("X.shape:", X.shape)
#             print("labels.shape:", labels.shape)
#             print("labels[:, :-FP_count, :-1].shape:", labels[:, :-FP_count, :-1].shape)
#             sleep(lasfjdkls)
            pi = labels[:, :-FP_count, :-1].float().sum(1, keepdim=True) / N
            ll = mvn.log_prob(X[:, :-FP_count, :], params) + (pi+1e-10).log()
            
        else:
#             print("X.shape:", X.shape)
#             print("labels.shape:", labels.shape)
#             sleep(temp)
            pi = labels.float().sum(1, keepdim=True) / N
            ll = mvn.log_prob(X, params) + (pi+1e-10).log()
        dataset['ll'] = ll.logsumexp(-1).mean().item()
    return dataset



def sample_mog_varNumFP(B, N, K,
        mvn=None, return_ll=False,
        alpha=1.0, onehot=True,
        rand_N=True, rand_K=True,
        device='cpu', FP_count=64):

    mvn = MultivariateNormalDiag(2) if mvn is None else mvn
    N = torch.randint(int(0.3*N), N, [1], dtype=torch.long).item() \
            if rand_N else N
    labels = sample_labels(B, N, K, alpha=alpha, rand_K=rand_K, device=device)
    params = mvn.sample_params([B, K], device=device)
    assert(len(params.shape) == 3)
    assert((B, K, 4) == params.shape), (B, K, 4, params.shape)
    params[:,0,:2] = 0.0 #set FP mu's to 0
    params[:,0,2:] = 3.0 #set FP std's to 3
    
#     params = mvn.sample_params_jdk([B, K], device=device)
    gathered_params = torch.gather(params, 1,
            labels.unsqueeze(-1).repeat(1, 1, params.shape[-1]))
    X = mvn.sample(gathered_params)
    
    assert(B == X.shape[0])
    dim = X.shape[2]
#         false_positives = -4 + 8*torch.rand(B, FP_count, dim).to(device)

    FP_labels = torch.zeros_like(labels).to(device)
    FP_labels[torch.where(labels == 0)] = 1.0

    
    if onehot:
          labels = F.one_hot(labels, K+1)

    gt_objects = params[:, :, :2]
    dataset = {'X':X, 'labels':labels, "gt_objects":gt_objects, 'FP_labels': FP_labels}
    if return_ll:
        if not onehot:
            labels = F.one_hot(labels, K)
        # recover pi from labels
        pi = labels[:, :-FP_count, :-1].float().sum(1, keepdim=True) / N
        ll = mvn.log_prob(X[:, :-FP_count, :], params) + (pi+1e-10).log()

        dataset['ll'] = ll.logsumexp(-1).mean().item()
    return dataset

def sample_mog_FP(B, N, K, sample_K=False, det_per_cluster=50, dim=2, onehot=True,
 add_false_positives=False, FP_count=64, meas_std=.1):
    # sleep(temps)
    device = 'cpu' if not torch.cuda.is_available() \
            else torch.cuda.current_device()

    if sample_K:
        K = np.random.randint(1,K+1)

    pi = torch.ones(B,K).to(device) #fix pi to have even size clusters
  
    # assert(N==K*det_per_cluster)
    N = K*det_per_cluster
    labels = torch.tensor([i//det_per_cluster for i in range(K*det_per_cluster)]).to(device)
    # labels = []
    # labels = torch.tensor([i//det_per_cluster for i in range(N)]).to(device)
    # for b_idx in range(B):
    #     num_clusters = np.random.randint(1,K+1)
    #     labels = torch.tensor([i//det_per_cluster for i in range(num_clusters*det_per_cluster)]).to(device)
    #     print("labels:", labels)
    # sleep(temps)
    labels = labels.repeat(B,1)

    gt_objects = -4 + 8*torch.rand(B, K, dim).to(device)
    # gt_objects = -6 + 12*torch.rand(B, K, dim).to(device)
    sigma = meas_std*torch.ones(B, K, dim).to(device)
    eps = torch.randn(B, N, dim).to(device)

    rlabels = labels.unsqueeze(-1).repeat(1, 1, dim)
    X = torch.gather(gt_objects, 1, rlabels) + \
            eps * torch.gather(sigma, 1, rlabels)

    if add_false_positives:
        false_positives = -4 + 8*torch.rand(B, FP_count, dim).to(device)
        X = torch.cat([X, false_positives], dim=1)
        labels = torch.tensor([i//det_per_cluster for i in range(K*det_per_cluster)] + [K for i in range(FP_count)]).to(device)
        labels = labels.repeat(B,1)

    if onehot:
        if add_false_positives:
            # print("labels:", labels)
            labels = F.one_hot(labels, K+1)
        else:        
            labels = F.one_hot(labels, K)        

 #   print("labels:", labels)
 #   print("labels.shape:", labels.shape)
 #   sleep(labelcheck)

    dataset = {"X":X, "labels":labels, "gt_objects":gt_objects, "ll": -1}
    return dataset

def sample_warped_mog(B, N, K,
        radial_std=0.4, tangential_std=0.1,
        alpha=5.0, onehot=True,
        rand_N=True, rand_K=True, device='cpu'):

    dataset = sample_mog(B, N, K,
            mvn=MultivariateNormalDiag(1),
            alpha=alpha, onehot=False,
            rand_N=rand_N, rand_K=rand_K,
            device=device)

    r, labels = dataset['X'], dataset['labels']
    N = r.shape[1]
    r = 2*math.pi*radial_std*r
    a = torch.gather(2*torch.randn(B, K).to(device), 1, labels).unsqueeze(-1)
    b = torch.gather(2*torch.randn(B, K).to(device), 1, labels).unsqueeze(-1)
    cos = r.cos()
    sin = r.sin()
    x = a*cos
    y = b*sin
    dx = b*cos
    dy = a*sin
    norm = (dx.pow(2) + dy.pow(2)).sqrt()
    t = tangential_std*torch.randn(B, N, 1).to(device)
    dx = t*dx/norm
    dy = t*dy/norm
    x = x + dx
    y = y + dy
    E = torch.cat([x, y], -1)
    rho = torch.gather(2*math.pi*torch.rand(B, K).to(device), 1, labels)
    rot = torch.stack([rho.cos(), -rho.sin(), rho.sin(), rho.cos()], -1)
    rot = rot.reshape(B, -1, 2, 2)
    X = torch.einsum('bni,bnij->bnj', E, rot)

    mu = torch.gather(min(K, 4.0)*torch.randn(B, K, 2).to(device),
            1, labels.unsqueeze(-1).repeat(1, 1, 2))
    X = X + mu
    if onehot:
        labels = F.one_hot(labels, K)

    dataset['X'] = X
    dataset['labels'] = labels
    return dataset

if __name__ == '__main__':

    ds = sample_mog(10, 300, 4, return_ll=True, rand_K=True)
    print(ds['ll'])
