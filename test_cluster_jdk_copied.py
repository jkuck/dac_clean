import torch
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import numpy as np

from utils.log import get_logger, Accumulator
from utils.misc import load_module
from utils.paths import results_path, benchmarks_path
from utils.tensor import to_numpy

from data.mog import sample_mog, sample_mog_FP
from params import B, N, K, rand_N, rand_K
from pymatgen.optimization import linear_assignment


parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))
print("args:", args)
FP_removal_model = 'models/mog.p'
if FP_removal_model is not None:
    # fp_run_name = 'fp_removal_network'
    # fp_run_name = 'const_num_fp_removal_network'
    fp_run_name = 'const_num_fp_removal_network_K16'
    run_name = getattr(args, 'run_name')
    setattr(args, 'run_name', fp_run_name)
    # print("args:", args)

    fp_removal_module, fp_removal_module_name = load_module(args.modelfile)
    fp_removal_model = fp_removal_module.load(args)
    setattr(args, 'run_name', run_name)
    # setattr(args, 'run_name', run_name + '_WITH_fp_removal_network')

REGEN_BENCHMARK = True
K=16
if REGEN_BENCHMARK:
    temp_testfile = os.path.join(results_path, 'temp_testfile.tar')
    #regenerate clusterfile for current params
    print('generating benchmark {}...'.format(temp_testfile))
    print("B,N,K:", B,N,K)    
    bench = []
    for _ in range(100):
        # bench.append(sample_mog(10, 3000, 12,

        # bench.append(sample_mog(B, N, K,
        #     rand_N=rand_N, rand_K=rand_K, return_ll=True))

        # print(B,N,K)
        # sleep(sdf)
        bench.append(sample_mog(B, N, K,
            alpha=1.0, onehot=True,
            rand_N=False, rand_K=False,
            add_false_positives=True,
            FP_count=64))

        # bench.append(sample_mog_FP(B=10, N=-1, K=16, sample_K=False, det_per_cluster=50,
        #  dim=2, onehot=True, add_false_positives=True, FP_count=64, meas_std=.1))
    torch.save(bench, temp_testfile)


@torch.jit.script
def my_cdist(x1, x2):
    # https://github.com/pytorch/pytorch/issues/15253
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


if not hasattr(model, 'cluster'):
    raise ValueError('Model is not for clustering')

save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

net.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'originalDAC_fullytrained.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_withFP.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_noFP_probably.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'cluster_loss_withFP.tar')))

net.eval()

if FP_removal_model is not None:
    fp_removal_net = fp_removal_model.net.cuda()
    fp_removal_dir = os.path.join(results_path, module_name, fp_run_name)
    fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model.tar')))

    fp_removal_net.eval()


if REGEN_BENCHMARK:
    test_loader = model.get_test_loader(filename=temp_testfile)
else:
    # test_loader = model.get_test_loader(filename=model.clusterfile)
    test_loader = model.get_test_loader(filename=model.testfile)

accm = Accumulator('model ll', 'oracle ll', 'ARI', 'NMI', 'k-MAE')
num_failure = 0
logger = get_logger('{}_{}'.format(module_name, args.run_name),
        os.path.join(save_dir, args.filename))
all_correct_counts = []
all_distances = []
for batch in tqdm(test_loader):
    if FP_removal_model is not None:
        params_, ll_, logits = fp_removal_net(batch['X'].cuda())
        mask = (logits > 0.0)

    else:
        mask = None

    params, labels, ll, fail = model.cluster(batch['X'].cuda(),
            max_iter=args.max_iter, verbose=False, check=True, mask=mask)
    true_labels = to_numpy(batch['labels'].argmax(-1))
    ari = 0
    nmi = 0
    mae = 0
    for b in range(len(labels)):
        labels_b = to_numpy(labels[b])
        ari += ARI(true_labels[b], labels_b)
        nmi += NMI(true_labels[b], labels_b, average_method='arithmetic')
        mae += abs(len(np.unique(true_labels[b])) - len(np.unique(labels_b)))
    ari /= len(labels)
    nmi /= len(labels)
    mae /= len(labels)

    oracle_ll = 0.0 if batch.get('ll') is None else batch['ll']
    accm.update([ll.item(), oracle_ll, ari, nmi, mae])
    num_failure += int(fail)

    COUNT_CORRECT_matching = False
    if COUNT_CORRECT_matching:
        CORRECT_THRESHOLD = .15
        all_pred_clusters = torch.stack(params, dim=1).squeeze()[:,:,:2]
        # print("len(params):", len(params))
        # print("params[0].shape:", params[0].shape)
        # print("all_pred_clusters.shape:", all_pred_clusters.shape)
        # sleep(asldfhj)
        gt_objects = batch['gt_objects'].cuda()
        batch_size = all_pred_clusters.shape[0]
        for img_idx in range(batch_size):

            pairwise_distance = my_cdist(all_pred_clusters[img_idx],gt_objects[img_idx])
            lin_assign = linear_assignment.LinearAssignment(pairwise_distance.cpu().detach())
            solution = lin_assign.solution
            association_list = zip([i for i in range(len(solution))], solution)
            cur_img_loss = 0.0
            correct_count = 0
            for assoc in association_list:
                cur_assoc_distance = pairwise_distance[assoc[0], assoc[1]]
                all_distances.append(cur_assoc_distance.item())
                if cur_assoc_distance <= CORRECT_THRESHOLD:
                    correct_count += 1
            all_correct_counts.append(correct_count)

    COUNT_CORRECT_greedy = True
    if COUNT_CORRECT_greedy:
        CORRECT_THRESHOLD = .15
        all_pred_clusters = torch.stack(params, dim=1).squeeze()[:,:,:2]
        # print("len(params):", len(params))
        # print("params[0].shape:", params[0].shape)
        # print("all_pred_clusters.shape:", all_pred_clusters.shape)
        # sleep(asldfhj)
        gt_objects = batch['gt_objects'].cuda()
        batch_size = all_pred_clusters.shape[0]
        pairwise_distances = torch.cdist(all_pred_clusters.contiguous(), gt_objects.contiguous())
        (batch_size1, object_count, pred_count) = pairwise_distances.shape
        assert(batch_size1 == batch_size)
        # print("pairwise_distances.shape:", pairwise_distances.shape)
        correct_count = 0
        distance_loss = 0.0
        for i in range(object_count):
            #find min distances per batch
            values, indices = pairwise_distances.view(batch_size, -1).min(dim=1)
            # distance_loss = distance_loss + torch.sum(values)
            distance_loss = distance_loss + torch.sum(values)
            correct_count += torch.sum(values < CORRECT_THRESHOLD).item()
            # print("correct_count:", correct_count)
            row_indices = indices // pred_count
            row_indices = row_indices.unsqueeze(dim=1)#.unsqueeze(dim=1)
            col_indices = indices % pred_count
            col_indices = col_indices.unsqueeze(dim=1)#.unsqueeze(dim=1)
            # print("row_indices:", row_indices)
            # print("col_indices:", col_indices)

            row_idx = [
                    torch.LongTensor(range(batch_size)).unsqueeze(1), 
                    row_indices 
                  ]
            pairwise_distances1 = pairwise_distances.clone() #avoid inplace operation error when taking gradient
            pairwise_distances1[row_idx] = np.inf
            col_idx = [
                    torch.LongTensor(range(batch_size)).unsqueeze(1), 
                    col_indices 
                  ]
            pairwise_distances1 = pairwise_distances1.permute(0, 2, 1)
            pairwise_distances1[col_idx] = np.inf
            pairwise_distances1 = pairwise_distances1.permute(0, 2, 1)
            pairwise_distances = pairwise_distances1
        all_correct_counts.append(correct_count/batch_size)
        all_distances.append(distance_loss.item()/(batch_size*object_count))
logger.info(accm.info())
logger.info('number of failure cases {}'.format(num_failure))
mean_correct_count = np.mean(all_correct_counts)
mean_distance = np.mean(all_distances)
print("mean correct count:", mean_correct_count)
print("mean distance:", mean_distance)