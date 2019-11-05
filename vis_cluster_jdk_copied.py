import os
import argparse

import torch

from utils.paths import results_path, benchmarks_path
from utils.misc import load_module
import matplotlib.pyplot as plt

from params import B, N, K, rand_N, rand_K

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--rand_N', action='store_true')
parser.add_argument('--rand_K', action='store_true')
parser.add_argument('--mode', type=str, default='cluster',
        choices=['cluster', 'step'])
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--save', action='store_true')
parser.add_argument('--filename', type=str, default=None)
# parser.add_argument('--K', type=int, default=4)

args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))

if not hasattr(model, 'cluster'):
    raise ValueError('Model is not for clustering')

save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

net.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'originalDAC_fullytrained.tar')))

# print("args.rand_N:", args.rand_N)
# print("args.rand_K:", args.rand_K)
# print("args.vN:", args.vN)
# print("args.vK:", args.vK)

K=4
print("B:", B)
print("N:", N)
print("K:", K)
print("rand_N:", rand_N)
print("rand_K:", rand_K)

# batch = model.sample(B, N, K,
#         rand_N=rand_N, rand_K=rand_K)

# batch = model.sample(args.vB, args.vN, args.vK,
#         rand_N=args.rand_N, rand_K=args.rand_K)

# batch = model.sample_mog_FP(B=args.vB, N=-1, K=8, sample_K=False, det_per_cluster=50, dim=2,
#  onehot=True, add_false_positives=False, FP_count=64, meas_std=.1)


# batch = model.sample_mog_FP(B=10, N=-1, K=16, sample_K=False, det_per_cluster=50, dim=2,
#  onehot=True, add_false_positives=True, FP_count=64, meas_std=.1)

# batch = model.sample_mog_FP(B=10, N=-1, K=16, sample_K=False, det_per_cluster=50,
#          dim=2, onehot=True, add_false_positives=True, FP_count=64, meas_std=.1)

batch = model.sample(B, N, K,
            alpha=1.0, onehot=True,
            rand_N=False, rand_K=False,
            add_false_positives=True,
            FP_count=64)

X = batch['X']

if args.mode == 'cluster':
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
        fp_removal_net = fp_removal_model.net.cuda()
        fp_removal_dir = os.path.join(results_path, module_name, fp_run_name)
        fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model.tar')))
        # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'const_numFP_partialtraining.tar')))      
        fp_removal_net.eval()
        params_, ll_, logits = fp_removal_net(batch['X'].cuda())
        mask = (logits > 0.0)
        params, labels, ll = model.cluster(X.cuda(), max_iter=args.max_iter, mask=mask)
    else:
        params, labels, ll = model.cluster(X.cuda(), max_iter=args.max_iter)
    print('plotting...')
    model.plot_clustering(X, params, labels, FP_included=(FP_removal_model is not None))
    print('log likelihood: {}'.format(ll.item()))
elif args.mode == 'step':
    model.plot_step(X.cuda())

if args.save:
    if not os.path.isdir('./jdk_figures'):
        os.makedirs('./jdk_figures')
    filename = os.path.join('./jdk_figures', '{}_{}.pdf'.format(module_name, args.mode)) \
            if args.filename is None else args.filename
    plt.savefig(filename, bbox_inches='tight')
plt.show()
