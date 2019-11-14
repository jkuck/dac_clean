import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import argparse
import json

from utils.log import get_logger, Accumulator
from utils.misc import load_module
from utils.paths import results_path, benchmarks_path
from data.bounding_boxes import BoundingBoxDataset, pad_collate

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--clip', type=float, default=10.0)
parser.add_argument('--save_all', action='store_true')
parser.add_argument('--regen_benchmarks', action='store_true')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# MODE = 'FP_removal'
MODE = 'clustering'
MODE1 = MODE

# MODE1 = 'FP_removal_then_cluster'

module, module_name = load_module(args.modelfile)
model = module.load(args)
exp_id = '{}_{}'.format(module_name, args.run_name)
save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

if not os.path.isdir(benchmarks_path):
    os.makedirs(benchmarks_path)
model.gen_benchmarks(force=args.regen_benchmarks)

# train_loader = model.get_train_loader()
# test_loader = model.get_test_loader()

# jdk_data/

# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000_GaussianCRPS_IOUp9_FPremovalNetProcessed.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000_GaussianCRPS_IOUp9.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000_GaussianCRPS_IOUp9_minScoreP2.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000_GaussianCRPS_IOUp5.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000_GausML_IOUp9_minScoreP2.json',\

# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny10.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_allTrain.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GausML_IOUp5_minScoreP2.json',\
# train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GausML_IOUp5_minScoreP2.json',\
train_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny100_GausML_IOUp5_minScoreP2.json',\
                                         num_classes=80, mode=MODE)
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp9_FPremovalNetProcessed.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GausML_IOUp5_minScoreP2.json',\
test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GausML_IOUp5_minScoreP2.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp9_minScoreP2.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp5.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GausML_IOUp9_minScoreP2.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000.json',\
                                         num_classes=80, mode=MODE)
train_loader = DataLoader(train_dataset, batch_size=100,
                        shuffle=True, num_workers=0, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=100,
                        shuffle=True, num_workers=0, collate_fn=pad_collate)

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model.load_from_ckpt()

    if MODE1 == 'FP_removal_then_cluster':
        model.metrics = ["loss", "FP_loss", "FP_bcent", "cluster_loss", "cluster_bcent"]
    print("model.metrics:", model.metrics)
    # save hyperparams
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)

    optimizer, scheduler = model.build_optimizer()
    logger = get_logger(exp_id, os.path.join(save_dir,
        'train_'+time.strftime('%Y%m%d-%H%M')+'.log'))
    accm = Accumulator(*model.metrics)
    train_accm = Accumulator('loss')

    tick = time.time()

    for epoch in range(model.num_steps):
        # print("epoch:", epoch)
        for t, batch in enumerate(train_loader, 1):
            # print("batch:", t)

            # print()
            # print()
            # print()
            # print('-'*80)
            # print("batch:")
            # print(batch)
            # print('-'*80)
            net.train()
            optimizer.zero_grad()
            loss = model.loss_fn(batch, mode=MODE1)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            train_accm.update(loss.item())


        scheduler.step()

        if epoch % args.test_freq == 0:
            line = 'step {}, lr {:.3e}, train loss {:.4f}, '.format(
                    epoch, optimizer.param_groups[0]['lr'], train_accm.get('loss'))
            line += test(accm=accm, verbose=False)
            logger.info(line)
            accm.reset()
            train_accm.reset()

        if epoch % args.save_freq == 0:
            if args.save_all:
                torch.save(net.state_dict(),
                        os.path.join(save_dir, 'model{}.tar'.format(t)))
            torch.save(net.state_dict(), os.path.join(save_dir, 'model.tar'))

    torch.save(net.state_dict(), os.path.join(save_dir, 'model.tar'))

def test(accm=None, verbose=True):
    net.eval()
    accm = Accumulator(*model.metrics) if accm is None else accm
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            accm.update(model.loss_fn(batch, mode=MODE1, train=False))

    line = accm.info(header='test')
    if verbose:
        logger = get_logger(exp_id, os.path.join(save_dir, 'test.log'))
        logger.info(line)
    return line

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        net.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar')))
        test()
