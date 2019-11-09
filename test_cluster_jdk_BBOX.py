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

from data.bounding_boxes import BoundingBoxDataset, pad_collate
from torch.utils.data import DataLoader

import mmcv
from mmdet.core import results2json, coco_eval, wrap_fp16_model, bbox2result
from mmdet.datasets import build_dataloader, get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--run_name', type=str, default='bbox_clustering')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print("agrs:", args)

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))
print("args:", args)
args.run_name = 'bbox_clustering'
FP_removal_model = 'models/mog.p'
if FP_removal_model is not None:
    # fp_run_name = 'fp_removal_network'
    # fp_run_name = 'const_num_fp_removal_network'
    fp_run_name = 'bbox_fp_removal'
    run_name = getattr(args, 'run_name')
    setattr(args, 'run_name', fp_run_name)
    print("args:", args)

    fp_removal_module, fp_removal_module_name = load_module(args.modelfile)
    fp_removal_model = fp_removal_module.load(args)
    setattr(args, 'run_name', run_name)
    # setattr(args, 'run_name', run_name + '_WITH_fp_removal_network')




if not hasattr(model, 'cluster'):
    raise ValueError('Model is not for clustering')

save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

net.load_state_dict(torch.load(os.path.join(save_dir, 'model_train1k.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'originalDAC_fullytrained.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_withFP.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_noFP_probably.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'cluster_loss_withFP.tar')))

net.eval()

if FP_removal_model is not None:
    fp_removal_net = fp_removal_model.net.cuda()
    fp_removal_dir = os.path.join(results_path, module_name, fp_run_name)
    fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train10k.tar')))

    fp_removal_net.eval()

BATCH_SIZE = 5
NUM_IMAGES = 1000
test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100.json',\
# test_dataset = BoundingBoxDataset(filename='/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start0_tiny1000.json',\
                                         num_classes=80, mode='FP_removal')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=1, collate_fn=pad_collate)


accm = Accumulator('model ll', 'oracle ll', 'ARI', 'NMI', 'k-MAE')
num_failure = 0
logger = get_logger('{}_{}'.format(module_name, args.run_name),
        os.path.join(save_dir, args.filename))

all_results = {}


def evaluate_bbox_results(bboxes, labels, batch, BATCH_SIZE, num_images):
    data_root = '/home/lyft/software/mmdetection/data/coco/'#jdk
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    dataset_cfg=dict(
        type='CocoDataset',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file=data_root + 'annotations/instances_train2017_start0_tiny1000.json',
        # ann_file=data_root + 'annotations/instances_val2017_start4k_tiny1000.json', 
        # ann_file=data_root + 'annotations/instances_val2017_start4200_tiny200.json', #43, 48, 49 are bad 
        # ann_file=data_root + 'annotations/instances_val2017_start4320_tiny10.json', 
        # img_prefix=data_root + 'val2017/',        
        # ann_file=data_root + 'annotations/instances_train2017_tiny5.json', #jdk
        # ann_file=data_root + 'annotations/instances_train2017.json',#jdk
        img_prefix=data_root + 'train2017/',        
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        # with_label=False,
        # test_mode=True))
        with_crowd=True, #jdk temp for upper bounding improvement, REMOVE FOR REAL TESTING, replace with 2 lines above!!
        with_label=True, #jdk temp for upper bounding improvement, REMOVE FOR REAL TESTING, replace with 2 lines above!!
        test_mode=False)

    dataset = get_dataset(dataset_cfg)
    assert(len(batch['cls_idx']) == BATCH_SIZE)
    assert(len(batch['img_idx']) == BATCH_SIZE)
    for idx in range(BATCH_SIZE):
        valid_bboxes = []
        for cluster_idx, cur_bboxes in enumerate(bboxes):
            if cluster_idx in labels[idx, :]:
                valid_bboxes.append(cur_bboxes[idx,:])
        if len(valid_bboxes) == 0:
            continue
        valid_bboxes = torch.cat(valid_bboxes, dim=0)
        #fill in junk scores variances etc.
        valid_bboxes = torch.cat([valid_bboxes, torch.zeros(valid_bboxes.shape[0], 6, device=valid_bboxes.device)], dim=1)
        cls_idx = int(batch['cls_idx'][idx])
        img_idx = int(batch['img_idx'][idx])
        if img_idx not in all_results:
            all_results[img_idx] = {}
        assert(cls_idx not in all_results[img_idx])
        # print("valid_bboxes.shape:", valid_bboxes.shape)
        # sleep(lskjfl)
        all_results[img_idx][cls_idx] = valid_bboxes

    boxes_as_results = []
    num_classes = 80
    for img_idx in range(num_images):
        if img_idx not in all_results:
            boxes_as_results.append(bbox2result(torch.ones(0), torch.ones(0), num_classes))
        else:
            img_bboxes = []
            img_labels = []
            for cls_idx in sorted(all_results[img_idx].keys()):
                img_bboxes.append(all_results[img_idx][cls_idx])
                img_labels.append(cls_idx*torch.ones(all_results[img_idx][cls_idx].shape[0]))

            boxes_as_results.append(bbox2result(torch.cat(img_bboxes, dim=0), torch.cat(img_labels, dim=0), num_classes))
    mmcv.dump(boxes_as_results, 'results.pkl')
    # print()
    # print()
    # print()
    # print()
    # print(boxes_as_results)
    result_files = results2json(dataset, boxes_as_results, 'results.pkl')
    coco_eval(result_files, ['bbox'], dataset.coco)

for batch in tqdm(test_loader):
    if FP_removal_model is not None:
        pred_bbox, logits = fp_removal_net(batch['X'].cuda())
        mask = (logits > 0.0)

    else:
        mask = None

    bboxes, labels, fail = model.cluster(batch['X'].cuda(),
            max_iter=args.max_iter, verbose=False, check=True, mask=mask)
    true_labels = to_numpy(batch['labels'].argmax(-1))
    print("predicted labels:", labels)
    print("true_labels:", true_labels)
    print("bboxes:", bboxes)
    print("batch['cls_idx']:", batch['cls_idx'])
    print("batch['img_idx']:", batch['img_idx'])
    evaluate_bbox_results(bboxes, labels, batch, BATCH_SIZE, NUM_IMAGES)

    num_failure += int(fail)



logger.info(accm.info())
logger.info('number of failure cases {}'.format(num_failure))

