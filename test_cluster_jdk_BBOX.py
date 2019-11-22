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
from mmdet.core import tensor2imgs#, get_classes, auto_fp16
from mmdet.datasets import build_dataloader, get_dataset

from models.base import compute_FP_removal_loss

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BATCH_SIZE = 20

NUM_IMAGES = 100
RETURN_CLS_INFO = True #if true 89 dimensional inputs, with one hot encoding of 80 classes
REMOVE_ALL_FP_DATA = False

# DATA_NAME ='start100000_tiny%d.json' % NUM_IMAGES
# DATA_NAME ='start0_tiny%d.json' % NUM_IMAGES
# DATA_NAME ='start100000_tiny%d_GaussianCRPS_IOUp5.json' % NUM_IMAGES
# DATA_NAME ='start100000_tiny%d_GausML_IOUp5_minScoreP2.json' % NUM_IMAGES

DATA_NAME ='start100000_tiny%d_GausML_IOUp5_minScoreP2.json' % NUM_IMAGES
# DATA_NAME ='start0_tiny%d_GausML_IOUp5_minScoreP2.json' % NUM_IMAGES

# DATA_NAME ='start0_tiny%d_GaussianCRPS_IOUp9.json' % NUM_IMAGES

COCO_DATA_NAME ='start100000_tiny%d.json' % NUM_IMAGES
# COCO_DATA_NAME ='start0_tiny%d.json' % NUM_IMAGES

DATASET_NAME = '/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_%s' % DATA_NAME
# DATASET_NAME = '/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp9_FPremovalNetProcessed.json'

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
# parser.add_argument('--run_name', type=str, default='bbox_clustering')
# parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--return_cls_info_model', type=bool, default=False) #if true 89 dimensional inputs, with one hot encoding of 80 classes')

args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print("agrs:", args)

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))
print("args:", args)
args.run_name = 'reproduce_NoClassInfo'
# args.run_name = 'bbox_clustering'
# args.run_name = 'bbox_twostage'
# args.run_name = 'trial'
FP_removal_model = 'models/mog.p'
# FP_removal_model = None
if FP_removal_model is not None:
    # fp_run_name = 'fp_removal_network'
    # fp_run_name = 'const_num_fp_removal_network'
    fp_run_name = 'bbox_fp_removal'
    run_name = getattr(args, 'run_name')
    setattr(args, 'run_name', fp_run_name)
    print("args:", args)
    args.return_cls_info_model=True
    fp_removal_module, fp_removal_module_name = load_module(args.modelfile)
    print("type(module)", type(module))
    # sleep(lsdfk)
    fp_removal_model = fp_removal_module.load(args)
    setattr(args, 'run_name', run_name)
    # setattr(args, 'run_name', run_name + '_WITH_fp_removal_network')



if not hasattr(model, 'cluster'):
    raise ValueError('Model is not for clustering')

save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny1000.tar')))


# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp9_minScoreP001_noFP.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp5_minScoreP2_noFP.tar')))


# net.load_state_dict(torch.load(os.path.join(save_dir, 's0tiny100k_GausML_IOUp5_minScoreP2_100epoch_withCls_LamdaP01.tar')))
net.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 's0tiny100_GausML_IOUp5_minScoreP2_2kepoch_noCls_Lamda1k.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp5_minScoreP2_noFP_2ktrain.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp5_minScoreP2_noFP_1ktrain_classInput.tar')))
#reproduce
# net.load_state_dict(torch.load(os.path.join(save_dir, 's0tiny1000_GausML_IOUp5_minScoreP2_2kepoch.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 's0tiny1000_GausML_IOUp5_minScoreP2_200epoch.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 's0tiny100_GausML_IOUp5_minScoreP2_2kepoch.tar')))




# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp5_minScoreP2_withFP_2ktrain.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_tiny100_GausML_IOUp5_minScoreP2_withFP.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_train1000_GausCRPS_ioup9.tar')))#semi decent results, best so far
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_train1000_GausCRPS_ioup5.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_train1000_GausCRPS_ioup9_FPnetProcessed.tar'))) 

# net.load_state_dict(torch.load(os.path.join(save_dir, 'quick_check_cluster_with_FP.tar')))

# net.load_state_dict(torch.load(os.path.join(save_dir, 'model_train1k.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'originalDAC_fullytrained.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_withFP.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'distance_loss_noFP_probably.tar')))
# net.load_state_dict(torch.load(os.path.join(save_dir, 'cluster_loss_withFP.tar')))

net.eval()

if FP_removal_model is not None:
    fp_removal_net = fp_removal_model.net.cuda()
    fp_removal_dir = os.path.join(results_path, module_name, fp_run_name)
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train10k.tar')))
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train1000_GausCRPS_ioup9.tar'))) #semi decent results, best so far
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train1000_GausCRPS_ioup9.tar')))

    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_tiny1000_GausML_IOUp5_minScoreP2_200train.tar'))) 
    #reproduce
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 's0tiny1000_GausML_IOUp5_minScoreP2_200epoch.tar'))) 

    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 's0tiny100k_GausML_IOUp5_minScoreP2_5epoch.tar'))) 
    fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 's0tiny100k_GausML_IOUp5_minScoreP2_100epoch_clsInfo.tar'))) 



    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_tiny1000_shortTraining.tar')))

    fp_removal_net.eval()

test_dataset = BoundingBoxDataset(filename=DATASET_NAME, num_classes=80, mode='FP_removal', return_cls_info=RETURN_CLS_INFO, remove_all_FP=REMOVE_ALL_FP_DATA)
# test_dataset = BoundingBoxDataset(filename=DATASET_NAME, num_classes=80, mode='clustering', return_cls_info=RETURN_CLS_INFO, remove_all_FP=REMOVE_ALL_FP_DATA)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=1, collate_fn=pad_collate)


accm = Accumulator('model ll', 'oracle ll', 'ARI', 'NMI', 'k-MAE')
num_failure = 0
logger = get_logger('{}_{}'.format(module_name, args.run_name),
        os.path.join(save_dir, args.filename))


def plot_coco_img():
    data_root = '/home/lyft/software/mmdetection/data/coco/'#jdk
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    dataset_cfg=dict(
        type='CocoDataset',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file=data_root + 'annotations/instances_train2017_%s' % COCO_DATA_NAME,
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



    img_tensor = dataset[0]['img'].data.unsqueeze(dim=0)
    print(img_tensor.shape)
    img_meta = dataset[0]['img_meta'].data
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)

    h, w, _ = img_meta['img_shape']
    img_show = imgs[0][:h, :w, :]

    bboxes = np.array([[23,24,100,100]])
    labels = np.array([1])

    mmcv.imshow_det_bboxes(
        img_show,
        bboxes,
        labels,
        show=True)#,
        # class_names=class_names,
        # score_thr=score_thr)

    print(dataset)
    print(dataset[0])
    img = dataset[0]['img']
    print('img:', img)
    print('img.shape:', img.shape)
    imgplot = plt.imshow(img)
    plt.show()
    sleep(dataset)

def evaluate_bbox_results(all_bboxes, all_labels, all_batches, BATCH_SIZE, num_images, SHOW_RESULTS=False):
    data_root = '/home/lyft/software/mmdetection/data/coco/'#jdk
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    dataset_cfg=dict(
        type='CocoDataset',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file=data_root + 'annotations/instances_train2017_%s' % COCO_DATA_NAME,
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
    all_results = {}

    num_batches = len(all_batches)
    for batch_idx in range(num_batches):
        # assert(len(all_batches[batch_idx]) == BATCH_SIZE), len(all_batches[batch_idx])
        cur_batch_size = len(all_batches[batch_idx]['gt_objects'])
        assert(cur_batch_size == len(all_batches[batch_idx]['cls_idx']))
        assert(cur_batch_size == len(all_batches[batch_idx]['img_idx']))
        # print("all_batches[batch_idx]:", all_batches[batch_idx])
        for instance_idx in range(cur_batch_size):
            valid_bboxes = []
            cur_batch_bboxes = all_bboxes[batch_idx]
            cur_batch_labels = all_labels[batch_idx]
            for cluster_idx, cur_bboxes in enumerate(cur_batch_bboxes):
                if cluster_idx in cur_batch_labels[instance_idx, :]:
                    valid_bboxes.append(cur_bboxes[instance_idx, :])
            if len(valid_bboxes) == 0:
                continue
            valid_bboxes = torch.cat(valid_bboxes, dim=0)
            #fill in junk scores variances etc.
            valid_bboxes = torch.cat([valid_bboxes, torch.zeros(valid_bboxes.shape[0], 6, device=valid_bboxes.device)], dim=1)
            
            # if (torch.sum(valid_bboxes, dim=1) > 0).any(): #we have valid (non-masked) boxes
            if True:
                cls_idx = int(all_batches[batch_idx]['cls_idx'][instance_idx])
                img_idx = int(all_batches[batch_idx]['img_idx'][instance_idx])
                # if cls_idx == 25 and img_idx == 6:
                #     print("valid_bboxes:", valid_bboxes)
                #     print("detections:")
                #     print(batch['X'][instance_idx])
                #     print("detection cur_batch_labels:", cur_batch_labels[instance_idx, :])
                if img_idx not in all_results:
                    all_results[img_idx] = {}
                assert(cls_idx not in all_results[img_idx])
                # print("valid_bboxes.shape:", valid_bboxes.shape)
                # sleep(lskjfl)
                all_results[img_idx][cls_idx] = valid_bboxes
            else:
                print("all invalid bboxes:")
                print(valid_bboxes)


    ####### get ground truth for comparison ###########
    all_gt_results = {}
    for batch_idx in range(num_batches):
        cur_batch_size = len(all_batches[batch_idx]['gt_objects'])
        assert(cur_batch_size == len(all_batches[batch_idx]['cls_idx']))
        assert(cur_batch_size == len(all_batches[batch_idx]['img_idx']))
        for instance_idx in range(cur_batch_size):
            gt_bboxes = all_batches[batch_idx]['gt_objects'][instance_idx]
            gt_bboxes = torch.cat([gt_bboxes, torch.zeros(gt_bboxes.shape[0], 6, device=gt_bboxes.device)], dim=1)
           
            cls_idx = int(all_batches[batch_idx]['cls_idx'][instance_idx])
            img_idx = int(all_batches[batch_idx]['img_idx'][instance_idx])
            

            if (torch.sum(gt_bboxes, dim=1) > 0).any(): #we have valid (non-masked) boxes
                if img_idx not in all_gt_results:
                    all_gt_results[img_idx] = {}
                assert(cls_idx not in all_gt_results[img_idx])
                all_gt_results[img_idx][cls_idx] = gt_bboxes

    ####### get input detections for comparison ###########
    all_det = {}
    for batch_idx in range(num_batches):
        cur_batch_size = len(all_batches[batch_idx]['X'])
        assert(cur_batch_size == len(all_batches[batch_idx]['cls_idx']))
        assert(cur_batch_size == len(all_batches[batch_idx]['img_idx']))
        for instance_idx in range(cur_batch_size):
            det_bboxes = all_batches[batch_idx]['X'][instance_idx]
            det_bboxes = torch.cat([det_bboxes, torch.zeros(det_bboxes.shape[0], 6, device=det_bboxes.device)], dim=1)
           
            cls_idx = int(all_batches[batch_idx]['cls_idx'][instance_idx])
            img_idx = int(all_batches[batch_idx]['img_idx'][instance_idx])
            

            if (torch.sum(det_bboxes, dim=1) > 0).any(): #we have valid (non-masked) boxes
                if img_idx not in all_det:
                    all_det[img_idx] = {}
                assert(cls_idx not in all_det[img_idx])
                all_det[img_idx][cls_idx] = det_bboxes


    boxes_as_results = []
    num_classes = 80
    for img_idx in range(0,num_images):
    # for img_idx in [6]:
        print('-'*80)
        print("img_idx:", img_idx)        
        if img_idx not in all_results:
            boxes_as_results.append(bbox2result(torch.ones(0), torch.ones(0), num_classes))
        else:
            img_bboxes = []
            img_labels = []
            for cls_idx in sorted(all_results[img_idx].keys()):
            # for cls_idx in sorted(all_results[img_idx].keys())[2:3]:
                print()
                print("non empty class:", cls_idx)
                valid_rows = torch.where(torch.sum(all_results[img_idx][cls_idx], dim=1) > 0)
                valid_bboxes = all_results[img_idx][cls_idx]#[valid_rows]
                print('valid_bboxes:')
                print(valid_bboxes)
                #get gt bounding box count
                if img_idx in all_gt_results and cls_idx in all_gt_results[img_idx] and (torch.sum(all_gt_results[img_idx][cls_idx], dim=1) > 0).any():
                    valid_gt_rows = torch.where(torch.sum(all_gt_results[img_idx][cls_idx], dim=1) > 0)
                    gt_bboxes = all_gt_results[img_idx][cls_idx][valid_gt_rows]
                    gt_bbox_count = gt_bboxes.shape[0]
                else:
                    gt_bbox_count = 0
                    gt_bboxes = None
                print('!'*10)
                print(valid_bboxes.shape[0], 'predicted bounding_boxes,', gt_bbox_count, 'gt bounding_boxes')
                # print("predicted bounding boxes:")
                # print(valid_bboxes)
                # print("ground truth bounding boxes:")
                # print(gt_bboxes)
                img_bboxes.append(valid_bboxes)
                img_labels.append(cls_idx*torch.ones(valid_bboxes.shape[0]))

            # print("torch.cat(img_bboxes, dim=0).shape:", torch.cat(img_bboxes, dim=0).shape)
            # print("torch.cat(img_labels, dim=0).shape:", torch.cat(img_labels, dim=0).shape)
            # sleep(aslkfjskl)
            boxes_as_results.append(bbox2result(torch.cat(img_bboxes, dim=0), torch.cat(img_labels, dim=0), num_classes))

        if SHOW_RESULTS:
            img_tensor = dataset[img_idx]['img'].data.unsqueeze(dim=0)
            img_meta = dataset[img_idx]['img_meta'].data
            print('@'*40)
            print("img_meta:", img_meta)
            # sleep(asldk)
            imgs = tensor2imgs(img_tensor, **img_norm_cfg)

            h, w, _ = img_meta['img_shape']
            img_show = imgs[0][:h, :w, :]

            pred_bboxes = img_meta['scale_factor']*torch.cat(img_bboxes, dim=0)[:,:5].cpu().numpy()
            pred_labels = np.zeros(pred_bboxes.shape[0], dtype=int)

            gt_bboxes_list = []
            for cls_idx in range(80):
            # for cls_idx in sorted(all_results[img_idx].keys())[:2]:
                if cls_idx in all_gt_results[img_idx] and (torch.sum(all_gt_results[img_idx][cls_idx], dim=1) > 0).any():
                    valid_gt_rows = torch.where(torch.sum(all_gt_results[img_idx][cls_idx], dim=1) > 0)
                    gt_bboxes = all_gt_results[img_idx][cls_idx][valid_gt_rows]
                    gt_bboxes_list.append(gt_bboxes)
            gt_bboxes = img_meta['scale_factor']*torch.cat(gt_bboxes_list, dim=0)[:,:5].cpu().numpy()
            gt_labels = np.ones(gt_bboxes.shape[0], dtype=int)

            all_bboxes = np.concatenate([pred_bboxes, gt_bboxes], axis=0)
            all_labels = np.concatenate([pred_labels, gt_labels], axis=0)

            det_bboxes_list = []
            for cls_idx in range(80):
            # for cls_idx in sorted(all_results[img_idx].keys()):
                if cls_idx in all_det[img_idx] and (torch.sum(all_det[img_idx][cls_idx], dim=1) > 0).any():
                    valid_det_rows = torch.where(torch.sum(all_det[img_idx][cls_idx], dim=1) > 0)
                    det_bboxes = all_det[img_idx][cls_idx][valid_det_rows]
                    det_bboxes_list.append(det_bboxes)
            det_bboxes = img_meta['scale_factor']*torch.cat(det_bboxes_list, dim=0)[:,:5].cpu().numpy()
            det_labels = np.ones(det_bboxes.shape[0], dtype=int)

            # print(boxes_as_results)
            # print(boxes_as_results.shape)

                #conversion doesn't seme to do anything..

                # print("boxes_as_results:", boxes_as_results)
                # print("boxes_as_results[0]:", boxes_as_results[0])
                # all_bboxes = np.vstack(boxes_as_results[0])
                # labels = [
                #     np.full(bbox.shape[0], i, dtype=np.int32)
                #     for i, bbox in enumerate(boxes_as_results[0])
                # ]
                # all_labels = np.concatenate(labels)


                # boxes_as_tesor = torch.tensor(all_bboxes)
                # boxes_as_tesor = torch.cat([boxes_as_tesor, torch.zeros(boxes_as_tesor.shape[0], 5)], dim=1)
                # dummy_labels = torch.ones(boxes_as_tesor.shape[0])
                # print("pre conversion all boxes:")
                # print(all_bboxes)
                # print("boxes_as_tesor.shape", boxes_as_tesor.shape)
                # print("dummy_labels.shape", dummy_labels.shape)
                # all_bboxes = bbox2result(boxes_as_tesor, dummy_labels, 80)
                # print("len(all_bboxes):", len(all_bboxes))
                # print("post conversion all_bboxes:", all_bboxes)

            plot_detections=False
            if plot_detections:
                #plot detections
                #press 'k' key to advance to the next image when shown
                mmcv.imshow_det_bboxes(
                    img_show,
                    det_bboxes[:,:4],
                    det_labels,
                    show=True),
                    # score_thr=.05)

            #press 'k' key to advance to the next image when shown
            mmcv.imshow_det_bboxes(
                img_show,
                all_bboxes[:,:4],
                all_labels,
                show=True,
                class_names=['pred', 'gt'])#,
                # class_names=class_names,
                # score_thr=score_thr)                
    mmcv.dump(boxes_as_results, 'results.pkl')
    # print()
    # print()
    # print()
    # print()
    # print(boxes_as_results)
    result_files = results2json(dataset, boxes_as_results, 'results.pkl')
    coco_eval(result_files, ['bbox'], dataset.coco)


def evaluate_bbox_results_DEBUG_WITH_GT(all_bboxes, all_labels, all_batches, BATCH_SIZE, num_images):
    data_root = '/home/lyft/software/mmdetection/data/coco/'#jdk
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    dataset_cfg=dict(
        type='CocoDataset',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file=data_root + 'annotations/instances_train2017_%s' % COCO_DATA_NAME,
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
    all_results = {}
    num_batches = len(all_batches)
    print("num_batches:", num_batches)
    print("type(all_batches):", type(all_batches))

    for batch_idx in range(num_batches):
        # assert(len(all_batches[batch_idx]) == BATCH_SIZE), len(all_batches[batch_idx])
        cur_batch_size = len(all_batches[batch_idx]['gt_objects'])
        assert(cur_batch_size == len(all_batches[batch_idx]['cls_idx']))
        assert(cur_batch_size == len(all_batches[batch_idx]['img_idx']))
        # print("all_batches[batch_idx]:", all_batches[batch_idx])
        for instance_idx in range(cur_batch_size):
            valid_bboxes = all_batches[batch_idx]['gt_objects'][instance_idx]
            valid_bboxes = torch.cat([valid_bboxes, torch.zeros(valid_bboxes.shape[0], 6, device=valid_bboxes.device)], dim=1)
           
            # print("valid_bboxes.shape:", valid_bboxes.shape)
            # print("all_batches[batch_idx]['gt_objects'][instance_idx].shape:", all_batches[batch_idx]['gt_objects'][instance_idx].shape)
            # sleep(skdfj)

            cls_idx = int(all_batches[batch_idx]['cls_idx'][instance_idx])
            img_idx = int(all_batches[batch_idx]['img_idx'][instance_idx])
            

            if (torch.sum(valid_bboxes, dim=1) > 0).any(): #we have valid (non-masked) boxes
                if img_idx not in all_results:
                    all_results[img_idx] = {}
                assert(cls_idx not in all_results[img_idx])
                # print("valid_bboxes.shape:", valid_bboxes.shape)
                # sleep(lskjfl)
                all_results[img_idx][cls_idx] = valid_bboxes

    boxes_as_results = []
    num_classes = 80
    for img_idx in range(num_images):
        # print("img_idx:", img_idx)
        if img_idx not in all_results:
            boxes_as_results.append(bbox2result(torch.ones(0), torch.ones(0), num_classes))
        else:
            img_bboxes = []
            img_labels = []
            for cls_idx in range(80):
                if cls_idx in all_results[img_idx] and (torch.sum(all_results[img_idx][cls_idx], dim=1) > 0).any():
                    # print("non empty class:", cls_idx)
                    # print("bboxes:", all_results[img_idx][cls_idx])
                    # sleep(temp1)
                    valid_rows = torch.where(torch.sum(all_results[img_idx][cls_idx], dim=1) > 0)

                    gt_bboxes = all_results[img_idx][cls_idx][valid_rows]
                    # gt_bboxes = all_results[img_idx][cls_idx][torch.where(all_results[img_idx][cls_idx] != -99)]
                    img_bboxes.append(gt_bboxes)
                    assert(gt_bboxes.shape[0] > 0), (gt_bboxes, all_results[img_idx][cls_idx])
                    img_labels.append(cls_idx*torch.ones(gt_bboxes.shape[0]))
                    # print("gt_bboxes:")
                    # print(gt_bboxes)
                    # print("img_labels:")
                    # print(img_labels)
                    # print("all_results[img_idx][cls_idx][torch.where(all_results[img_idx][cls_idx] != -99)]")
                    # print(all_results[img_idx][cls_idx][torch.where(all_results[img_idx][cls_idx] != -99)])
                    # print("valid_rows:", valid_rows)
                    # sleep(qcheck)
            boxes_as_results.append(bbox2result(torch.cat(img_bboxes, dim=0), torch.cat(img_labels, dim=0), num_classes))
    # sleep(temporary)
    mmcv.dump(boxes_as_results, 'results.pkl')
    # print()
    # print()
    # print()
    # print()
    # print(boxes_as_results)
    result_files = results2json(dataset, boxes_as_results, 'results.pkl')
    coco_eval(result_files, ['bbox'], dataset.coco)

all_bboxes = []
all_labels = []
all_batches = []

isTP_predTP_count = 0
isFP_predTP_count = 0
isTP_predFP_count = 0
isFP_predFP_count = 0

for batch in tqdm(test_loader):
    # plot_coco_img()
    # sleep(pltimg)

    mask = batch['det_mask'].cuda().unsqueeze(dim=2)#.float()




    # FP_removal_model = None
    fp_cutoff_score = 9999999
    if FP_removal_model is not None:
        pred_bbox, logits = fp_removal_net(batch['X'].cuda(), mask=mask)
        ind = (logits > fp_cutoff_score)
        mask[ind] = True          

         
        #debug, check loss
        CHECK_LOSS = False
        if CHECK_LOSS:
            FP_labels = batch['FP_labels'].cuda().float()
            logits = logits.squeeze()
            # print("batch['X'].shape:", batch['X'].shape)
            # print("logits:", logits)
            # print("FP_labels:", FP_labels)
            # print("mask.shape:", mask.shape)

            loss, bcent = compute_FP_removal_loss(logits, FP_labels, mask.squeeze())



            # print("loss:", loss)
            # print("bcent:", bcent)
            # print()
            # sleep(checkloss)
        #end debug, check loss
        # print("mask.shape:", mask.shape)
        # print("logits.shape:", logits.shape)
        # print("ind.shape:", ind.shape)


    true_labels = batch['labels']
    # print("mask.shape:",mask.shape)
    # print("logits.shape:",logits.shape)
    # print("true_labels.shape:",true_labels.shape)
    # print("predicted labels.shape:",labels.shape)
    for instance_idx in range(true_labels.shape[0]):
        for detection_idx in range(true_labels.shape[1]):
            if true_labels[instance_idx][detection_idx] == -99:
                continue
            gt_obj_count = batch['gt_obj_count'][instance_idx]
            # if true_labels[instance_idx][detection_idx] == gt_obj_count and labels[instance_idx][detection_idx] == 51:
            if true_labels[instance_idx][detection_idx] == gt_obj_count and logits[instance_idx][detection_idx] > fp_cutoff_score:
                # is FP, predicted FP
                isFP_predFP_count += 1
            # if true_labels[instance_idx][detection_idx] != gt_obj_count and labels[instance_idx][detection_idx] == 51:
            if true_labels[instance_idx][detection_idx] != gt_obj_count and logits[instance_idx][detection_idx] > fp_cutoff_score:
                # is TP, predicted FP
                isTP_predFP_count += 1
            # if true_labels[instance_idx][detection_idx] == gt_obj_count and labels[instance_idx][detection_idx] != 51:
            if true_labels[instance_idx][detection_idx] == gt_obj_count and logits[instance_idx][detection_idx] < fp_cutoff_score:
                # is FP, predicted TP
                isFP_predTP_count += 1
            # if true_labels[instance_idx][detection_idx] != gt_obj_count and labels[instance_idx][detection_idx] != 51:
            if true_labels[instance_idx][detection_idx] != gt_obj_count and logits[instance_idx][detection_idx] < fp_cutoff_score:
                # is TP, predicted TP                
                isTP_predTP_count += 1


    #only keep gt obects that correspond to detections that were not removed by fp_removal network to upper bound performance
    remaining_gt_objects_each_instance = []
    for instance_idx in range(true_labels.shape[0]):
        remaining_gt_indices = []
        gt_obj_count = batch['gt_obj_count'][instance_idx]
        for detection_idx in range(true_labels.shape[1]):
            if logits[instance_idx][detection_idx] < fp_cutoff_score and batch['labels'][instance_idx][detection_idx] < gt_obj_count and\
               batch['det_mask'][instance_idx][detection_idx] != 1 and\
               batch['labels'][instance_idx][detection_idx] not in remaining_gt_indices:
                remaining_gt_indices.append(batch['labels'][instance_idx][detection_idx])
        remaining_gt_objects = []
        # print("remaining_gt_indices:", remaining_gt_indices)
        for gt_idx in remaining_gt_indices:
            # print(gt_idx)
            remaining_gt_objects.append(batch['gt_objects'][instance_idx][int(gt_idx.item())])

        padded_remaining_gt_objects = -99*torch.ones(batch['gt_objects'].shape[1], batch['gt_objects'].shape[2])
        if len(remaining_gt_indices) > 0:
            # print("len(remaining_gt_indices):", len(remaining_gt_indices))
            remaining_gt_objects = torch.stack(remaining_gt_objects, dim=0)
            padded_remaining_gt_objects[:remaining_gt_objects.shape[0]] = remaining_gt_objects
            # print("remaining_gt_objects.shape:", remaining_gt_objects.shape)
        # else:
            # print("no remaining gt objects")
        remaining_gt_objects_each_instance.append(padded_remaining_gt_objects)

    # print("batch['gt_objects'].shape:", batch['gt_objects'].shape)
    batch['gt_objects'] = torch.stack(remaining_gt_objects_each_instance, dim=0)
    # print("batch['gt_objects'].shape:", batch['gt_objects'].shape)
    # sleep(temps)

    # gt_objects = torch.tensor(img_data['assoc_gt_bboxes'])
    # cur_gt_object_count = gt_objects.shape[0]

    # print(batch)
    # sleep(sldkf)
    # gt_count_cur_img = batch['gt_count_per_img'].item()
    # bboxes, labels, fail = model.cluster(batch['X'].cuda(),
    #         max_iter=gt_count_cur_img-1, verbose=False, check=True, mask=mask)

    # gt_count_cur_img = batch['gt_count_per_img']
    # bboxes, labels, fail = model.cluster(batch['X'].cuda(),
    #         max_iter=args.max_iter, verbose=False, check=True, mask=mask,
            # max_iter_tensor=gt_count_cur_img)

    # print("batch['X'].shape:", batch['X'].shape)
    # sleep(lwnebo)
    bboxes, labels, fail = model.cluster(batch['X'][:,:,:9].cuda(), #for fp removal network that uses cls info and clustering network that does not
    # bboxes, labels, fail = model.cluster(batch['X'].cuda(),
            max_iter=args.max_iter, verbose=False, check=True, mask=mask)

    # print(len(bboxes))
    # print(labels.shape)
    # sleep(temps)

    all_bboxes.append(bboxes)
    all_labels.append(labels)
    all_batches.append(batch)



    # print("predicted labels:", labels)
    # print("true_labels:", true_labels)
    # sleep(labelcheck)
    # print("predicted bboxes:", bboxes)
    # print("gt bboxes:", batch['gt_objects'])
    # sleep(lsfjalskdfj)
    # print("batch['cls_idx']:", batch['cls_idx'])
    # print("batch['img_idx']:", batch['img_idx'])
    num_failure += int(fail)

print("isTP_predTP_count:", isTP_predTP_count)
print("isFP_predTP_count:", isFP_predTP_count)
print("isTP_predFP_count:", isTP_predFP_count)
print("isFP_predFP_count:", isFP_predFP_count)
# sleep(fpcounts)
# evaluate_bbox_results(all_bboxes, all_labels, all_batches, BATCH_SIZE, NUM_IMAGES)
evaluate_bbox_results_DEBUG_WITH_GT(all_bboxes, all_labels, all_batches, BATCH_SIZE, NUM_IMAGES)




logger.info(accm.info())
logger.info('number of failure cases {}'.format(num_failure))

#tp_count: 2848
#fp_count: 38301

#counting with 51 on label
fp_cutoff_score = 0
isTP_predTP_count: 1492
isFP_predTP_count: 2177
isTP_predFP_count: 1356
isFP_predFP_count: 36124

fp_cutoff_score = 2
isTP_predTP_count: 1660
isFP_predTP_count: 4656
isTP_predFP_count: 1188
isFP_predFP_count: 33645

fp_cutoff_score = 4
isTP_predTP_count: 1303
isFP_predTP_count: 6144
isTP_predFP_count: 1545
isFP_predFP_count: 32157

#counting with logits
fp_cutoff_score = 0
isTP_predTP_count: 1795
isFP_predTP_count: 3646
isTP_predFP_count: 1053
isFP_predFP_count: 34655

fp_cutoff_score = 1
isTP_predTP_count: 2300
isFP_predTP_count: 7249
isTP_predFP_count: 548
isFP_predFP_count: 31052

fp_cutoff_score = 2
isTP_predTP_count: 2570
isFP_predTP_count: 11760
isTP_predFP_count: 278
isFP_predFP_count: 26541

fp_cutoff_score = 3
isTP_predTP_count: 2718
isFP_predTP_count: 17541
isTP_predFP_count: 130
isFP_predFP_count: 20760

fp_cutoff_score = 4
isTP_predTP_count: 2801
isFP_predTP_count: 25081
isTP_predFP_count: 47
isFP_predFP_count: 13220
