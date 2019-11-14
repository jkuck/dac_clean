import torch
from torch.utils.data import Dataset
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import argparse
import os

import sys
sys.path.insert(0, "/home/lyft/software/dac_clean")
from utils.misc import load_module
from utils.paths import results_path, benchmarks_path

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--run_name', type=str, default='bbox_clustering')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
args, _ = parser.parse_known_args()

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))
print("bounding_boxes.py args:", args)
args.run_name = 'bbox_clustering'
FP_removal_model = 'models/mog.p'
FP_removal_model = None
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
    fp_removal_net = fp_removal_model.net.cuda()
    fp_removal_dir = os.path.join(results_path, module_name, fp_run_name)
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train10k.tar')))
    fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_train1000_GausCRPS_ioup9.tar')))
    # fp_removal_net.load_state_dict(torch.load(os.path.join(fp_removal_dir, 'model_tiny1000_shortTraining.tar')))

    fp_removal_net.eval()



def create_FPremovalNet_processed_dataset(old_data_file, new_data_file):
    '''
    run detection through fp removal network and save with detections removed, and unassociated gt removed
    '''
    with open(old_data_file, "r") as f:
        all_img_data = json.load(f)

    new_all_img_data = []

    for (img_idx, img_data) in enumerate(all_img_data):
        new_img_data = {}
        for cls_idx, cls_data in img_data.items():

            # something like this to only include positive number of detections if (torch.tensor(cls_data['det_assoc']) != -1).any()

            gt_objects = torch.tensor(cls_data['assoc_gt_bboxes'])
            gt_object_count = gt_objects.shape[0]
            X = torch.tensor(cls_data['det_bboxes'])
            # X = X[:, :5] #only position and score
            X = X[:, :9] #position, score, and uncertainty
            labels = torch.tensor(cls_data['det_assoc'])#.unsqueeze(dim=1)
            labels[torch.where(labels == -1)] = gt_object_count #convert from -1 to gt_object_count indicating FP class
            if gt_object_count == 0:
                continue
            # FP_removal_model = None
            fp_cutoff_score = 1
            x_cuda = X.cuda().unsqueeze(dim=0)
            pred_bbox, logits = fp_removal_net(x_cuda)
            logits = logits.squeeze()
            # print("X.shape:", X.shape)
            # print("logits.shape:", logits.shape)
            # print("labels.shape:", labels.shape)
            # print("x_cuda.shape:", x_cuda.shape)
            # print()
            # assert(logits.shape == labels.shape), (X.shape, logits.shape, labels.shape)
            if logits.shape == labels.shape: #really weird, this should alwayus be true..
                X = X[logits < fp_cutoff_score]
                assert(len(X.shape) == 2), (X.shape, logits.shape, labels.shape)              
                # print('a unique labels:', np.unique(labels.numpy(), return_inverse=True))
                # print("original labels:", labels)
                labels = labels[logits < fp_cutoff_score]
                trimmed_labels, inverse_ind = np.unique(labels.numpy(), return_inverse=True)
                # print("trimmed_labels, inverse_ind:", trimmed_labels, inverse_ind)
                if gt_object_count in trimmed_labels:
                    gt_object_count = trimmed_labels.shape[0] - 1
                    new_labels_for_gather = torch.tensor(trimmed_labels[:-1]).unsqueeze(dim=1).repeat(1,4)
                    # print("1 gt_objects.shape:", gt_objects.shape)
                    # print("1 new_labels_for_gather.shape:", new_labels_for_gather.shape)
                    gt_objects = torch.gather(gt_objects, dim=0, index=new_labels_for_gather)
                    _, new_labels = np.unique(trimmed_labels, return_inverse=True)
                    new_labels = torch.tensor(new_labels, dtype=torch.long)
                    labels = torch.gather(new_labels, dim=0, index=torch.tensor(inverse_ind))
                    # print("gt_objects.shape:", gt_objects.shape)
                    # print("trimmed_labels:", trimmed_labels)
                    # print("new_labels_for_gather.shape:", new_labels_for_gather.shape)
                    # print("updated labels:", labels)
                    # sleep(xyz)
                    # print("a")
                else:
                    # print("b")
                    gt_object_count = trimmed_labels.shape[0]
                    new_labels_for_gather = torch.tensor(trimmed_labels).unsqueeze(dim=1).repeat(1,4)
                    # print("2 gt_objects.shape:", gt_objects.shape)
                    # print("2 new_labels_for_gather.shape:", new_labels_for_gather.shape)
                    # print("2 gt_objects:", gt_objects)
                    # print("2 new_labels_for_gather:", new_labels_for_gather)

                    gt_objects = torch.gather(gt_objects, dim=0, index=new_labels_for_gather)
                    _, new_labels = np.unique(trimmed_labels, return_inverse=True)
                    new_labels = torch.tensor(new_labels, dtype=torch.long)
                    labels = torch.gather(new_labels, dim=0, index=torch.tensor(inverse_ind))
                assert(labels.shape[0] == X.shape[0]), (labels.shape, X.shape)

                if X.shape[0] == 0:
                    continue

                new_gt_obj_count = gt_objects.shape[0]
                labels[torch.where(labels == new_gt_obj_count)] = -1 #convert FP back to -1
                assert((labels < new_gt_obj_count).all())

                new_cls_data = {'all_gt_bboxes': cls_data['all_gt_bboxes'],
                                'assoc_gt_bboxes': gt_objects.tolist(),
                                'det_bboxes': X[:,:9].tolist(),
                                'det_assoc': labels.tolist(),
                                'merged_detections': cls_data['merged_detections'],}
                new_img_data[cls_idx] = new_cls_data
            if len(new_img_data) > 0:
                new_all_img_data.append(new_img_data)


    with open(new_data_file, "w") as f:
        json.dump(new_all_img_data, f)        
             
class BoundingBoxDataset(Dataset):
    '''
    Bounding Box dataset
    # NOTE, currently don't return class info

    input should have following info: (for reference, we get data in
     /home/lyft/software/perceptionresearch/object_detection/mmdetection/mmdet/core/post_processing/bbox_nms.py

     and save it in /home/lyft/software/perceptionresearch/object_detection/mmdetection/tools/test.py)
    all_gt_bboxes: torch.tensor (num_gt_bboxes, 4) [x1, y1, x2, y2] format (x2 > x1, y2 > y1)
    assoc_gt_bboxes: torch.tensor (num_gt_bboxes associated with det, 4) [x1, y1, x2, y2] format (x2 > x1, y2 > y1)
    det_bboxes: torch.tensor (num_det, 9) format:
        [x1, y1, x2, y2, score, x, x std, y std, ln(w + 1) std, ln(h + 1) std]
        DOUBLE CHECK WE ARE STORING THE standard deviations! (thought we trained to 
        predict something else that is more stable?)
    det_assoc: torch.tensor (num_det) gt indices that detection are associated to (-1 for FP, NOTE we change to gt_object count)
    merged_detections: (number_merged_dets, 9)
        [x1, y1, x2, y2, score, x, x std, y std, ln(w + 1) std, ln(h + 1) std]


    '''

    def __init__(self, filename='./jdk_data/bboxes_with_assoc.json', num_classes=80,
                 mode='FP_removal'):
        """
        Args:
            filename (string): Path to the json file with data.
            num_classes (int): number of classes
            mode (string): 'FP_removal' or 'clustering'
                FP_removal gives all TP and FP bboxes for FP removal training
                clustering gives only TP bboxes, with some randomly removed
        """
        print("about to load data")
        with open(filename, "r") as f:
            all_img_data = json.load(f)
        # print("all_img_data:", all_img_data)
        print("file loaded")

        if mode == 'FP_removal':
            self.all_img_data = [(img_idx, cls_idx, img_cls_data) for (img_idx, img_data) in enumerate(all_img_data) for cls_idx, img_cls_data in img_data.items()]
        elif mode == 'clustering': #only include instances with some TP detections
            # for img_data in all_img_data:
            #     for cls_idx, img_cls_data in img_data.items():
            #         print("img_cls_data['det_assoc']:", img_cls_data['det_assoc'])
            #         print("img_cls_data['det_assoc'] != -1:", img_cls_data['det_assoc'] != -1)
            #         sleep(ldfjs)
            self.all_img_data = [(img_idx, cls_idx, img_cls_data) for (img_idx, img_data) in enumerate(all_img_data)\
                                              for cls_idx, img_cls_data in img_data.items()\
                                              if (torch.tensor(img_cls_data['det_assoc']) != -1).any()]
        # print("self.all_img_data:", self.all_img_data)
        # print("all_img_data[0]:", all_img_data[0])
        # print("all_img_data[0][0]:", all_img_data[0][0])
        
        # self.all_img_data = self.all_img_data[:len(self.all_img_data)//5]
        print("data turned into list")

        self.num_classes = num_classes
        self.mode = mode
        count_fraction_tp = True
        if count_fraction_tp:
            tp_count = 0
            fp_count = 0
            total_det_count = 0
            gt_object_count = 0
            for img_idx, cls_idx, img_data in self.all_img_data:
                fp_count += img_data['det_assoc'].count(-1)
                tp_count += len(img_data['det_assoc']) - img_data['det_assoc'].count(-1)
                total_det_count += torch.tensor(img_data['det_bboxes']).shape[0]
                gt_objects = torch.tensor(img_data['assoc_gt_bboxes'])
                cur_gt_object_count = gt_objects.shape[0]
                gt_object_count += cur_gt_object_count
                # print(img_data['det_assoc'])
                # print("img_data['det_assoc'].count(-1):", img_data['det_assoc'].count(-1))
                # print("img_data['det_assoc'].count(0):", img_data['det_assoc'].count(0))
                # sleep(temp)
            print("tp_count:", tp_count)
            print("fp_count:", fp_count)
            assert(tp_count + fp_count == total_det_count), (tp_count, fp_count,tp_count + fp_count, total_det_count)
            print("gt_object_count:", gt_object_count)
            # sleep(counts)
    def __len__(self):
        return len(self.all_img_data)

    def __getitem__(self, idx):
        # is this helpful, what does it do?
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        cur_img_idx = self.all_img_data[idx][0]
        cur_cls_idx = self.all_img_data[idx][1]
        cur_data = self.all_img_data[idx][2]
        gt_objects = torch.tensor(cur_data['assoc_gt_bboxes'])
        gt_object_count = gt_objects.shape[0]
        if gt_object_count > 0:
            assert(gt_objects.shape[1] == 4)
        else:
            gt_objects = torch.zeros((0,4))

        X = torch.tensor(cur_data['det_bboxes'])
        # X = X[:, :5] #only position and score
        if X.shape[0] != 0:
            X = X[:, :9] #position, score, and uncertainty
            # print('orig X:', X)
            X[:, 2] = X[:, 2] - X[:, 0]
            X[:, 3] = X[:, 3] - X[:, 1]
            # print('xywh X:', X)
            # X[:, 7] = torch.exp(X[:, 7])
            # X[:, 8] = torch.exp(X[:, 8])
            # print('transform std xywh X:', X)
            # print("X.shape:", X.shape)
            X = torch.cat([X, torch.zeros((X.shape[0], 80), device=X.device)], dim=1)
            X[:, 9+int(cur_cls_idx)] = 1

        else:
            X = torch.zeros((0,89), device=X.device)

        # print('X.shape:', X.shape)

        assert(len(X.shape) == 2), (X.shape)        
        labels = torch.tensor(cur_data['det_assoc'])#.unsqueeze(dim=1)


        # MIN_SCORE = .2
        # if MIN_SCORE is not None:
        #     labels = labels[torch.where(X[:,4] > MIN_SCORE)]            
        #     X = X[torch.where(X[:,4] > MIN_SCORE)]
        # # print("X.shape:", X.shape)
        # # sleep(xshape)

        remove_all_FP = False
        if remove_all_FP:
            #remove FP detections and labels
            X = X[torch.where(labels != -1)]
            labels = labels[torch.where(labels != -1)]


        assert((labels < gt_object_count).all()), (labels, gt_object_count)
        # assert((labels != 0).all()), (labels, gt_object_count)
        labels[torch.where(labels == -1)] = gt_object_count #convert from -1 to gt_object_count indicating FP class
        FP_labels = torch.zeros(labels.shape)
        FP_labels[torch.where(labels == gt_object_count)] = 1
        # X: (num_bbox, dimensionality) all bounding boxes
        # gt_objects: (num_gt_bbox, 4) gt output bounding boxes
        # labels: (num_bbox) integers in [0, gt_object_count-1] assigning each X to a gt bbox or gt_object_count 
        #         representing a FP
        # FP_labels: (num_bbox) {0, 1}, classify each X as FP (1) or TP (0)
        assert(X.shape[1] == 5 or X.shape[1] == 9 or X.shape[1] == 89)
        
        assert(len(FP_labels.shape) == 1)




        if self.mode == 'clustering': #keep 1 to gt_count gt_objects
        # if self.mode == 'asdf': #keep 1 to gt_count gt_objects


            run_filter_network_to_remove_FP = False
            if run_filter_network_to_remove_FP:
                # FP_removal_model = None
                fp_cutoff_score = 1
                x_cuda = X.cuda().unsqueeze(dim=0)
                pred_bbox, logits = fp_removal_net(x_cuda)
                logits = logits.squeeze()
                # print("X.shape:", X.shape)
                # print("logits.shape:", logits.shape)
                # print("labels.shape:", labels.shape)
                # print("x_cuda.shape:", x_cuda.shape)
                # print()
                # assert(logits.shape == labels.shape), (X.shape, logits.shape, labels.shape)
                if logits.shape == labels.shape: #really weird, this should alwayus be true..
                    X = X[logits < fp_cutoff_score]
                    assert(len(X.shape) == 2), (X.shape, logits.shape, labels.shape)              
                    # print('a unique labels:', np.unique(labels.numpy(), return_inverse=True))
                    labels = labels[logits < fp_cutoff_score]
                    trimmed_labels, inverse_ind = np.unique(labels.numpy(), return_inverse=True)
                    # print("trimmed_labels, inverse_ind:", trimmed_labels, inverse_ind)
                    if gt_object_count in trimmed_labels:
                        gt_object_count = trimmed_labels.shape[0] - 1
                        new_labels_for_gather = torch.tensor(trimmed_labels[:-1]).unsqueeze(dim=1).repeat(1,4)
                        gt_objects = torch.gather(gt_objects, dim=0, index=new_labels_for_gather)
                        _, new_labels = np.unique(trimmed_labels, return_inverse=True)
                        new_labels = torch.tensor(new_labels, dtype=torch.long)
                        labels = torch.gather(new_labels, dim=0, index=torch.tensor(inverse_ind))
                        # print("gt_objects.shape:", gt_objects.shape)
                        # print("trimmed_labels:", trimmed_labels)
                        # print("new_labels_for_gather.shape:", new_labels_for_gather.shape)
                        # print("updated labels:", labels)
                        # sleep(xyz)
                        # print("a")
                    else:
                        # print("b")
                        gt_object_count = trimmed_labels.shape[0]
                        new_labels_for_gather = torch.tensor(trimmed_labels).unsqueeze(dim=1).repeat(1,4)
                        gt_objects = torch.gather(gt_objects, dim=0, index=new_labels_for_gather)
                        _, new_labels = np.unique(trimmed_labels, return_inverse=True)
                        new_labels = torch.tensor(new_labels, dtype=torch.long)
                        labels = torch.gather(new_labels, dim=0, index=torch.tensor(inverse_ind))
                    assert(labels.shape[0] == X.shape[0]), (labels.shape, X.shape)
                    # print('final labels:', labels)
            #randomly select 1 to gt_object_count detections to keep
            # print("gt_object_count:", gt_object_count)
            if gt_object_count > 0:
                num_gt_objects_to_keep = 1 + torch.randint(gt_object_count,(1,)).item()
            else:
                num_gt_objects_to_keep = 0

            assert(len(X.shape) == 2), (X.shape)
            # print("asldfkj:", gt_object_count, num_gt_objects_to_keep)
            if gt_object_count not in labels:
                remove_all_FP = True

            permuted_gt_indices = torch.randperm(gt_object_count)
            # print()
            # print()
            # print('-'*80)
            # print("permuted_gt_indices:", permuted_gt_indices)
            # print("gt_object_count:", gt_object_count)
            # print("num_gt_objects_to_keep:", num_gt_objects_to_keep)
            if not remove_all_FP:
                permuted_gt_indices = torch.cat([permuted_gt_indices, torch.tensor([-9835739])], dim=0) #JUNK, make sure removed
                # print("a permuted_gt_indices:", permuted_gt_indices)
                temp_idx = permuted_gt_indices[num_gt_objects_to_keep].item()
                # print("temp_idx:", temp_idx)
                permuted_gt_indices[num_gt_objects_to_keep] = gt_object_count
                # print("c permuted_gt_indices:", permuted_gt_indices)
                if num_gt_objects_to_keep < gt_object_count:
                    permuted_gt_indices[-1] = temp_idx
                # print("d permuted_gt_indices:", permuted_gt_indices)
                gt_obj_indices_to_keep = permuted_gt_indices[:num_gt_objects_to_keep+1]
                # print("gt_obj_indices_to_keep:", gt_obj_indices_to_keep)
            else:
                gt_obj_indices_to_keep = permuted_gt_indices[:num_gt_objects_to_keep]
                
            # print("permuted_gt_indices:", permuted_gt_indices)
            # print("gt_object_count:", gt_object_count)
            # print("num_gt_objects_to_keep:", num_gt_objects_to_keep)

            # sleep(perm)


            det_indices_to_keep = torch.zeros_like(labels, dtype=torch.bool)
            # print("labels.type():", labels.type())
            # print("gt_obj_indices_to_keep.type():", gt_obj_indices_to_keep.type())
            # print("det_indices_to_keep.type():", det_indices_to_keep.type())

            for gt_idx in gt_obj_indices_to_keep:
                det_indices_to_keep += (labels == gt_idx)  #logical or

            old_labels = labels #debuggin
            # print()
            # print()
            # print('-'*80)
            # print("num_gt_objects_to_keep:", num_gt_objects_to_keep)
            # print("gt_object_count:", gt_object_count)
            # print("gt_obj_indices_to_keep:", gt_obj_indices_to_keep)

            # print("a labels:", labels)

            if remove_all_FP:
                gt_objects = gt_objects[gt_obj_indices_to_keep]
            else: # skip 'ground truth index' for false positives at the end
                gt_objects = gt_objects[gt_obj_indices_to_keep[:-1]]
            X = X[det_indices_to_keep]
            labels = labels[det_indices_to_keep]

            #reindex labels
            # print("old_labels:", old_labels)
            # print("labels:", labels)
            gt_obj_indices_to_keep1, _ = np.unique(labels.numpy(), return_inverse=True)
            _, new_gt_obj_indices_to_keep = np.unique(gt_obj_indices_to_keep1, return_inverse=True)
            
            # print("gt_obj_indices_to_keep1:", gt_obj_indices_to_keep1)
            # print("torch.sort(gt_obj_indices_to_keep)[0]:", torch.sort(gt_obj_indices_to_keep)[0])

            # print("gt_obj_indices_to_keep1:", gt_obj_indices_to_keep1)
            # print("gt_obj_indices_to_keep:", gt_obj_indices_to_keep)
            assert((torch.tensor(gt_obj_indices_to_keep1) == torch.sort(gt_obj_indices_to_keep)[0]).all()), (gt_obj_indices_to_keep1, gt_obj_indices_to_keep)
            # print("gt_obj_indices_to_keep1:", gt_obj_indices_to_keep1)
            # print("a new_gt_obj_indices_to_keep:", new_gt_obj_indices_to_keep)

            new_gt_obj_indices_to_keep = torch.cat([torch.tensor(new_gt_obj_indices_to_keep, dtype=torch.int), 
                                                    -1*torch.ones(gt_object_count - num_gt_objects_to_keep, dtype=torch.int)],
                                                    dim=0)
            # print("b new_gt_obj_indices_to_keep:", new_gt_obj_indices_to_keep)
            _, permutation_indices = torch.sort(permuted_gt_indices)
            # print("permuted_gt_indices:", permuted_gt_indices)
            # print("permutation_indices:", permutation_indices)
            # if not remove_all_FP:
            #     permutation_indices = torch.cat([permutation_indices, torch.tensor([gt_object_count])], dim=0)
            # print("b permutation_indices:", permutation_indices)
            orig_order_new_gt_obj_indices_to_keep = new_gt_obj_indices_to_keep[permutation_indices]
            # print("orig_order_new_gt_obj_indices_to_keep:", orig_order_new_gt_obj_indices_to_keep)

            new_labels = torch.gather(orig_order_new_gt_obj_indices_to_keep, dim=0, index=labels)
            # print("new_labels:", new_labels)
            assert((new_labels >= 0).all()), new_labels
            if remove_all_FP:
                assert((new_labels < num_gt_objects_to_keep).all())
            else:
                assert((new_labels < num_gt_objects_to_keep+1).all())
            labels=new_labels.long()

            # print("X.shape:", X.shape)
            # print("labels.shape:", labels.shape)
            # print("gt_objects.shape:", gt_objects.shape)

            # print("old_labels:", old_labels)
            # print("labels:", labels)

            # print("gt_object_count:", gt_object_count)
            # print("num_gt_objects_to_keep:", num_gt_objects_to_keep)


            # sleep(temps)
            # sleep(labelshape)
            FP_labels = torch.zeros(labels.shape)

            #the number of gt that match to the sampled TP detections
            # remaining_gt_count = torch.max(labels) + 1
            # labels = F.one_hot(labels, remaining_gt_count)

        if self.mode == 'randomly_remove_some_detections': #accidentally implemented this for 'clustering first'
        # if self.mode == 'clustering': #accidentally implemented this for 'clustering first'
            #remove FP detections and labels
            X = X[torch.where(labels != gt_object_count)]
            labels = labels[torch.where(labels != gt_object_count)]
            #randomly select 1 to num_detections detections to keep
            num_detections = X.shape[0]
            num_detections_to_keep = 1 + torch.randint(num_detections,(1,)).item()
            indices_to_keep = torch.randperm(num_detections)[:num_detections_to_keep]
            X = X[indices_to_keep]
            labels = labels[indices_to_keep]
            unique_labels, new_labels = np.unique(labels.numpy(), return_inverse=True)
            #only keep gt objects that match to one of the sampled TP detections
            # print()
            # print("gt_object_count:", gt_object_count)
            # print("labels:", labels)
            # print("unique_labels:", unique_labels)
            # print("gt_objects.shape:", gt_objects.shape)
            gt_objects = gt_objects[unique_labels]
            #reindex labels
            labels = torch.tensor(new_labels)
            # sleep(labelshape)
            FP_labels = torch.zeros(labels.shape)
        # print("X.shape:", X.shape)
        # print("labels.shape:", labels.shape)
        # print("gt_objects.shape:", gt_objects.shape)
        # sleep(temp)
        # print("FP_labels.shape:", FP_labels.shape)
        # print()
        # print()
        # if len(X.shape) == 3:
        #     assert(X.shape[0] == 0)
        #     print(len(X.shape))
        #     X = X.squeeze(dim=0)
        #     print(len(X.shape))

        assert(len(X.shape) == 2), (X.shape)
        sample = {'X':X, 'labels':labels, "gt_objects":gt_objects,\
                  'FP_labels': FP_labels, 'cls_idx': cur_cls_idx, 'img_idx':  cur_img_idx,
                  'gt_obj_count': gt_objects.shape[0]}


        return sample

def pad_collate(batch, verbose = False):
    # max_det = 0
    # max_gt = 0
    # for data in batch:
    #     if data['labels'].shape[0] > max_det:
    #         max_det = data['labels'].shape[0]
    #     if data['gt_objects'].shape[1] > max_det:
    #         max_gt = data['gt_objects'].shape[1]
    
    #list of dictionaries to dictionary of lists:
    batch_dict = {k: [dic[k] for dic in batch] for k in batch[0]}
    # print(batch_dict['X'])
    # print("batch_dict['X'][0].shape:", batch_dict['X'][0].shape)
    # print("pad_sequence(batch_dict['X'], padding_value=-99).shape:", pad_sequence(batch_dict['X'], padding_value=-99).shape)
    batch_dict['X'] = pad_sequence(batch_dict['X'], padding_value=-99).permute(1,0,2)
    batch_dict['gt_objects'] = pad_sequence(batch_dict['gt_objects'], padding_value=-99).permute(1,0,2)
    # print(batch_dict['labels'])
    if verbose:
        print()
        print('labels before padding')
        for cur_labels in batch_dict['labels']:
            print("cur_labels:", cur_labels)
            print("cur_labels.shape:", cur_labels.shape)
    # label_pad_val = torch.max()
    batch_dict['labels'] = pad_sequence(batch_dict['labels'], padding_value=-99).permute(1,0)
    # print("batch_dict['labels']")
    # print(batch_dict['labels'])
    # batch_dict['gt_count_per_img'] = torch.max(batch_dict['labels'], dim=1)[0] + 1
    batch_dict['gt_count_per_img'] = torch.tensor(batch_dict['gt_obj_count'])
    # print("gt_counts:")
    # print(torch.max(batch_dict['labels'], dim=1)[0] + 1)
    # print(batch_dict['gt_obj_count'])
    # print()
    if verbose:
        print("gt_count_per_img:")
        print(batch_dict['gt_count_per_img'])
    max_batch_gt_count = torch.max(batch_dict['labels']) + 1
    #set pad value to the the maximum label for one hot operation
    batch_dict['onehot_labels'] = torch.clone(batch_dict['labels'])
    batch_dict['onehot_labels'][torch.where(batch_dict['onehot_labels'] == -99)] = max_batch_gt_count - 1

    if verbose:   
        print()
        print('onehot_labels after padding')
        print(batch_dict['onehot_labels'])


    # print("max_batch_gt_count:", max_batch_gt_count)
    # print("batch_dict['onehot_labels'].type():", batch_dict['onehot_labels'].type())
    # print("batch_dict['onehot_labels']")
    # print(batch_dict['onehot_labels'])
    # print("max_batch_gt_count:", max_batch_gt_count)
    # print("batch_dict['onehot_labels']:", batch_dict['onehot_labels'])
    # print("max_batch_gt_count.type():", max_batch_gt_count.type())
    # print("batch_dict['onehot_labels'].type():", batch_dict['onehot_labels'].type())
    batch_dict['onehot_labels'] = F.one_hot(batch_dict['onehot_labels'].long(), max_batch_gt_count.int())
    if verbose:
        print()
        print('onehot_labels after onehot')
        print(batch_dict['onehot_labels'])
    batch_dict['FP_labels'] = pad_sequence(batch_dict['FP_labels'], padding_value=-99).permute(1,0)
    batch_dict['det_mask'] = torch.zeros_like(batch_dict['labels'], dtype=torch.bool)
    batch_dict['det_mask'][torch.where(batch_dict['labels'] == -99)] = 1
    if batch_dict['gt_objects'].shape[1] > 0:
        batch_dict['gt_mask'] = torch.zeros(batch_dict['gt_objects'].shape[:2], dtype=torch.bool)
        batch_dict['gt_mask'][torch.where(batch_dict['gt_objects'][:,:,0] == -99)] = 1
    else:
        batch_dict['gt_mask'] = gt_objects = torch.zeros((0))

    if verbose:
        print()
        print('det_mask')
        print(batch_dict['det_mask'])
    # exit(0)       
    # print("batch_dict:")
    # print(batch_dict)
    # sleep(temps)
    # print('@'*80)
    # print(batch)
    # print('!!!')
    # print(*batch)
    # sleep(temp)


    return batch_dict


# def create_clustering_datafile(infilename='./jdk_data/bboxes_with_assoc.json',
#                                outfilename='./jdk_data/TP_bboxes_with_assoc.json'):
#     '''
#     remove all FP detections for training a clustering network
#     '''
if __name__ == '__main__':
    old_data_file = '/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp9.json'
    new_data_file = '/home/lyft/software/perceptionresearch/object_detection/mmdetection/jdk_data/bboxes_with_assoc_train2017_start100000_tiny100_GaussianCRPS_IOUp9_FPremovalNetProcessed.json'
    create_FPremovalNet_processed_dataset(old_data_file=old_data_file, new_data_file=new_data_file)