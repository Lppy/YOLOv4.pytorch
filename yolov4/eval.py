import time
import numpy as np
import torch
import torchvision.ops as ops
import torch.nn.functional as F

import os
from PIL import Image
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
import torch.nn as nn

class Yolo_eval_dataset(dataset.Dataset):
    def __init__(self, image_dir, images, mode='coco', target_size=[608, 608]):
        super(Yolo_eval_dataset, self).__init__()
        self.image_dir = image_dir
        self.images = images
        self.mode = mode
        self.target_size = target_size
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        if self.mode == 'coco':
            image_name = image_info["file_name"]
            image_id = image_info["id"]
            image_height = image_info["height"]
            image_width = image_info["width"]
            img = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        elif self.mode == 'image':
            image_name = image_info
            image_id = image_info[:image_info.rfind('.')]
            img = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
            image_height, image_width = img.height, img.width

        img = img.resize(self.target_size)
        img = self.toTensor(img)
        assert len(img.shape) == 3
        return img, image_id, image_height, image_width


class Yolo_inference(nn.Module):
    def __init__(self, anchor_mask, num_classes, anchors, stride, image_size):
        super(Yolo_inference, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.stride = stride
        self.image_size = image_size
        self.scale_x_y = 1.05

        self.num_anchors = len(self.anchor_mask)
        self.anchors = torch.zeros([1, self.num_anchors, 1, 1, 2])
        for i, m in enumerate(self.anchor_mask):
            self.anchors[0, i, 0, 0, 0] = anchors[m][0] / self.stride
            self.anchors[0, i, 0, 0, 1] = anchors[m][1] / self.stride

        H, W = self.image_size // self.stride, self.image_size // self.stride
        shift_x = torch.arange(0, W)
        shift_y = torch.arange(0, H)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        self.shifts = torch.cat([shift_x.view(1, 1, H, W, 1), shift_y.view(1, 1, H, W, 1)], dim=-1)

    def forward(self, feat):
        B, _, H, W = feat.shape
        device = feat.get_device() if feat.is_cuda else 'cpu'

        feat = feat.view(B, self.num_anchors, self.num_classes+5, H*W).permute(0, 1, 3, 2).contiguous()
        feat = feat.view(B, self.num_anchors*H*W, self.num_classes+5)

        confi_score = feat[..., 4:5]
        class_score = feat[..., 5:]
        confi_score = torch.sigmoid(confi_score)
        # class_score = F.softmax(class_score, dim=2)
        class_score = torch.sigmoid(class_score)
        confidences = confi_score * class_score

        xy = feat[..., :2].view(B, self.num_anchors, H, W, 2)
        wh = feat[..., 2:4].view(B, self.num_anchors, H, W, 2)
        xy = torch.sigmoid(xy) * self.scale_x_y - (self.scale_x_y - 1) / 2.0
        wh = torch.exp(wh) # e^x * anchor / (grid * stride)

        shifts = self.shifts.to(device=device)
        anchors = self.anchors.to(device=device)

        xy += shifts
        wh *= anchors
        xy[..., 0] /= W
        wh[..., 0] /= W
        xy[..., 1] /= H
        wh[..., 1] /= H
        xy = xy.view(B, self.num_anchors*H*W, 2)
        wh = wh.view(B, self.num_anchors*H*W, 2)
        boxes = torch.cat([xy, wh], dim=2).view(B, self.num_anchors*W*H, 1, 4).repeat([1, 1, self.num_classes, 1])

        return boxes, confidences

def batch_iou_xywh(boxes_1, boxes_2, diou=False):
    assert boxes_2.size(0) == boxes_2.size(0)
    C = boxes_1.size(0)
    A = boxes_1.size(1)
    B = boxes_2.size(1)
    boxes_1 = boxes_1.view(C, A, 1, 4).repeat(1, 1, B, 1)
    boxes_2 = boxes_2.view(C, 1, B, 4).repeat(1, A, 1, 1)

    boxes_a = torch.cat([boxes_1[..., :2] - boxes_1[..., 2:]/2.0, boxes_1[..., :2] + boxes_1[..., 2:]/2.0], dim=-1)
    boxes_b = torch.cat([boxes_2[..., :2] - boxes_2[..., 2:]/2.0, boxes_2[..., :2] + boxes_2[..., 2:]/2.0], dim=-1)

    boxes_a = torch.cat([torch.min(boxes_a[..., :2], boxes_a[..., 2:]), torch.max(boxes_a[..., :2], boxes_a[..., 2:])], dim=-1)
    boxes_b = torch.cat([torch.min(boxes_b[..., :2], boxes_b[..., 2:]), torch.max(boxes_b[..., :2], boxes_b[..., 2:])], dim=-1)

    i_xymin = torch.max(boxes_a[..., :2], boxes_b[..., :2])
    i_xymax = torch.min(boxes_a[..., 2:], boxes_b[..., 2:])
    i_wh = (i_xymax - i_xymin).clamp(min=0.0)

    w_a, h_a = boxes_1[..., 2], boxes_1[..., 3]
    w_b, h_b = boxes_2[..., 2], boxes_2[..., 3]

    ia = i_wh[..., 0] * i_wh[..., 1]
    ua = w_a * h_a + w_b * h_b - ia
    iou = ia / (ua + 1e-9)

    if not diou:
        return iou

    u_xymin = torch.min(boxes_a[..., :2], boxes_b[..., :2])
    u_xymax = torch.max(boxes_a[..., 2:], boxes_b[..., 2:])
    u_wh = (u_xymax - u_xymin).clamp(min=0.0)

    c2 = torch.pow(u_wh[..., 0], 2) + torch.pow(u_wh[..., 1], 2)
    p2 = torch.pow(boxes_1[..., 0] - boxes_2[..., 0], 2) + torch.pow(boxes_1[..., 1] - boxes_2[..., 1], 2)

    return iou - p2 / c2

'''
def batch_iou(boxes_a, boxes_b):
    c = boxes_a.size(0)
    A = boxes_a.size(1)
    B = boxes_b.size(1)
    boxes_a = boxes_a.view(c, A, 1, 4).repeat(1, 1, B, 1)
    boxes_b = boxes_b.view(c, 1, B, 4).repeat(1, A, 1, 1)

    i_xymin = torch.max(boxes_a[:,:,:,:2], boxes_b[:,:,:,:2])
    i_xymax = torch.min(boxes_a[:,:,:,2:], boxes_b[:,:,:,2:])
    i_wh = (i_xymax - i_xymin).clamp(min=0.0)

    w_a, h_a = boxes_a[:,:,:,2] - boxes_a[:,:,:,0], boxes_a[:,:,:,3] - boxes_a[:,:,:,1]
    w_b, h_b = boxes_b[:,:,:,2] - boxes_b[:,:,:,0], boxes_b[:,:,:,3] - boxes_b[:,:,:,1]

    ia = i_wh[:,:,:,0] * i_wh[:,:,:,1]
    ua = w_a * h_a + w_b * h_b - ia
    overlap = ia / (ua + 1e-9)
    return overlap
'''

def fast_nms(boxes, scores, k, iou_threshold):
    # Dets x #cls x 4
    # Dets x #cls
    n_cls = scores.size(1)
    sorted_index = torch.sort(scores, dim=0, descending=True)[1]
    new_boxes = scores.new_zeros([n_cls, k, 4])
    new_scores = scores.new_zeros([n_cls, k])
    classes = scores.new_zeros([n_cls, k])
    for c in range(n_cls):
        keep_index = sorted_index[:k, c]
        new_boxes[c] = boxes[:, c][keep_index]
        new_scores[c] = scores[:, c][keep_index]
        classes[c] = torch.ones_like(new_scores[c], dtype=torch.int) * c

    iou = batch_iou_xywh(new_boxes, new_boxes, diou=True)
    iou = iou.triu_(diagonal=1)
    iou_max = torch.max(iou, dim=1)[0]

    keep = iou_max <= iou_threshold
    boxes = new_boxes[keep]
    scores = new_scores[keep]
    classes = classes[keep]
    return boxes, scores, classes


class YoloV4_eval:
    def __init__(self, model, n_classes, nms_thresh, conf_thresh, max_per_image=100, use_cuda=True, fast=True):
        self.model = model
        self.n_classes = n_classes
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.max_per_image = max_per_image
        self.use_cuda = use_cuda
        self.fast = fast
        if use_cuda:
            self.model.cuda()
    
    def detect_one_image(self, img):
        self.model.eval()
        assert len(img.shape) == 4 and img.shape[0] == 1
        if self.use_cuda:
            img = img.cuda()
 
        t1 = time.time()
        boxes_and_confs = self.model(img)
        # 3 x 2 
        # B x #inst x #cls x 4
        # B x #inst x #cls
        t2 = time.time()

        boxes = []
        box_scores = []
        for l in range(len(boxes_and_confs)):
            _boxes, _box_scores = boxes_and_confs[l]
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = torch.cat(boxes, dim=1)[0].detach()
        box_scores = torch.cat(box_scores, dim=1)[0].detach()

        t3 = time.time()

        # fast nms
        if self.fast:
            boxes_, scores_, classes_ = fast_nms(boxes, box_scores, self.max_per_image, self.nms_thresh)

            image_thresh = torch.sort(scores_)[0][-self.max_per_image]
            mask = (scores_ >= image_thresh) & (scores_ >= self.conf_thresh)
            boxes_ = boxes_[mask]
            scores_ = scores_[mask]
            classes_ = classes_[mask]

            t4 = time.time()
            return (boxes_, scores_, classes_), (t2 - t1, t4 - t3)

        mask = box_scores >= self.conf_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(self.n_classes):
            class_mask = mask[:, c]
            class_boxes = boxes[:, c][class_mask]
            class_box_scores = box_scores[:, c][class_mask]
            iou_boxes = class_boxes.new_zeros(class_boxes.shape)
            wh_2 = class_boxes[:, 2:]/2
            iou_boxes[:, 2:] = class_boxes[:, :2] + wh_2
            iou_boxes[:, :2] = class_boxes[:, :2] - wh_2
            keep = ops.nms(iou_boxes, class_box_scores, self.nms_thresh)
            class_boxes = class_boxes[keep]
            class_box_scores = class_box_scores[keep]
            classes = torch.ones_like(class_box_scores, dtype=torch.int) * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = torch.cat(boxes_, dim=0)
        scores_ = torch.cat(scores_, dim=0)
        classes_ = torch.cat(classes_, dim=0) 
        
        image_thresh = torch.sort(scores_)[0][-self.max_per_image]
        mask = scores_ >= image_thresh
        boxes_ = boxes_[mask]
        scores_ = scores_[mask]
        classes_ = classes_[mask]

        t4 = time.time()
        return (boxes_, scores_, classes_), (t2 - t1, t4 - t3)

        # mask = box_scores >= self.conf_thresh
        # boxes_ = []
        # scores_ = []
        # classes_ = []
        # sorted_scores = torch.sort(box_scores, dim=0)[0]
        # for c in range(self.n_classes):
        #     image_thresh = sorted_scores[:, c][-self.max_per_image]
        #     class_mask = mask[:, c] & (box_scores[:, c] >= image_thresh)
        #     class_boxes = boxes[:, c][class_mask]
        #     class_box_scores = box_scores[:, c][class_mask]
        #     classes = torch.ones_like(class_box_scores, dtype=torch.int) * c
        #     boxes_.append(class_boxes)
        #     scores_.append(class_box_scores)
        #     classes_.append(classes)
        # boxes_ = torch.cat(boxes_, dim=0)
        # scores_ = torch.cat(scores_, dim=0)
        # classes_ = torch.cat(classes_, dim=0) 

        # iou_boxes = boxes_.new_zeros(boxes_.shape)
        # wh_2 = boxes_[:, 2:]/2
        # iou_boxes[:, 2:] = boxes_[:, :2] + wh_2
        # iou_boxes[:, :2] = boxes_[:, :2] - wh_2

        # keep = ops.boxes.batched_nms(iou_boxes, scores_, classes_, self.nms_thresh)
        # # keep = ops.nms(iou_boxes, scores_, self.nms_thresh)
        # boxes_ = boxes_[keep]
        # scores_ = scores_[keep]
        # classes_ = classes_[keep]

        # if scores_.size(0) > self.max_per_image:
        #     image_thresh = torch.sort(scores_)[0][-self.max_per_image]
        #     mask = scores_ >= image_thresh
        #     boxes_ = boxes_[mask]
        #     scores_ = scores_[mask]
        #     classes_ = classes_[mask]

        # t4 = time.time()
        # return (boxes_, scores_, classes_), (t2 - t1, t4 - t3)

    def detect_batch_images(self, imgs):
        # to be finished
        self.model.eval()
        batch_size = imgs.shape[0]
        assert len(imgs.shape) == 4
        if self.use_cuda:
            imgs = imgs.cuda()
 
        t1 = time.time()
        boxes_and_confs = self.model(imgs)
        # 3 x 2 
        # B x #inst x #cls x 4
        # B x #inst x #cls
        t2 = time.time()

        boxes = []
        box_scores = []
        for l in range(len(boxes_and_confs)):
            _boxes, _box_scores = boxes_and_confs[l]
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = torch.cat(boxes, dim=1)
        box_scores = torch.cat(box_scores, dim=1)

        t3 = time.time()
        mask = box_scores >= self.conf_thresh
        boxes_all = []
        scores_all = []
        classes_all = []
        for b in range(batch_size):
            boxes_ = []
            scores_ = []
            classes_ = []
            for c in range(self.n_classes):
                class_mask = mask[b, :, c]
                class_boxes = boxes[b, :, c][class_mask]
                class_box_scores = box_scores[b, :, c][class_mask]
                keep = ops.nms(class_boxes, class_box_scores, self.nms_thresh)
                class_boxes = class_boxes[keep]
                class_box_scores = class_box_scores[keep]
                classes = torch.ones_like(class_box_scores, dtype=torch.int) * c
                boxes_.append(class_boxes)
                scores_.append(class_box_scores)
                classes_.append(classes)
            boxes_ = torch.cat(boxes_, dim=0)
            scores_ = torch.cat(scores_, dim=0)
            classes_ = torch.cat(classes_, dim=0) 
            image_thresh = torch.sort(scores_)[0][-self.max_per_image]
            mask_t = scores_ >= image_thresh
            boxes_all.append(boxes_[mask_t].unsqueeze(0).detach())
            scores_all.append(scores_[mask_t].unsqueeze(0).detach())
            classes_all.append(classes_[mask_t].unsqueeze(0).detach())
        boxes_all = torch.cat(boxes_all, dim=0)
        scores_all = torch.cat(scores_all, dim=0)
        classes_all = torch.cat(classes_all, dim=0) 
        t4 = time.time()

        return (boxes_all, scores_all, classes_all), (t2 - t1, t4 - t3)