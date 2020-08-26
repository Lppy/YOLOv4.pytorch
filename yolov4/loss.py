import torch
import torch.nn as nn
from yolov4.eval import batch_iou_xywh
from torch.nn import functional as F
import numpy as np 
import math

def CIOU_loss(boxes_1, boxes_2):
    '''
    :param boxes_1: (..., 4)   pred_xywh
    :param boxes_2: (..., 4)   label_xywh
    :temp boxes_a: (..., 4)    pred_x1y1x2y2
    :temp boxes_b: (..., 4)    label_x1y1x2y2
    '''

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

    u_xymin = torch.min(boxes_a[..., :2], boxes_b[..., :2])
    u_xymax = torch.max(boxes_a[..., 2:], boxes_b[..., 2:])
    u_wh = (u_xymax - u_xymin).clamp(min=0.0)

    c2 = torch.pow(u_wh[..., 0], 2) + torch.pow(u_wh[..., 1], 2)
    p2 = torch.pow(boxes_1[..., 0] - boxes_2[..., 0], 2) + torch.pow(boxes_1[..., 1] - boxes_2[..., 1], 2)

    atan1 = torch.atan(w_a / (h_a + 1e-9))
    atan2 = torch.atan(w_b / (h_b + 1e-9))
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou_loss = 1.0 - iou + p2 / c2 + a * v
    return ciou_loss.unsqueeze(-1)
'''
def bbox_minmax_iou(boxes_a_min, boxes_a_max, boxes_b_min, boxes_b_max):
    A = boxes_a_min.size(0)
    B = boxes_b_min.size(0)
    boxes_a_min = boxes_a_min.view(A, 1, 2).repeat(1, B, 1)
    boxes_b_min = boxes_b_min.view(1, B, 2).repeat(A, 1, 1)
    boxes_a_max = boxes_a_max.view(A, 1, 2).repeat(1, B, 1)
    boxes_b_max = boxes_b_max.view(1, B, 2).repeat(A, 1, 1)

    i_xymin = torch.max(boxes_a_min, boxes_b_min)
    i_xymax = torch.min(boxes_a_max, boxes_b_max)
    i_wh = (i_xymax - i_xymin).clamp(min=0.0)

    w_a, h_a = boxes_a_max[..., 0] - boxes_a_min[..., 0], boxes_a_max[..., 1] - boxes_a_min[..., 1]
    w_b, h_b = boxes_b_max[..., 0] - boxes_b_min[..., 0], boxes_b_max[..., 1] - boxes_b_min[..., 1]

    ia = i_wh[..., 0] * i_wh[..., 1]
    ua = w_a * h_a + w_b * h_b - ia
    overlap = ia / (ua + 1e-9)
    return overlap
'''
class Yolo_loss(nn.Module):
    def __init__(self, anchors, image_size, n_classes):
        super(Yolo_loss, self).__init__()
        self.strides = [8, 16, 32]
        self.image_size = image_size # 608
        self.n_classes = n_classes
        
        self.iou_thresh = 0.213 # <1 :Using multiple anchors for a single ground truth
        self.ignore_thresh = 0.7
        self.scale_x_y = 1.05

        self.anchors = anchors #[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.n_anchors = 3
        
        self.anchors = torch.tensor(self.anchors)
        self.shifts, self.layered_anchors = [], []
        for l in range(3):
            H, W = self.image_size // self.strides[l], self.image_size // self.strides[l]

            shift_x = torch.arange(0, W)
            shift_y = torch.arange(0, H)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            self.shifts.append(torch.cat([shift_x.view(1, 1, H, W, 1), shift_y.view(1, 1, H, W, 1)], dim=-1))

            layered_anchors = torch.zeros([1, self.n_anchors, 1, 1, 2])
            for i, m in enumerate(self.anchor_masks[l]):
                layered_anchors[0, i, 0, 0, 0] = self.anchors[m, 0]
                layered_anchors[0, i, 0, 0, 1] = self.anchors[m, 1]
            self.layered_anchors.append(layered_anchors)

    def forward(self, feats, target, labels):
        loss, loss_xy_all, loss_wh_all, loss_obj_all, loss_cls_all = 0, 0, 0, 0, 0
        n_ch = 5 + self.n_classes
        B = feats[0].shape[0]
        device = labels.get_device() if labels.is_cuda else 'cpu'
        labels = labels.type(feats[0].dtype)

        for l, feat in enumerate(feats):
            _, _, H, W = feat.shape
            shifts = self.shifts[l].to(device=device)
            layered_anchors = self.layered_anchors[l].to(device=device)
            object_mask = target[l][..., 4:5] == 1
            
            feat = feat.view(B, self.n_anchors, n_ch, H*W).permute(0, 1, 3, 2).contiguous()
            feat = feat.view(B, self.n_anchors, H, W, n_ch)
            feat[..., :2] = torch.sigmoid(feat[..., :2]) * self.scale_x_y - (self.scale_x_y - 1) / 2.0

            confi_score = torch.sigmoid(feat[..., 4:5])
            class_score = torch.sigmoid(feat[..., 5:])

            pred_bboxes = torch.cat([(feat[..., :2] + shifts) * self.strides[l], torch.exp(feat[..., 2:4]) * layered_anchors], dim=-1)
            target_bboxes = target[l][..., :4]
            bbox_loss_scale = 2 - target[l][..., 2:3] * target[l][..., 3:4] / (self.image_size ** 2)

            loss_xy = object_mask * bbox_loss_scale * CIOU_loss(pred_bboxes, target_bboxes) / 2 
            loss_wh = loss_xy

            true_bboxes = labels[..., :4].clone()
            true_bboxes[..., :2] = (labels[..., 2:4] + labels[..., :2]) / 2
            true_bboxes[..., 2:4] = labels[..., 2:4] - labels[..., :2]

            iou = batch_iou_xywh(pred_bboxes.view(B, self.n_anchors*W*H, 4), true_bboxes)
            best_iou = torch.max(iou, dim=2, keepdim=True)[0].view(B, self.n_anchors, W, H, 1)
            valid_mask = (best_iou < self.ignore_thresh) | object_mask

            loss_obj = valid_mask * F.binary_cross_entropy(confi_score, target[l][..., 4:5], reduction='none')
            loss_cls = object_mask * F.binary_cross_entropy(class_score, target[l][..., 5:], reduction='none')

            loss_xy_all += loss_xy.sum() / B
            loss_wh_all += loss_wh.sum() / B
            loss_obj_all += loss_obj.sum() / B
            loss_cls_all += loss_cls.sum() / B

        loss = loss_xy_all + loss_wh_all + loss_obj_all + loss_cls_all
        return loss, loss_xy_all, loss_wh_all, loss_obj_all, loss_cls_all