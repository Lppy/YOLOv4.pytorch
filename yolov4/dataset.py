import random
import sys
import numpy as np
import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

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

class Yolo_dataset(Dataset):
    def __init__(self, anchors, image_size, lable_path, cfg):
        super(Yolo_dataset, self).__init__()
        self.cfg = cfg
        self.toTensor = transforms.ToTensor()
        self.anchors = torch.tensor(anchors)
        self.anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.strides = [8, 16, 32]
        self.image_size = image_size # 608

        self.n_classes = 80
        self.n_anchors = 3

        self.iou_thresh = 0.213
        self.ignore_thresh = 0.7

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(j) for j in i.split(',')])

        self.truth = truth
        self.images = list(self.truth.keys())
        
    def __len__(self):
        return len(self.images)

    def build_target(self, labels, n_ch):
        assert (labels[:, 4] < self.n_classes).all(), 'Class id must be less than #classes!'
        B = labels.size(0)
        n_objs = (labels.sum(dim=1) > 0).sum()
        device = labels.get_device() if labels.is_cuda else 'cpu'

        gt_xy = (labels[:, 2:4] + labels[:, :2]) / 2
        gt_wh = labels[:, 2:4] - labels[:, :2]

        target = []
        for l in range(3):
            H, W = self.image_size // self.strides[l], self.image_size // self.strides[l]
            target.append(labels.new_zeros(self.n_anchors, H, W, n_ch))

        if n_objs == 0:
            return target, box_loss_scale

        anchors = self.anchors.to(device=device)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        
        wh = gt_wh[:n_objs]
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        boxes_iou = bbox_minmax_iou(boxes_min, boxes_max, anchors_min, anchors_max)
        _, best_anchor_idx = torch.max(boxes_iou, dim=1)

        for box_idx, a in enumerate(best_anchor_idx):
            for l in range(3):
                if a in self.anchor_masks[l]:
                    grid_xy = gt_xy[box_idx] / self.strides[l]
                    i = int(torch.floor(grid_xy[0]))
                    j = int(torch.floor(grid_xy[1]))
                    k = self.anchor_masks[l].index(a)
                    c = int(labels[box_idx, 4])
                    target[l][k, j, i, :2] = gt_xy[box_idx]
                    target[l][k, j, i, 2:4] = gt_wh[box_idx]
                    target[l][k, j, i, 4] = 1
                    target[l][k, j, i, 5 + c] = 1
                
                if self.iou_thresh < 1:
                    for idx, mask_i in enumerate(self.anchor_masks[l]):
                        if mask_i == a: continue
                        if boxes_iou[box_idx, mask_i] > self.iou_thresh:
                            grid_xy = gt_xy[box_idx] / self.strides[l]
                            i = int(torch.floor(grid_xy[0]))
                            j = int(torch.floor(grid_xy[1]))
                            k = idx
                            c = int(labels[box_idx, 4])
                            target[l][k, j, i, :2] = gt_xy[box_idx]
                            target[l][k, j, i, 2:4] = gt_wh[box_idx]
                            target[l][k, j, i, 4] = 1
                            target[l][k, j, i, 5 + c] = 1
        
        return target

    def __getitem__(self, index, rot=0):
        img_path_list = [self.images[index]]
        num_mix = 4 if random.randint(0, 1) else 1
        for i in range(1, num_mix):
            img_path_list.append(random.choice(self.images))

        bboxes_list = []
        for i in range(num_mix):
            img_path = img_path_list[i]
            bboxes = torch.tensor(self.truth.get(img_path), dtype=torch.float)
            img_path = os.path.join(self.cfg.dataset_dir, img_path)
            bboxes_list.append(bboxes)
            img_path_list[i] = img_path

        img_list = []
        for i in range(num_mix):
            bboxes = bboxes_list[i]
            img = Image.open(img_path_list[i]).convert('RGB')
            img = random_hsv_augmentation(img)
            img = self.toTensor(img)
            img, bboxes = random_crop_and_resize(img, bboxes, enlarge=True)
            img, bboxes = random_horizontal_flip(img, bboxes)
            img_list.append(img)
            bboxes_list[i] = bboxes

        if num_mix == 1:
            res_image, res_bboxes = img_list[0], bboxes_list[0]
        else:
            res_image, res_bboxes = mosaic_augmentation(img_list, bboxes_list)

        out_bboxes = torch.zeros([self.cfg.boxes, 5])
        out_bboxes[:min(res_bboxes.shape[0], self.cfg.boxes)] = res_bboxes[:min(res_bboxes.shape[0], self.cfg.boxes)]
        target = self.build_target(out_bboxes, 5 + self.n_classes)
        return res_image, out_bboxes, target



def filter_bboxes(bboxes, target_rect):
    x1, y1, x2, y2 = target_rect

    bboxes[:, 0] = bboxes[:, 0].clamp(x1, x2)
    bboxes[:, 2] = bboxes[:, 2].clamp(x1, x2)
    bboxes[:, 1] = bboxes[:, 1].clamp(y1, y2)
    bboxes[:, 3] = bboxes[:, 3].clamp(y1, y2)

    valid_bbox_mask = ~(
        ((bboxes[:, 1] == y1) & (bboxes[:, 3] == y1)) |
        ((bboxes[:, 0] == x1) & (bboxes[:, 2] == x1)) |
        ((bboxes[:, 1] == y2) & (bboxes[:, 3] == y2)) |
        ((bboxes[:, 0] == x2) & (bboxes[:, 2] == x2))
    )

    bboxes = bboxes[valid_bbox_mask]
    return bboxes

def mosaic_augmentation(images, bboxes, min_offset=0.2):
    # images: 4 x [C x H x W]
    # bboxes: 4 x [#bbox x 5], x1y1x2y2
    _, height, width = images[0].shape
    center_x = random.randint(int(width * min_offset), int(width * (1 - min_offset)))
    center_y = random.randint(int(height * min_offset), int(height * (1 - min_offset)))

    res_bboxes = []
    res_image = images[0]
    res_bboxes.append(filter_bboxes(bboxes[0], [0, 0, center_x, center_y]))

    res_image[:, :center_y, center_x:] = images[1][:, :center_y, center_x:]
    res_bboxes.append(filter_bboxes(bboxes[1], [center_x, 0, width, center_y]))

    res_image[:, center_y:, :center_x] = images[2][:, center_y:, :center_x]
    res_bboxes.append(filter_bboxes(bboxes[2], [0, center_y, center_x, height]))

    res_image[:, center_y:, center_x:] = images[3][:, center_y:, center_x:]
    res_bboxes.append(filter_bboxes(bboxes[3], [center_x, center_y, width, height]))

    res_bboxes = torch.cat(res_bboxes, dim=0)
    return res_image, res_bboxes

def random_crop_and_resize(image, bboxes, target_size=[608, 608], ratio=0.2, enlarge=False):
    # image: C x H x W
    # bboxes: #bbox x 5, x1y1x2y2
    _, height, width = image.shape
    r_height, r_width = int(height * ratio), int(width * ratio)

    if enlarge:
        offset_left, offset_right = random.randint(-r_width, r_width), random.randint(-r_width, r_width)
        offset_top, offset_bottom = random.randint(-r_height, r_height), random.randint(-r_height, r_height)

        src_rect = [max(0, offset_left), max(0, offset_top), 
            min(width, width-offset_right), min(height, height-offset_bottom)]

        dst_height = height-offset_bottom-offset_top
        dst_width  = width-offset_right-offset_left
        dst_rect = [max(0, -offset_left), max(0, -offset_top), max(0, -offset_left)+src_rect[2]-src_rect[0],
            max(0, -offset_top)+src_rect[3]-src_rect[1]]
    else:
        offset_left, offset_right = random.randint(0, r_width), random.randint(0, r_width)
        offset_top, offset_bottom = random.randint(0, r_height), random.randint(0, r_height)

        src_rect = [offset_left, offset_top, width-offset_right, height-offset_bottom]

        dst_height = height-offset_bottom-offset_top
        dst_width  = width-offset_right-offset_left
        dst_rect = [0, 0, dst_width, dst_height]

    image_cropped = torch.zeros([3, dst_height, dst_width])
    image_cropped[:] = image.mean(dim=(1, 2), keepdim=True)
    image_cropped[:, dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
        image[:, src_rect[1]:src_rect[3], src_rect[0]:src_rect[2]]

    image_resized = F.interpolate(image_cropped.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

    if bboxes.shape[0] == 0:
        return image_resized, bboxes
    
    bboxes[:, 0] = (bboxes[:, 0] - offset_left).clamp(0, dst_width-1)
    bboxes[:, 2] = (bboxes[:, 2] - offset_left).clamp(0, dst_width-1)
    bboxes[:, 1] = (bboxes[:, 1] - offset_top).clamp(0, dst_height-1)
    bboxes[:, 3] = (bboxes[:, 3] - offset_top).clamp(0, dst_height-1)

    valid_bbox_mask = ~(
        ((bboxes[:, 1] == dst_height-1) & (bboxes[:, 3] == dst_height-1)) |
        ((bboxes[:, 0] == dst_width-1) & (bboxes[:, 2] == dst_width-1)) |
        ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
        ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0))
    )

    if valid_bbox_mask.sum() == 0:
        return image_resized, bboxes[valid_bbox_mask]
    
    valid_bbox_index = torch.nonzero(valid_bbox_mask).view(-1).tolist()
    random.shuffle(valid_bbox_index)
    bboxes = bboxes[valid_bbox_index]
    bboxes[:, 0] *= (target_size[0] / dst_width)
    bboxes[:, 2] *= (target_size[0] / dst_width)
    bboxes[:, 1] *= (target_size[1] / dst_height)
    bboxes[:, 3] *= (target_size[1] / dst_height)
    return image_resized, bboxes

def random_horizontal_flip(image, bboxes):
    # image: C x H x W
    # bboxes: #bbox x 5, x1y1x2y2
    if random.randint(0, 1):
        width = image.shape[2]
        image = torch.flip(image, [2,])
        bboxes[:, 0], bboxes[:, 2] = width - bboxes[:, 2] - 1, width - bboxes[:, 0] - 1
    return image, bboxes

def random_hsv_augmentation(image, hue=.1, sat=1.5, exp=1.5):
    # PIL image: H x W x C [0 - 255]
    f_hue = random.uniform(-hue, hue)
    f_sat = random.uniform(1./sat, sat)
    f_exp = random.uniform(1./exp, exp)

    image = cv2.cvtColor(np.array(image).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv = cv2.split(image)
    hsv[1] *= f_sat
    hsv[2] *= f_exp
    hsv[0] += 179 * f_hue
    hsv_src = cv2.merge(hsv)
    image = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)
    image = Image.fromarray(image.astype(np.uint8), mode='RGB')

    # image = image.convert('HSV')
    # image = np.array(image)
    # image[..., 0] = image[..., 0] + int(179 * f_hue)
    # image[..., 1] = (image[..., 1] * f_sat).astype(np.uint8)
    # image[..., 2] = (image[..., 2] * f_exp).astype(np.uint8)
    # image = Image.fromarray(image, mode='HSV').convert('RGB')
    # image = np.clip(np.array(image), 0, 255)
    # image = Image.fromarray(image, mode='RGB')

    return image


def random_rotate(image, bboxes, target):
    # image: C x H x W
    # bboxes: #bbox x 5, x1y1x2y2
    if target == 0:
        pass
    elif target == 1:
        image = torch.transpose(image, 1, 2)
        new_bboxes = bboxes.clone()
        bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = new_bboxes[:, 1], new_bboxes[:, 0], new_bboxes[:, 3], new_bboxes[:, 2]
        width = image.shape[2]
        image = torch.flip(image, [2,])
        bboxes[:, 0], bboxes[:, 2] = width - bboxes[:, 2] - 1, width - bboxes[:, 0] - 1
    elif target == 2:
        height, width = image.shape[1:]
        image = torch.flip(image, [1,])
        bboxes[:, 1], bboxes[:, 3] = height - bboxes[:, 3] - 1, height - bboxes[:, 1] - 1
        image = torch.flip(image, [2,])
        bboxes[:, 0], bboxes[:, 2] = width - bboxes[:, 2] - 1, width - bboxes[:, 0] - 1
    elif target == 3:
        image = torch.transpose(image, 1, 2)
        new_bboxes = bboxes.clone()
        bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = new_bboxes[:, 1], new_bboxes[:, 0], new_bboxes[:, 3], new_bboxes[:, 2]
        height = image.shape[1]
        image = torch.flip(image, [1,])
        bboxes[:, 1], bboxes[:, 3] = height - bboxes[:, 3] - 1, height - bboxes[:, 1] - 1
    else:
        assert False
    return image, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        b = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    import os.path as osp
    import sys

    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    this_dir = osp.dirname(__file__)
    lib_path = osp.join(this_dir, '..')
    add_path(lib_path)

    from cfg import Cfg
    
    # from dataset import Yolo_dataset

    random.seed(2020)
    np.random.seed(2020)
    Cfg.dataset_dir = '../'
    dataset = Yolo_dataset(Cfg.anchors, Cfg.image_size, '../'+Cfg.train_label, Cfg)
    for i in range(4):
        out_img, out_bboxes, _, _ = dataset.__getitem__(7, i)
        out_img = np.array(out_img.mul_(255)).transpose(1, 2, 0)
        out_img = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out_img = draw_box(out_img, out_bboxes)
        cv2.imwrite("output/%d.jpg"%i, out_img)