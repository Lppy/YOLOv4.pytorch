import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

from cfg import Cfg
from easydict import EasyDict as edict
import torch
from yolov4.model import Yolov4
from yolov4.eval import YoloV4_eval, Yolo_eval_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Evaluate yolov4 model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-r', '--resume', dest='resume', action='store_true')
    parser.add_argument('-nf', '--nonfast', dest='nonfast', action='store_true')

    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='-1',
                        help='GPU')
    parser.add_argument('-w', '--weight_file', dest='weight_file', type=str, default=None,
                        help='model weight file')

    parser.add_argument('-m', '--mode', dest='mode', type=str, default='coco',
                        help='evaluate mode')
    parser.add_argument('-dir', '--images_dir', dest='images_dir', type=str, default=None,
                        help='images dir')
    parser.add_argument('-gt', '--ground_truth', dest='gt_annotations_path', type=str, default=None,
                        help='ground truth annotations file')
    
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)

def convert_anns(ann):
    cat = ann['category_id']
    bbox = ann['bbox']
    x, y, w, h = bbox
    x1, y1 = x - w / 2, y - h / 2
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    ann['category_id'] = cat
    ann['bbox'] = [x1, y1, w, h]
    return ann

torch.multiprocessing.set_sharing_strategy('file_system')
def test(eval_model, dataset):
    # output bbox is [x_center, y_center, w, h]
    predictions = []
    total = [0, 0, 0]
    num_images = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, data in enumerate(dataloader):
        # if i==100: break
        img, image_id, image_height, image_width = data

        start = time.time()
        (boxes, scores, classes), (inference_time, nms_time) = eval_model.detect_one_image(img)
        finish = time.time()

        boxes[:, 0] *= image_width.item()
        boxes[:, 2] *= image_width.item()
        boxes[:, 1] *= image_height.item()
        boxes[:, 3] *= image_height.item()
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        if type(image_id) is torch.Tensor:
            image_id = int(image_id[0])
        else:
            image_id = image_id[0]
        predictions.append([boxes, scores, classes, image_id])

        total[0] += inference_time
        total[1] += nms_time
        total[2] += finish - start

        sys.stdout.write('im_detect: {:d}/{:d} #bbox:{:d} inf:{:.3f}s nms:{:.3f}s   \r' \
            .format(i + 1, num_images, len(boxes), inference_time, nms_time))
        sys.stdout.flush()

    print('\navg_inference: {:.3f}s, avg_nms: {:.3f}s, avg_per_image: {:.3f}s, FPS:{:.2f}' \
        .format(total[0]/num_images, total[1]/num_images, total[2]/num_images, num_images/total[2]))
    
    result = [{
        'category_id': int(item[2][idx]),
        'image_id': item[3],
        'bbox': item[0][idx].tolist() if type(item[0][idx]) is not list else item[0][idx],
        'score': float(item[1][idx])
    } for (i,item) in enumerate(predictions) for idx in range(len(item[0]))]

    return result

def test_coco(model, annotations, cfg, resFile):
    use_cuda = 1 if torch.cuda.is_available() else 0

    eval_model = YoloV4_eval(model, cfg.num_classes, nms_thresh=0.6, conf_thresh=0.000001, use_cuda=use_cuda, fast=not cfg.nonfast)
    dataset = Yolo_eval_dataset(cfg.images_dir, annotations["images"], mode="coco", target_size=[cfg.image_size, cfg.image_size])

    if not cfg.resume:
        result = test(eval_model, dataset)
        sorted_annotations = result
        sorted_annotations = list(map(convert_anns, sorted_annotations))
        with open(resFile, 'w') as f:
            json.dump(sorted_annotations, f)
    else:
        with open(resFile) as f:
            sorted_annotations = json.load(f)

    if "annotations" not in annotations:
        print("Test-dev mode.")
        return

    annType = "bbox"
    cocoGt = COCO(cfg.gt_annotations_path)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def test_images(model, cfg, confidence=0.6, vis=True):
    use_cuda = 1 if torch.cuda.is_available() else 0

    images = os.listdir(cfg.images_dir)
    eval_model = YoloV4_eval(model, cfg.num_classes, nms_thresh=0.6, conf_thresh=0.000001, use_cuda=use_cuda, fast=not cfg.nonfast)
    dataset = Yolo_eval_dataset(cfg.images_dir, images, mode='image', target_size=[cfg.image_size, cfg.image_size])

    result = test(eval_model, dataset)
    with open("images_results.json", 'w') as f:
        json.dump(result, f)
    if vis:
        try:
            os.mkdir("output")
            print('Created image output directory.')
        except OSError:
            pass
        all_classes = open(cfg.name_classes).readlines()
        all_classes = [item.strip() for item in all_classes]
        annotations = convert_det(result, confidence=confidence, topn=100)
        for img, dets in annotations.items():
            img_path = os.path.join(cfg.images_dir, img)
            img_in = cv2.imread(img_path)
            if len(img_in.shape) == 2:
                img_in = img_in[:,:,np.newaxis]
                img_in = np.concatenate((img_in,img_in,img_in), axis=2)

            classes = [all_classes[ann["category_id"]] for ann in dets]
            bboxes = [ann["bbox"] for ann in dets]
            confidences = [ann["score"] for ann in dets]
            img_out = vis_detections(img_in, classes, bboxes, confidences)
            res_path = os.path.join('output', img)
            cv2.imwrite(res_path, img_out)

def convert_det(data, confidence=0.01, topn=100):
    temp = {}
    for item in data:
        confi = item['score']
        img = item['image_id']
        if confi < confidence:
            continue
        if img not in temp:
            temp[img] = []
        bbox = item['bbox']
        bbox = bbox[0]-bbox[2]/2, bbox[1]-bbox[3]/2, bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        temp[img].append({
            "category_id": item['category_id'], 
            "bbox": bbox, 
            "score":confi
        })
    for key in temp:
        temp[key].sort(key=lambda x:x["score"], reverse=True)
    for key, value in temp.items():
        if len(value) > topn:
            temp[key] = value[:topn]
    return temp

def vis_detections(im, class_name, dets, score):
    for idx, bbox in enumerate(dets):
        bbox = [int(np.round(x)) for x in bbox]
        color = (0x6f, 0xff, 0xd2)#(0x63, 0x76, 0xdc)
        width = len(class_name[idx])*15+100
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+width, bbox[1]+35), color, -1)
        cv2.putText(im, '%s: %.1f' % (class_name[idx], score[idx]*100), (bbox[0], bbox[1] + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), thickness=2)
    return im



if __name__ == "__main__":
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        model = Yolov4(cfg.anchors, n_classes=cfg.num_classes, image_size=cfg.image_size, inference=True)
        pretrained_dict = torch.load(cfg.weight_file)
        model.load_state_dict(pretrained_dict)

        model.eval()
        model.to(device=device)

        if cfg.mode == 'coco':
            annotations_file_path = cfg.gt_annotations_path
            with open(annotations_file_path) as annotations_file:
                try:
                    annotations = json.load(annotations_file)
                except:
                    assert False, "Annotations file does not exist!"
            test_coco(model=model,
                 annotations=annotations,
                 cfg=cfg, 
                 resFile=cfg.weight_file+'_pred.json')
        elif cfg.mode == 'image':
            test_images(model=model, cfg=cfg)
        elif cfg.mode == 'camera': pass
        else:
            assert False, cfg.mode + " evaluate is not implemented!"