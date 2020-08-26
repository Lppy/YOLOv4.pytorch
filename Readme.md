# YOLO v4

A concise, fast and easy to modify PyTorch implementation for YOLOv4.



## Environment

```
python 3
pytorch >= 1.2.0
```

Except PyTorch, you should also run:

```
pip install -r requirements.txt
```



## Pre-trained model

YOLOv4 weight [yolov4.pth](https://www.dropbox.com/s/0ii2ijnfunyhdi7/yolov4.pth?dl=0) is converted from [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)'s pytorch YOLOv4  [weight](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ). The latter one  is converted from the [official](https://github.com/AlexeyAB/darknet) YOLOv4 [weight](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights).



The pretrained 137 layers weight [yolov4.conv.137.pth](https://www.dropbox.com/s/ogxid7oliw591tq/yolov4.conv.137.pth?dl=0) is converted from Tianxiaomo/pytorch-YOLOv4 which is converted from the official weight.



## Experiment

The results of the YOLOv4 weight converted from the official YOLOv4 repo, evaluated on MSCOCO-2017-test-dev set:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.653
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.477
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.556
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.596
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
```



The results of the YOLOv4 weight trained by this repo with the pre-trained 137 layers weight and frozen backbone, evaluated on MSCOCO-2017-test-dev set after 20 epochs training on MSCOCO 2014 train+val-minival5k:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.638
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.695
```

After 40 epochs training:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.641
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.505
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.564
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.607
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
```



## Train

Firstly, prepare the annotations file data/train.txt:

```
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```

This file for MSCOCO 2014 can be converted by scripts/coco_annotation.py. It needs a soft link to the dataset:

```
cd data
ln -s /path/to/MSCOCO coco
```



Train the model:

```
python train.py \
-g 0,1 \
-pretrained data/yolov4.conv.137.pth \
-r checkpoints/Yolov4_epochXXX.pth
```

arguments:

```
-g 0,1 # gpus
-pretrained data/yolov4.conv.137.pth # path of the pretrained weight
-r checkpoints/Yolov4_epochXXX.pth # optional, use if you wanted to resume from one checkpoint
```

Modify the cfg.py for a different batch size, # of epochs, and etc.



## Evaluate

Detect and evaluate 5000 mini-validation COCO2014 images:

```
python evaluate.py \
-w checkpoints/Yolov4_epoch20.pth \
-g 0 \
-gt data/5k.json \
-dir data/coco/images/val2014/
```

arguments:

```
-w checkpoints/Yolov4_epoch20.pth # weight to evaluate
-g 0 # gpu
-gt data/5k.json # annotation file
-dir data/coco/images/val2014/ # path of images
```

Note, fastnms is used as default NMS method, the FPS on TITAN V with fastnms and image size 608 is 22. However, the AP will be damaged by 1 point.




Detect 20288 test-dev MSCOCO-2017 images:

```
python evaluate.py \
-w checkpoints/Yolov4_epoch20.pth \
-g 0 \
-gt data/image_info_test-dev2017.json \
-dir data/test2017/ \
-nf
```

arguments:

```
-w checkpoints/Yolov4_epoch20.pth # weight to detect
-g 0 # gpu
-gt data/image_info_test-dev2017.json # annotation file with only images' info
-dir data/test2017/ # path of images
-nf # won't use the fastnms if specified
```



Detect all images in a folder:

```
python evaluate.py \
-m image \
-w checkpoints/Yolov4_epoch20.pth \
-g 0 \
-dir ../imgs/ 
```

arguments:

```
-m image # image mode
-w checkpoints/Yolov4_epoch29.pth # weight to detect
-g 0 # gpu
-dir ../imgs/ # path of images
```



## Reference & Thanks

I take a lot from these repositories:

https://github.com/jwyang/faster-rcnn.pytorch

https://github.com/qqwweee/keras-yolo3

https://github.com/Tianxiaomo/pytorch-YOLOv4

https://github.com/miemie2013/Keras-YOLOv4