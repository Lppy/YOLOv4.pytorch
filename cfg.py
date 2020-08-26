from easydict import EasyDict

Cfg = EasyDict()
Cfg.batch = 80
Cfg.TRAIN_EPOCHS = 300
Cfg.learning_rate = 0.001
Cfg.num_classes = 80
Cfg.boxes = 60 # box num


Cfg.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
Cfg.image_size = 608
Cfg.checkpoints = 'checkpoints'
Cfg.train_label = 'data/train.txt'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''