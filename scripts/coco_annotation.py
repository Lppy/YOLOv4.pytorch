import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..')
add_path(lib_path)


import json
from collections import defaultdict

name_box_id = defaultdict(list)
id_name = dict()

f = open("data/coco/annotations/instances_train2014.json", encoding='utf-8')
train2014 = json.load(f)
f = open("data/coco/annotations/instances_val2014.json", encoding='utf-8')
val2014 = json.load(f)

f = open("data/5k.json", encoding='utf-8')
minival = json.load(f)

def map_cat_id(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat

minival_ids = {}
for item in minival['images']:
    minival_ids[item['id']] = True
print("Minival: %d."%len(minival_ids))

vaild_minival_ids = {}
annotations = train2014['annotations']
for ant in annotations:
    id = ant['image_id']
    if id in minival_ids:
        vaild_minival_ids[id] = True
        continue
    name = 'data/coco/images/train2014/COCO_train2014_%012d.jpg' % id
    cat = ant['category_id']
    cat = map_cat_id(cat)
    name_box_id[name].append([ant['bbox'], cat])

annotations = val2014['annotations']
for ant in annotations:
    id = ant['image_id']
    if id in minival_ids:
        vaild_minival_ids[id] = True
        continue
    name = 'data/coco/images/val2014/COCO_val2014_%012d.jpg' % id
    cat = ant['category_id']
    cat = map_cat_id(cat)
    name_box_id[name].append([ant['bbox'], cat])

print("Trainval: %d."%len(name_box_id))
print("Valid minival: %d"%len(vaild_minival_ids))

f = open('train.txt', 'w')
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
