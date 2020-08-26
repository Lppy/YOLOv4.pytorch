import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..')
add_path(lib_path)


import torch
from yolov4.model import Yolov4


model = Yolov4(inference=True)
pretrained_dict = torch.load('data/pretrained_model/yolov4.conv.137.pth')
state_dict = model.state_dict()
new_state_dict = {}

backbone_list = [ 'down1', 'down2', 'down3', 'down4', 'down5' ]

for key, value in pretrained_dict.items():
	is_backbone = False
	for item in backbone_list:
		if key.find(item)!=-1:
			is_backbone = True
	if not is_backbone:
		continue
	if key in state_dict:
		new_state_dict[key] = value
		continue
	items = key.split('.')
	if key.find('resblock')!=-1:
		items[2] = 'res_blocks'
		items.insert(4, 'res')
	else:
		assert False, key
	new_key = ".".join(items)
	new_state_dict[new_key] = value

torch.save(new_state_dict, "new_CSPDarknet53.pth")

import json
json.dump(list(new_state_dict.keys()), open("key.json","w"))