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
pretrained_dict = torch.load('data/pretrained_model/yolov4.pth')
# pretrained_dict = torch.load('data/pretrained_model/yolov4.conv.137.pth')
state_dict = model.state_dict()
new_state_dict = {}

for key, value in pretrained_dict.items():
	if key in state_dict:
		new_state_dict[key] = value
		continue
	items = key.split('.')
	if key.find('resblock')!=-1:
		items[2] = 'res_blocks'
		items.insert(4, 'res')
	elif key.find('neek')!=-1:
		items[0] = 'neck'
		layer = int(items[1][4:])
		if layer>=1 and layer <=3:
			items.insert(1, 'CBAx3_1')
		elif layer>=4 and layer <=6:
			items.insert(1, 'CBAx3_2')
		elif layer>=7 and layer <=8:
			items.insert(1, 'upConv_1')
			items[2] = 'conv%d'%(layer-6)
		elif layer>=9 and layer <=13:
			items.insert(1, 'CBAx5_1')
		elif layer>=14 and layer <=15:
			items.insert(1, 'upConv_2')
			items[2] = 'conv%d'%(layer-13)
		elif layer>=16 and layer <=20:
			items.insert(1, 'CBAx5_2')
		else:
			assert False, key
	elif key.find('head')!=-1:
		layer = int(items[1][4:])
		if layer>=1 and layer <=2:
			items.insert(1, 'out1')
		elif layer>=9 and layer <=10:
			items.insert(1, 'out2')
		elif layer>=17 and layer <=18:
			items.insert(1, 'out3')
		elif layer>=4 and layer <=8:
			items.insert(1, 'CBAx5_1')
		elif layer>=12 and layer <=16:
			items.insert(1, 'CBAx5_2')
		else:
			assert False, key
	else:
		assert False, key
	new_key = ".".join(items)
	new_state_dict[new_key] = value

state_dict.update(new_state_dict)
torch.save(state_dict, "new_yolov4.pth")
# torch.save(new_state_dict, "new_yolov4.conv.137.pth")