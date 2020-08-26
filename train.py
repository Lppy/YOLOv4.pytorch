import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader
from torch import optim

from cfg import Cfg
from yolov4.dataset import Yolo_dataset
from yolov4.model import Yolov4
from yolov4.loss import Yolo_loss


def train(model, device, config, logger, epochs=20, log_step=100):
    train_dataset = Yolo_dataset(config.anchors, config.image_size, config.train_label, config)
    n_train = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)

    logger.write(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Checkpoints:     {config.checkpoints}
        Device:          {device.type}
        Images size:     {config.image_size}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:      {config.pretrained}
    \n''')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    for i in range(config.init_epoch):
        scheduler.step()
    
    mGPUs = torch.cuda.device_count() > 1
    # criterion = Yolo_loss(config.anchors, config.image_size, n_classes=config.classes)
    # if torch.cuda.is_available():
    #     criterion.cuda()
    #     criterion = torch.nn.DataParallel(criterion)

    model.train()
    model.zero_grad()
    for epoch in range(config.init_epoch, epochs):
        epoch_loss = 0
        epoch_step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=80) as pbar:
            for i, data in enumerate(train_loader):
                epoch_step += 1
                images, bboxes, targets = data
                for i in range(3):
                    targets[i] = targets[i].to(device=device)
                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                # bboxes_pred = model(images)
                # loss, loss_xy, loss_wh, loss_obj, loss_cls = criterion(bboxes_pred, targets, bboxes)
                loss, loss_xy, loss_wh, loss_obj, loss_cls = model(images, targets, bboxes)
                if mGPUs:
                    loss = loss.mean()
                    loss_xy = loss_xy.mean()
                    loss_wh = loss_wh.mean()
                    loss_obj = loss_obj.mean()
                    loss_cls = loss_cls.mean()
                loss.backward()

                epoch_loss += loss.item()

                optimizer.step()
                model.zero_grad()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss': f'{loss.item():.3e}'})

                if epoch_step % log_step == 1:
                    logger.write("[epoch %3d][imgs %5d/%5d] loss: %.4f, lr: %.2e\n" \
                                    % (epoch+1, epoch_step*config.batch, n_train, loss.item(), scheduler.get_last_lr()[0]))
                    logger.write("\t\tloss_xy: %.4f, loss_wh: %.4f, loss_obj: %.4f, loss_cls %.4f\n" \
                                  % (loss_xy.item(), loss_wh.item(), loss_obj.item(), loss_cls.item()))
                    logger.flush()
        logger.write("[epoch %3d] loss: %.4f, lr: %.2e\n\n" \
                                    % (epoch+1, epoch_loss/epoch_step, scheduler.get_last_lr()[0]))
        scheduler.step()
        try:
            os.mkdir(config.checkpoints)
            print('Created checkpoint directory')
        except OSError:
            pass
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(config.checkpoints, f'Yolov4_epoch{epoch + 1}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(config.checkpoints, f'Yolov4_epoch{epoch + 1}.pth'))
        print(f'Checkpoint {epoch + 1} saved !')
        


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-r', '--resume', type=str, default='none',
                        help='resume file', dest='resume')
    parser.add_argument('-dir', '--data-dir', type=str, default='./',
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')

    args = vars(parser.parse_args())
    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Yolov4(cfg.anchors, backbone_weight=cfg.pretrained, n_classes=cfg.classes)
    if cfg.resume == 'none':
        cfg.init_epoch = 0
    else:
        state_dict = torch.load(cfg.resume)
        model.load_state_dict(state_dict)
        cfg.init_epoch = int(cfg.resume[cfg.resume.find('epoch')+5:cfg.resume.rfind('.')])

    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger = open("log_from_epoch%d.txt"%cfg.init_epoch, "w")
    logger.write(f'Using device {device}\n')

    try:
        train(model=model,
              device=device,
              config=cfg,
              logger=logger,
              epochs=cfg.TRAIN_EPOCHS)
        logger.close()
    except KeyboardInterrupt:
        logger.close()
        exit()