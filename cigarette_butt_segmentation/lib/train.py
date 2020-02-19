import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from lib.eval import eval_net
from lib.unet.unet_model import UNet

from torch.utils.tensorboard import SummaryWriter
from lib.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

import json



def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0003,
              save_cp=True,
              img_scale=0.5,
              step_size=1,
              gamma=0.1,
              pos_weight=70):
    
    scores = []
    path_train = "data/cig_butts/train"
    path_val = "data/cig_butts/val"
				  
    dir_train_img = 'data/cig_butts/train/images'

    dir_val_img = 'data/cig_butts/val/images'

    dir_checkpoint = 'checkpoints/'

    annotations_train = json.load(open(f"{path_train}/coco_annotations.json", "r"))
    annotations_val= json.load(open(f"{path_val}/coco_annotations.json", "r"))

    train = BasicDataset(dir_train_img, annotations_train, img_scale)
    val = BasicDataset(dir_val_img, annotations_val, img_scale)
    n_val = int(len(val))
    n_train = int(len(train))
    
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))

    for epoch in range(epochs):
        net.train()
       
       
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(train) // (2*batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    scores.append(val_score)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        scheduler.step()
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    return scores
    
