import shutil

import torch.backends.cudnn as cudnn
import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms

from models import classification_model, CombinedActivations
from losses import regularized_loss, fine_regularized_loss



cudnn.benchmark = True


class Trainer(object):
    def __init__(self, cfg, device,exp_name):
        """
        This method initializes settings
        :param settings: application settings
        """

        self.cfg = cfg
        self.images_dir = cfg.IMAGES_DIR
        self.device = device
        self.ckpt_dir = os.path.join(self.cfg.RES_DIR,exp_name)
        os.makedirs(self.ckpt_dir,exist_ok=True)
        shutil.copyfile('config.yml',os.path.join(self.ckpt_dir,'config.yml'))

        wandb.init(
            project="ct",
            id=wandb.util.generate_id(),
            settings=wandb.Settings(start_method="fork"),
            name=exp_name,
        )
        t = transforms.Compose([transforms.ToTensor(),transforms.Resize(299),transforms.CenterCrop(256),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_ds = ImageFolder(root=os.path.join(self.images_dir,'train'),transform=t)
        val_ds = ImageFolder(root=os.path.join(self.images_dir,'test'),transform=t)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        if cfg.AVG:
            self.net = CombinedActivations(
                encoder_name=cfg.ENCODER_NAME,
                encoder_depth=cfg.ENCODER_DEPTH,
                encoder_weights=cfg.ENCODER_WEIGHTS,
                in_channels=cfg.N_CHANNELS,
                classes=cfg.NUM_CLASSES,cfg=cfg
            )
        else:
            self.net = classification_model(
                encoder_name=cfg.ENCODER_NAME,
                encoder_depth=cfg.ENCODER_DEPTH,
                encoder_weights=cfg.ENCODER_WEIGHTS,
                in_channels=cfg.N_CHANNELS,
                classes=cfg.NUM_CLASSES,
            )
        self.net.to(self.device)
        self.trained_encoder = None
        if cfg.USE_REGULARIZED_LOSS:
            self.trained_encoder = classification_model(
                encoder_name=cfg.ENCODER_NAME,
                encoder_depth=cfg.ENCODER_DEPTH,
                encoder_weights=cfg.ENCODER_WEIGHTS,
                in_channels=cfg.N_CHANNELS,
                classes=cfg.NUM_CLASSES,
            ).get_encoder(
                name=cfg.ENCODER_NAME,
                in_channels=cfg.N_CHANNELS,
                depth=cfg.ENCODER_DEPTH,
                weights=cfg.ENCODER_WEIGHTS
            )
            self.trained_encoder.to(self.device)
        weights = cfg.WEIGHTS
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,device=self.device))

        if cfg.SOFT_FREEZE:
            cfg.LR *= 100
            base_blk = []
            blk1 = []
            blk2 = []
            blk3 = []
            blk4 = []
            cls_head = []
            for name,p in self.net.named_parameters():
                if name in ['encoder.features.conv0.weight','encoder.features.norm0.weight','encoder.features.norm0.bias' ]:
                    base_blk.append(p)
                elif 'denseblock1' in name or 'transition1' in name:
                    blk1.append(p)
                elif 'denseblock2' in name or 'transition2' in name:
                    blk2.append(p)
                elif 'denseblock3' in name or 'transition3' in name:
                    blk3.append(p)
                elif 'denseblock4' in name:
                    blk4.append(p)
                elif 'classification_head' in name or name in ['encoder.features.norm5.weight','encoder.features.norm5.bias']:
                    cls_head.append(p)
                else:
                    print(name)
                    raise Exception()

            params_list = [{'params':cls_head,'lr':cfg.LR},{'params':blk4,'lr':cfg.LR},
                           {'params':blk3,'lr':cfg.LR},{'params':blk2,'lr':cfg.LR / 10},
                           {'params':blk1,'lr':cfg.LR/100},{'params':base_blk,'lr':cfg.LR / 1000}]
            self.optimizer = torch.optim.Adam(params_list)
        elif cfg.AVG:
            self.optimizer = torch.optim.Adam(self.net.parameters_to_grad(), lr=cfg.LR)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.LR_DECAY_STEP_SIZE,
                                                    gamma=cfg.GAMMA_DECAY)
        self.step=0



    def run_epoch(self,dl,train_or_val):
        bar = tqdm(enumerate(dl), total=len(dl))
        losses = []
        accs = []

        for i, (inputs,labels) in bar:
            if train_or_val == 'train':
                self.net.train()  # Set model to training mode
                self.optimizer.zero_grad()
            else:
                self.net.eval()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)


            with torch.set_grad_enabled(train_or_val == 'train'):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)
                if self.cfg.USE_REGULARIZED_LOSS:
                    if self.cfg.FINE_REGULARIZED_LOSS:
                        loss = fine_regularized_loss(self.net.encoder, self.trained_encoder, loss)
                    else:
                        loss = regularized_loss(self.net.encoder, self.trained_encoder, loss, alpha=self.cfg.ALPHA)
                if train_or_val == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.step+=1

            losses.append(loss.item())

            accs.append(torch.sum(preds == labels.data).item() / batch_size)

            if i % 10 == 0 or i == len(dl) -1:
                bar.set_description(f'{train_or_val} loss: {np.mean(losses)} {train_or_val} accuracy: {np.mean(accs)} iter: {i}')
                logs = {
                    f'{train_or_val} loss': float(np.mean(losses)),
                    f'{train_or_val} accuracy': float(np.mean(accs)),
                }
                wandb.log(logs,step=self.step)

        if train_or_val == 'train':
            self.scheduler.step()
        return float(np.mean(accs))


    def train(self):
        best_acc = 0.0
        num_epochs = self.cfg.NUM_EPOCHS
        self.step = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.run_epoch(self.train_loader,'train')
            epoch_acc_val = self.run_epoch(self.val_loader,'val')
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val

                torch.save({'unet': self.net.state_dict(), 'encoder': self.net.encoder.state_dict()},
                           os.path.join(self.ckpt_dir, 'unet_best_val_dice.pt'))
                torch.save({'optimizer': self.optimizer.state_dict()}, os.path.join(self.ckpt_dir, 'optimizer.pt'))


