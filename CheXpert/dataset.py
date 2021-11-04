import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CheXpertDataSet(Dataset):
    def __init__(self,images_dir, data_PATH, nnClassCount, policy, transform = None):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
                label = list(npline[idx])
                for i in range(nnClassCount):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == 'diff':
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            elif policy == 'ones':              # All U-Ones
                                label[i] = 1
                            else:
                                label[i] = 0                    # All U-Zeroes
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_names.append(os.path.join(images_dir,image_name))
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

def chexpert_ds_creator(cfg,policy='diff'):

    pathFileTrain_frt = f'{cfg.IMAGES_DIR}/train_frt.csv' ###
    pathFileTrain_lat = f'{cfg.IMAGES_DIR}/train_lat.csv'
    pathFileValid_frt = f'{cfg.IMAGES_DIR}/valid_frt.csv'
    pathFileValid_lat = f'{cfg.IMAGES_DIR}/valid_lat.csv'
    pathFileTest_frt = f'{cfg.IMAGES_DIR}/test_frt.csv'
    pathFileTest_lat = f'{cfg.IMAGES_DIR}/test_lat.csv'
    pathFileTest_agg = f'{cfg.IMAGES_DIR}/test_agg.csv'
    # Tranform data
    transformList = []
    transformList.append(transforms.Resize((320, 320))) # 320
    transformList.append(transforms.ToTensor())
    transformSequence = transforms.Compose(transformList)
    datasetTrain_frt = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/', pathFileTrain_frt, cfg.NUM_CLASSES, policy, transformSequence)
    datasetTrain_lat = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileTrain_lat, cfg.NUM_CLASSES, policy, transformSequence)
    datasetValid_frt = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileValid_frt, cfg.NUM_CLASSES, policy, transformSequence)
    datasetValid_lat = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileValid_lat, cfg.NUM_CLASSES, policy, transformSequence)
    datasetTest_frt = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileTest_frt, cfg.NUM_CLASSES, policy, transformSequence)
    datasetTest_lat = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileTest_lat, cfg.NUM_CLASSES, policy, transformSequence)
    datasetTest_agg = CheXpertDataSet('/home/dsi/shaya/chexpert_v1_small/',pathFileTest_agg, cfg.NUM_CLASSES, policy, transformSequence)
    if cfg.FRONT_OR_LAT =="FRONT":
        return datasetTrain_frt,datasetValid_frt
    return datasetTrain_lat,datasetValid_lat