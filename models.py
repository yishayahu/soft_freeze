import os
import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.encoders.densenet import densenet_encoders

import numpy as np
encoders = {}
encoders.update(densenet_encoders)

def get_encoder(name, in_channels=3, depth=5, weights=None):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        if not os.path.exists(weights):
            settings = encoders[name]["pretrained_settings"][weights]
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))
        else:
            state_dict = torch.load(weights, map_location='cpu')
            if 'encoder' in state_dict:
                state_dict = state_dict['encoder']
            state_dict["classifier.bias"] = []
            state_dict["classifier.weight"] = []
            encoder.load_state_dict(state_dict)

    encoder.set_in_channels(in_channels)

    return encoder

class ClassificationModel(torch.nn.Module):
    def initialize(self):
        self.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        labels = self.classification_head(*features)

        return labels

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class classification_model(ClassificationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1,):
        super(classification_model, self).__init__()

        # encoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                   weights=encoder_weights)

        self.classification_head = ClassificationHead(in_channels=1024,
                                                      classes=classes)
        self.name = 'u-{}'.format(encoder_name)

    def forward(self, x):
        features = self.encoder(x)
        output = self.classification_head(features[-1])
        return output



class CombinedModel(ClassificationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1,device='cpu',cfg=None):

        super(CombinedModel, self).__init__()
        self.cfg = cfg
        # encoder
        self.encoder_base = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                        weights=encoder_weights)
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                   weights=encoder_weights)
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        self.device = device
        #ToDo: change num_features to the exact num
        self.classification_head = ClassificationHead(num_ftrs=1024*6*6,
                                                      out_channels=classes)

        if cfg.LAYER_WISE:
            self.blocks = []
            for (name,p1),(name2,p2) in zip(self.encoder_base.named_parameters(),self.encoder.named_parameters()):
                self.blocks.append([(name,p1,p2)])
        else:
            self.blocks = [[],[],[],[],[]]
            for (name,p1),(name2,p2) in zip(self.encoder_base.named_parameters(),self.encoder.named_parameters()):
                assert name == name2
                if name in ['features.conv0.weight','features.norm0.weight','features.norm0.bias' ]:
                    self.blocks[0].append((name,p1,p2))
                elif 'denseblock1' in name or 'transition1' in name:
                    self.blocks[1].append((name,p1,p2))
                elif 'denseblock2' in name or 'transition2' in name:
                    self.blocks[2].append((name,p1,p2))
                elif 'denseblock3' in name or 'transition3' in name:
                    self.blocks[3].append((name,p1,p2))
                elif 'denseblock4' in name or name in ['features.norm5.weight','features.norm5.bias' ]:
                    self.blocks[4].append((name,p1,p2))
                else:
                    print(name)
                    raise Exception()
        self.middle_layer = []
        for i in range(len(self.blocks)):
            w = torch.nn.Parameter(torch.tensor(np.random.normal()))
            self.register_parameter(name=f'w{i}', param=w)
            self.middle_layer.append(w)


    def forward(self, x):
        empty_encoder = get_encoder(self.encoder_name, in_channels=self.in_channels, depth=self.encoder_depth)
        empty_encoder.to(self.device)
        new_state_dict = empty_encoder.state_dict()
        for block, middle in zip(self.blocks,self.middle_layer):
            for layer in block:
                assert layer[0] in new_state_dict
                w1 = torch.sigmoid(middle)
                new_state_dict[layer[0]] = layer[1] * (1-w1) + layer[2] * w1
        for name,p in empty_encoder.named_parameters():
            p.requires_grad = False
            p.copy_(new_state_dict[name])
        features = empty_encoder(x)
        output = self.classification_head(*features)
        return output
    def parameters_to_grad(self):
        return [{'params':list((self.encoder.parameters())),'lr':self.cfg.LR},{'params':self.middle_layer,'lr':self.cfg.LR_FOR_MIDDLE_LAYER}]


def store_model_activations(store_list):
    def store_hook(_, __, output):
        store_list.append(output)
    return store_hook
def combine_model_activations(store_list,middle_layer_index,middle_layer):
    def add_hook(_, __, output):
        w1 = torch.sigmoid(middle_layer[middle_layer_index[0]])
        middle_layer_index[0]+=1
        output = store_list.pop(0) * (1-w1)  + (output *w1)
        return output
    return add_hook

class CombinedActivations(ClassificationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1,cfg=None):

        super(CombinedActivations, self).__init__()
        self.cfg = cfg
        # encoder
        self.encoder_base = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                        weights=encoder_weights)
        if cfg.ONLY_MIDDLE:
            encoder_weights = cfg.BEST_MODEL_PATH
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                   weights=encoder_weights)

        #ToDo: change num_features to the exact num
        self.classification_head = ClassificationHead(num_ftrs=1024*6*6,
                                                      out_channels=classes)
        self.middle_layer = []
        self.middle_layer_index = [0]
        if cfg.LAYER_WISE:
            self.store_list = []
            counter = 0
            for name, layer in self.encoder_base.named_modules():
                if 'denselayer' in name and 'conv2' in name:
                    layer.register_forward_hook(store_model_activations(self.store_list))
                    counter+=1
            for name, layer in self.encoder.named_modules():
                if 'denselayer' in name and 'conv2' in name:
                    layer.register_forward_hook(combine_model_activations(self.store_list,self.middle_layer_index,self.middle_layer))

        else:
            counter = 5
        for i in range(counter):

            loc = 0
            if not cfg.LAYER_WISE:
                loc = (i-5)/2
            w = torch.nn.Parameter(torch.tensor(np.random.normal(loc)))
            self.register_parameter(name=f'w{i}', param=w)
            self.middle_layer.append(w)


    def forward(self, x):
        self.middle_layer_index[0] = 0
        if self.cfg.LAYER_WISE:
            self.encoder_base(x)
            features = self.encoder(x)

        else:
            enc_stages = self.encoder.get_stages()
            base_enc_stages = self.encoder_base.get_stages()
            features = []
            for i in range(len(enc_stages)):
                if i ==0:
                    continue
                x1 = base_enc_stages[i](x)
                x2 = enc_stages[i](x)
                w1 = torch.sigmoid(self.middle_layer[i-1])
                if isinstance(x1, (list, tuple)):
                    x1, skip1 = x1
                    x2, skip2 = x2
                    x = x1 * (1-w1) + x2 * w1
                    features.append(skip1)
                else:
                    x = x1 * (1-w1) + x2 * w1
                    features.append(x)
        output = self.classification_head(*features)
        return output
    def parameters_to_grad(self):
        return [{'params':list((self.encoder.parameters())),'lr':self.cfg.LR},{'params':self.middle_layer,'lr':self.cfg.LR_FOR_MIDDLE_LAYER}]




