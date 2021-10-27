import numpy as np
import torch
from torch import nn


def regularized_loss(encoder_to_train, trained_encoder, loss, alpha=1):
    custom_loss = 0.0
    if alpha == 0:
        return loss

    for (m_to_train, m_trained) in zip(encoder_to_train.modules(), trained_encoder.modules()):
        if isinstance(m_to_train, nn.Conv2d):
            temp_loss = torch.sum(((m_to_train.weight.data - m_trained.weight.data) ** 2)) ** 0.5
            custom_loss += temp_loss
        elif isinstance(m_to_train, nn.BatchNorm2d):
            temp_loss = ((torch.sum(((m_to_train.weight.data - m_trained.weight.data) ** 2), 0)) ** 0.5)
            custom_loss += temp_loss

    loss = loss + alpha * custom_loss
    return loss

def fine_regularized_loss(encoder_to_train, trained_encoder, loss, max_weight=np.array(1).astype('float32'),
                          with_bn=False, beta=1):
    custom_loss = 0.0
    i = 0
    for ((name,m_to_train),(name2,m_trained) ) in zip(encoder_to_train.named_modules(), trained_encoder.named_modules()):
        assert name == name2
        if isinstance(m_to_train, nn.Conv2d) or  isinstance(m_to_train, nn.BatchNorm2d):

            if 'features.conv0' in name or 'features.norm0' in name:
                alpha = 5
            elif 'denseblock1' in name or 'transition1' in name:
                alpha = 4
            elif 'denseblock2' in name or 'transition2' in name:
                alpha = 3
            elif 'denseblock3' in name or 'transition3' in name:
                alpha = 2
            elif 'denseblock4' in name:
                alpha = 1
            elif 'classification_head' in name or name in ['features.norm5','features.norm5']:
                alpha = 0
            else:
                print(name)
                raise Exception()

            if isinstance(m_to_train, nn.Conv2d):

                temp_loss = torch.sum(((m_to_train.weight.data - m_trained.weight.data) ** 2)) ** 0.5



                custom_loss += alpha * temp_loss
            elif isinstance(m_to_train, nn.BatchNorm2d):
                if with_bn:
                    temp_loss = ((torch.sum(((m_to_train.weight.data - m_trained.weight.data) ** 2), 0)) ** 0.5)

                    custom_loss += alpha * temp_loss

    loss = loss + custom_loss
    return loss

def regularized_loss_l2(encoder_to_train, loss, alpha=1):
    custom_loss = 0.0
    for m_to_train in encoder_to_train.modules():
        if isinstance(m_to_train, nn.Conv2d):
            temp_loss = torch.sum(((m_to_train.weight.data) ** 2)) ** 0.5
            custom_loss += temp_loss
        elif isinstance(m_to_train, nn.BatchNorm2d):
            temp_loss = ((torch.sum(((m_to_train.weight.data) ** 2), 0)) ** 0.5)
            custom_loss += temp_loss

    loss = loss + alpha * custom_loss
    return loss