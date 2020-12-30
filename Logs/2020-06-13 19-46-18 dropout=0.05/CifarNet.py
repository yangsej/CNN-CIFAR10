# -*- coding: utf-8 -*-

# (1) 내용: Training a classifier using CNN

# (1) Classification 정확도를 향상시키고,
# (2) 각 epoch마다 test set에 대한 정확도를 구하기,
# (3) 과제보고서 작성하여 제출하기
# (4) 소스코드도 함께 제출 함

# CIFAR-10 은 10개 클래스 총 6만장의 데이터셋으로 이루어져 있으며, training dataset 5만장, test dataset 1만장으로 구성되어 있습니다.
# 이번 과제는 10개 class를 분류하는 baseline model을 수정해서 학습시켜 baseline model 보다 높은 정확도를 내는 모델을 만들고 학습하는 것 입니다.
# model을 수정하거나 epoch 수를 늘리는 등 수업시간에 배운 내용을 바탕으로 정확도를 높이고, 어떤 방법으로 validation 정확도가 얼마나 증가 했는지 보고서로 작성해서 제출하세요.
# 단, data augmentation, learning rate scheduler는 사용하지 마세요.

# 제출할 파일은 총 3가지로 소스파일, model weights파일, 보고서 입니다. 보고서에 반드시 들어갈 내용으로는 어떻게 성적을 향상시켰는지, 그리고 참고 문헌입니다.
# 보고서 분량은 최대 7페이지를 넘어가지 않도록 작성해주세요.
# 이번 과제와 관련하여 궁금한 점이 있으면 과제 QnA 게시판 을 통해 질문해주세요. 이번 과제는 과제 30%중 절반인 15%를 차지합니다. 많은 비중을 차지하는 만큼 잘 작성하여 제출해 주시기 바랍니다.
"""
Created on Sun May 31 20:38:03 2020

@author: RML
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        dropout_p = 0.05
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_p),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 10),
        )

        self.fc_layer.apply(self.__init_weight__)

    def __init_weight__(self, m: nn.Module):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

def _cifarnet(pretrained=False, path=None):
    model = CifarNet()
    if pretrained:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model
