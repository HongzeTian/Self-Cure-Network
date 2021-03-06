import cv2
import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import time

from src.train import RafDataSet

#Notes:
#1: Surprise
#2: Fear
#3: Disgust
#4: Happiness
#5: Sadness
#6: Anger
#7: Neutral

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out
if __name__ == '__main__':
    # 模型存储路径
    model_save_path = "with_tqinghua_models/epoch25_acc0.8928.pth"#修改为你自己保存下来的模型文件
    img_path = r"D:\thz\data\RAF-DB\Image\aligned\test_0001_aligned.jpg"#待测试照片位置

    # ------------------------ 加载数据 --------------------------- #

    preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
    res18 = Res18Feature(pretrained = False)
    checkpoint = torch.load(model_save_path)
    res18.load_state_dict(checkpoint['model_state_dict'])
    res18.cuda()
    res18.eval()

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(r'D:\thz\data\RAF-DB', phase = 'test', transform = data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size = 2,
                                         num_workers = 2,
                                         shuffle = False,
                                         pin_memory = True)

    bingo_cnt = 0
    sample_cnt = 0
    tar = []
    pre = []
    for batch_i, (imgs, targets, _) in enumerate(val_loader):
        time1=time.time()

        _, outputs = res18(imgs.cuda())
        _, predicts = torch.max(outputs, 1)
        targets = targets.cuda()
        correct_num  = torch.eq(predicts,targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += outputs.size(0)
        time3=time.time()
        tar += targets.data.tolist()
        pre += predicts.data.tolist()
        print((time3-time1)*1000)

    matrix = np.zeros((7, 7))
    for pred, targ in zip(tar, pre):
        matrix[pred, targ] += 1

    np.savetxt('data_new.csv', matrix, delimiter=',')


