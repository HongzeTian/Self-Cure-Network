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
import torch.onnx

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
    def __init__(self, pretrained, num_classes = 4):
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
    model_save_path = "with_tqinghua_models_new_labels/epoch39_acc0.9115.pth"#修改为你自己保存下来的模型文件

    # ------------------------ 加载数据 --------------------------- #

    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    res18 = Res18Feature(pretrained = False)
    checkpoint = torch.load(model_save_path)
    res18.load_state_dict(checkpoint['model_state_dict'])
    res18.eval()

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = res18(x)
    print(res18)
    # Export the model
    torch.onnx.export(res18,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "emotion_recognition_4_labels.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=9,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                )