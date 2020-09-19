# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:56:53 2020

@author: 安辰
"""
import torch
import torch.nn as nn
import torch.functional as f
import torchvision
import torchsummary as summary
import torchvision.transforms as transforms
import os
import json

'''设置transform'''

data_transform={
    
    "train":transforms.Compose([
            
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    
        ]),
    
    "val":transforms.Compose([
            
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    
        ])
}

'''获取数据集路径'''

'''如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃'''

data_root=os.path.abspath(os.path.join(os.getcwd(), "../"))

image_root=data_root+"/DataSet/flower_data/"


'''获取数据'''

train_dataset=torchvision.datasets.ImageFolder(root=image_root+"train",transform=data_transform["train"])

val_dataset=torchvision.datasets.ImageFolder(root=image_root+"val",transform=data_transform["val"])

'''获取键值对'''
'''{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}'''

flower_list=train_dataset.class_to_idx

flower_reverse_list=dict((val,key) for key,val in flower_list.items())


'''写入json文件'''
'''json.dumps将一个Python数据结构转换为JSON'''

json_flower=json.dumps(flower_reverse_list,indent=4)

with open("flower_indices.json","w") as json_file:
    json_file.write(json_flower)
    
'''设置参数'''

batch_size=32

epoch_total=5

class_num=5

learning_rate=0.001

'''装载数据'''

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

val_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

'''搭建神经网络'''

class FlowerNet(nn.Module):
    
    def __init__(self):
        
        super(FlowerNet,self).__init__()
        
        self.classifier=nn.Sequential(
                
            nn.Linear(32*32*3, 128),
            nn.ReLU(inplace=True),
           
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, class_num)
            
        )

    def forward(self,x):
        
        x=torch.flatten(x,start_dim=1)
        
        out= self.classifier(x)
        
        return out

model=FlowerNet()

summary.summary(model, input_size=(3,32,32),batch_size=batch_size,device="cpu")

'''损失函数'''

loss_function=nn.CrossEntropyLoss()

'''优化器'''

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    
'''开始训练'''

step_total=len(train_loader)

for epoch in range(epoch_total):
    
    for step,(image,label) in enumerate(train_loader):
        
        pred=model(image)
        
        loss=loss_function(pred,label)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if (step+1) % 100 == 0:
            
            print("Epoch:[{}/{}],Step:[{}/{}],Loss:{:.4f}".format(epoch, epoch_total,step+1,step_total,loss.item()))
        
with torch.no_grad():
    
    correct=0
    
    total=0
    
    for image,label in val_loader:
        
        pred=model(image)
    
        predict=torch.max(pred,1)[1]
        
        correct += (predict == label).sum().item()
        
        total+=label.shape[0]
        
    print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))


