# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:25:14 2020

@author: 安辰
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary as summary

'''定义参数'''
batch_size=64
total_epoch=10
lr=0.001
classes_num=10

'''获取数据集'''
train_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=True,download=False,transform=transforms.ToTensor())

test_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=False,transform=transforms.ToTensor())

'''数据装载'''
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        
        
        self.feature=nn.Sequential(
                
            nn.Conv2d(3, 96, kernel_size=3,stride=1),
            
            # 增加计算量，降低容量，可以在内存中载入更大的模型
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
                
            nn.Conv2d(96, 256, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
                         
            nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,384,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
                
        )
        
        self.classifier=nn.Sequential(
                
           nn.Dropout(p=0.5),
           nn.Linear(1024,2048),
           nn.ReLU(inplace=True),
           
           nn.Dropout(p=0.5),
           nn.Linear(2048,2048),
           nn.ReLU(inplace=True),
           
           nn.Linear(2048, classes_num),
                
        )
       
        
    def forward(self,x):
        
        out=self.feature(x)
        out=out.reshape(-1,1024)
        out=self.classifier(out)
        return out
        
model=AlexNet()

summary.summary(model, input_size=(3, 32, 32),batch_size=batch_size,device="cpu")

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

'''开始训练'''
total_step = len(train_loader)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
with torch.no_grad():
    correct=0
    total=0
    for images, labels in test_loader:
        outputs=model(images)
        predicted=torch.max(outputs,1)[1]
        total+=labels.shape[0]
        correct += (predicted == labels).sum().item()
        
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        
        
        
