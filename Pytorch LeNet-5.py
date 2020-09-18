# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:31:44 2020

@author: 安辰
"""
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

'''定义参数'''
batch_size=64
lr=0.001
num_classes=10

'''获取数据集'''
train_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=True,download=False,transform=transforms.ToTensor())

test_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=False,download=False,transform=transforms.ToTensor())

'''装载数据'''
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

'''设计LeNet_5'''
class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet, self).__init__()
        '''第一层卷积，卷积核大小为5*5，步距为1，输入通道为3，输出通道为6'''
        self.conv1=nn.Conv2d(3,6,kernel_size=5,stride=1)
        
        '''第一层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一办'''
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        '''第二层卷积，卷积核大小为5*5，步距为1，输入通道为6，输出通道为16'''
        self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1)
        
        '''第二层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一办'''
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        '''第一层全连接层，维度由16*5*5=>120'''
        self.linear1=nn.Linear(16*5*5,120)
        
        '''第二层全连接层，维度由120=>84'''
        self.linear2=nn.Linear(120,84)
        
        '''第二层全连接层，维度由64=>10'''
        self.linear3=nn.Linear(84,num_classes)
        
    def forward(self, x):
        '''将数据送入第一个卷积层'''
        out=torch.sigmoid(self.conv1(x))
        
        '''将数据送入第一个池化层'''
        out=self.pool1(out)
        
        '''将数据送入第二个卷积层'''
        out=torch.sigmoid(self.conv2(out))
        
        '''将数据送入第二个池化层'''
        out=self.pool2(out)
        
        '''将池化层后的数据进行Flatten，使数据变成能够被FC层接受的Vector'''
        out=out.reshape(-1,16*5*5)
        
        '''将数据送入第一个全连接层'''
        out=torch.sigmoid(self.linear1(out))
        
        '''将数据送入第二个全连接层'''
        out=torch.sigmoid(self.linear2(out))
        
        '''将数据送入第三个全连接层得到输出'''
        out=self.linear3(out)
        
        return out
    
model = LeNet(num_classes)

'''设置损失函数'''
criterion=nn.CrossEntropyLoss()

'''设置优化器'''
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

