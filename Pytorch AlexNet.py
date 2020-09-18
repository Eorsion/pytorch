# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:25:14 2020

@author: 安辰
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchsummary as summary

'''定义参数'''
batch_size=128
total_epoch=10
lr=0.001
classes_num=10

'''获取数据集'''
train_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=True,download=True,transform=transforms.ToTensor())

test_dataset=torchvision.datasets.CIFAR10(root='../DataSet/',train=False,transform=transforms.ToTensor())

'''数据装载'''
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        
        '''第一层卷积层，卷积核为3*3，通道数为96，步距为1，原始图像大小为32*32，有R、G、B三个通道'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(32-3)/1+1=30,所以feature map的维度为96*30*30'''
        
        self.conv1=nn.Conv2d(3,96,kernel_size=3,stride=1)
        
        '''经过一次批归一化，将数据拉回到正态分布'''
        
        self.bn1=nn.BatchNorm2d(96)
        
        '''第一层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为30*30，通道数为96个'''
        
        '''这样经过第一层池化层之后，得到的feature map的大小为(30-3)/2+1=14,所以feature map的维度为96*14*14'''
        
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''第二层卷积层，卷积核为3*3，通道数为256，步距为1，前一层的feature map的大小为14*14，通道数为96个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(14-3)/1+1=12,所以feature map的维度为256*12*12'''
        
        self.conv2=nn.Conv2d(96,256,kernel_size=3,stride=1)
        
        '''经过一次批归一化，将数据拉回到正态分布'''
        
        self.bn2=nn.BatchNorm2d(256)
        
        '''第二层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为12*12，通道数为256个'''
        
        '''这样经过第二层池化层之后，得到的feature map的大小为(12-3)/2+1=5,所以feature map的维度为256*5*5'''
        
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''第三层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为256个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''
        
        self.conv3=nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1)
        
        '''第四层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''
        
        self.conv4=nn.Conv2d(384,384,kernel_size=3,padding=1,stride=1)
        
        '''第五层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''
        
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为256*5*5'''
        
        self.conv5=nn.Conv2d(384,256,kernel_size=3,padding=1,stride=1)
        
        '''第三层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为5*5，通道数为256个'''
        
        '''这样经过第三层池化层之后，得到的feature map的大小为(5-3)/2+1=2,所以feature map的维度为256*2*2'''
        
        self.pool3=nn.MaxPool2d(kernel_size=3,stride=2)
        
        '''经过第一层全连接层'''
        
        self.linear1=nn.Linear(1024,2048)
        
        '''经过第一次DropOut层'''
        
        self.dropout1=nn.Dropout(0.5)
        
        '''经过第二层全连接层'''
        
        self.linear2=nn.Linear(2048,2048)
        
        '''经过第二层DropOut层'''
        
        self.dropout2=nn.Dropout(0.5)
        
        '''经过第三层全连接层，得到输出结果'''
        
        self.linear3=nn.Linear(2048,10)
        
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.pool1(out)
        
        
        out=self.conv2(out)
        out=self.bn2(out)
        out=F.relu(out)
        out=self.pool2(out)
        
        out=F.relu(self.conv3(out))
        
        out=F.relu(self.conv4(out))
        
        out=F.relu(self.conv5(out))
        
        out=self.pool3(out)
        
        out=out.reshape(-1,256*2*2)
        
        out=F.relu(self.linear1(out))
        
        out=self.dropout1(out)
        
        out=F.relu(self.linear2(out))
        
        out=self.dropout2(out)
        
        out=self.linear3(out)
        
        return out
        
model=AlexNet()

summary.summary(model, input_size=(3, 32, 32),batch_size=128,device="cpu")

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

'''开始训练'''
total_step = len(train_loader)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):

#         images=Variable(images)
#         labels=Variable(labels)
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
