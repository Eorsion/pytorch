import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary as summary 
import os

'''定义超参数'''
batch_size=64
epoch_total=10
classes_num=10
learning_rate=1e-3

'''定义Transform'''

data_transform={
    
    "train":transforms.Compose([
            
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ]),    
    
    "val":transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ])
    
}

'''获取图片地址'''
data_root=os.path.abspath(os.path.join(os.getcwd(),"../"))
image_root=data_root+"/DataSet/flower_data/"

'''获取数据'''
train_dataset=torchvision.datasets.ImageFolder(root=image_root+"train",transform=data_transform["train"])

val_dataset=torchvision.datasets.ImageFolder(root=image_root+"val",transform=data_transform["val"])

'''装载数据'''

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True)

'''定义公共层：conv+bn+relu'''

def CBR(in_channels,out_channels):
    
    cbr=nn.Sequential(
        
        nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return cbr

'''定义VGG'''

class VGG(nn.Module):
    
    def __init__(self, block_nums):
        
        super(VGG,self).__init__()
        
        self.block1=self._make_layers(in_channels=3,out_channels=64,block_num=block_nums[0])
        self.block2=self._make_layers(in_channels=64,out_channels=128,block_num=block_nums[1])
        self.block3=self._make_layers(in_channels=128,out_channels=256,block_num=block_nums[2])
        self.block4=self._make_layers(in_channels=256,out_channels=512,block_num=block_nums[3])
        self.block5=self._make_layers(in_channels=512,out_channels=512,block_num=block_nums[4])
        
        self.classifier=nn.Sequential(
            
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, classes_num)
        )
    
    
    def _make_layers(self,in_channels,out_channels,block_num):
        
        blocks=[]
        blocks.append(CBR(in_channels,out_channels))
        
        for i in range(1,block_num):
            
            blocks.append(CBR(out_channels,out_channels))
        
        blocks.append(nn.MaxPool2d(kernel_size=2,stride=2))
            
        return nn.Sequential(*blocks)
        
    def forward(self,x):
        
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=torch.flatten(x,start_dim=1)
        out=self.classifier(x)
        
        return out


def VGG16():
    block_nums=[2,2,3,3,3]
    model=VGG(block_nums)
    return model

model=VGG16()

print(model)
summary.summary(model, input_size=(3,224,224),device="cpu")

'''设置损失函数和优化器'''
model = VGG16()

loss_function=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
'''开始训练'''
def train():
    
    step_total=len(train_loader)
    
    for epoch in range(epoch_total):
        
        for step,(image,label) in enumerate(train_loader):
            
            model=VGG16()
            
            pred=model(image)
            
            loss=loss_function(pred,label)
            
            optimizer.zero_grad()
            
            loss.back_ward()
            
            optimizer.step()
            
            print("Epoch:[{}/{}],Step:[{}/{}],loss:{:.4f}".format(epoch,epoch_total,step+1,step_total,loss.item()))

if __name__ == '__main__':
    
    train()
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        