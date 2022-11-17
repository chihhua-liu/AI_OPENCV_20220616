
# import torch
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# BATCH_SIZE = 5
# x = torch.linspace(1, 10, 10)
# y = torch.linspace(11, 20, 10)
# print(x)
# print(y)
# torch_dataset = TensorDataset(x, y)  # 將x, y 轉換成 tensor shuffle ,batch ...
# # num_workers=0 不用多核心CPU
# loader = DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0 )
# for epoch in range(10):
#     for step, (batch_x, batch_y) in enumerate(loader):
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(),
#               '| batch y: ', batch_y.numpy())

# __getitem__：返回一個數據樣本(obj[index]等價於obj.__getitem__(index))
# __len__：返回樣本的數量(len(obj)等價於obj.__len__())
# import torch
# from torchvision.datasets import mnist
# from torch.utils.data import DataLoader
# import numpy as np
#
# def data_transform(x):
#     x = np.array(x,dtype = 'float32')/255
#     x = x.reshape((-1,))   # DNN used 1D = 784*1
#     x = torch.from_numpy(x)
#     return x
#
# trainset =mnist.MNIST('./dataset/mnist', train =True, transform = data_transform,download=True )
# testset =mnist.MNIST('./dataset/mnist', train =False, transform = data_transform,download=True )
#
# print(trainset)
#
# train_data = DataLoader(trainset, batch_size = 64, shuffle =True)
# teat_data = DataLoader(testset, batch_size = 128, shuffle = False)
#
# print('trainset.data=',len(trainset.data))
# print('testset.data=',len(testset.data))
# print('trainset.data.numpy().shape=',trainset.data.numpy().shape)
# print('trainset.targets.numpy().shape=',trainset.targets.numpy().shape)
# print('trainset.targets.numpy()=',trainset.targets.numpy())
#
# print('testset.data.numpy().shape=',testset.data.numpy().shape)
# print('testset.targets.numpy().shape=',testset.targets.numpy().shape)
# print('testset.targets.numpy()=',testset.targets.numpy())
#
# import matplotlib.pyplot as plt
#
# plt.imshow(trainset.data[0].numpy(),cmap='gray')
# plt.show()
#
# for images, labels in train_data: # batch_size=64 ，一次get  64筆資料
#     print(images.shape)
#     print(labels.shape)
#     print(images)
#     print(labels)
#
#     break

####################################################
# Build DNN Model(Functional API): method        ##
#####################################################
import torch
import numpy as np
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
%matplotlib inline

def data_transform(x):
    x = np.array(x, dtype = 'float32')/255
    x = (x-0.5)/0.5
    x = x reshape((-1,))
    x = torch.from_numpy(x)
    return x

trainset = mnist.MNIST('./dataset/minst', train=True, transform = data_transform, download =True)
testset = mnist.MNIST('./dataset/minst', train=False, transform = data_transform, download =True)
train_data = DataLoader(trainset, batch_size = 64, shuffle = True)
test_data = DataLoader(testset, batch_size = 128, shuffle = False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.linear(500,250)
        self.fc3 = nn.linear(250,125)
        self.fc4 = nn.linear(125,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

mlp = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), 1e-3)

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0

    mlp.train()

    for im, label in train_data:
        #        im = Variable(im)
        #        label = Variable(label)
        # 前向傳播
        # print(im.shape)
        out = mlp(im)

        loss = criterion(out, label)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 記錄誤差
        train_loss += loss.item()

        # 計算分類的準確率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    # 在測試集上檢驗效果
    eval_loss = 0
    eval_acc = 0
    mlp.eval()  # 將模型改為預測模式

    for im, label in test_data:
        #        im = Variable(im)
        #        label = Variable(label)

        out = mlp(im)
        loss = criterion(out, label)
        # 記錄誤差

        eval_loss += loss.item()
        # 記錄準確率

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
                  eval_acc / len(test_data)))