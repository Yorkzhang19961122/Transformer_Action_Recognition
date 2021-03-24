import time
import torch
import numpy as np
from models.transformer_encoder import *
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt

# 初始化数据集
LABELS = ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"]
DATASET_PATH = "data/HAR_pose_activities/database/"

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH+ "X_test.txt"

Y_train_path = DATASET_PATH + "Y_train.txt"
Y_test_path = DATASET_PATH + "Y_test.txt"

# n_steps = 32 # 32 timesteps per series
#
# # Load the networks inputs
# def load_X(X_path):
#     file = open(X_path, 'r')
#     X_ = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
#     file.close()
#     blocks = int(len(X_) / n_steps)
#     X_ = np.array(np.split(X_, blocks))
#
#     return X_
#
# # Load the networks outputs
# def load_y(y_path):
#     file = open(y_path, 'r')
#     y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
#     file.close()
#     # for 0-based indexing
#     return y_ - 1
#
# X_train = load_X(X_train_path)
# X_test = load_X(X_test_path)
# Y_train = load_y(Y_train_path)
# Y_test = load_y(Y_test_path)
# print('lenlenlen:', len(np.loadtxt(Y_train_path)))
# print(X_train)
# print('111:', X_train[0,:,:])
# print('111.size:', X_train[0,:,:].shape)
# print('222:', Y_train[0])
# print(X_train[9000,:,:])
# print(X_train[9000,:,:].shape)
# print(Y_train[9000])
# print('222:', Y_train[0].shape)
# # 可视化数据
# training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
# print('len(X_train):', training_data_count)
# test_data_count = len(X_test)  # 1197 test series
# print('len(X_test):', test_data_count)
# n_input = len(X_train[0][0])  # num input parameters per timestep
# print('feature_num:', n_input)
# print('X_train_shape, y_train_shape, X_test_shape, y_test_shape')
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
#
# # print("(X shape, y shape, every X's mean, every X's standard deviation)")
# # print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
# # print("\nThe dataset has not been preprocessed, is not normalised etc")
#
# # 生成每个batch的数据
# def extract_batch_size(_train, _labels, _unsampled, batch_size):
#     # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data.
#     # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
#     # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size
#
#     shape = list(_train.shape)  # 把矩阵转为list shape=[22625, 32, 36](list)
#     shape[0] = batch_size  # shape=[batch_size, 32, 36](list)
#     batch_s = np.empty(shape)  # 返回一个shape形状的多维数组 batch_s.shape=[batch_size, 32, 36]
#     batch_labels = np.empty((batch_size, 1)) # 返回一个存放labels的数组 batch_labels.shape=[batch_size, 1]
#
#     for i in range(batch_size):
#         # 随机取出batch_size个32x36的矩阵作为骨骼数据与标签用作训练
#         # Loop index
#         # index = random sample from _unsampled (indices)
#         index = random.choice(_unsampled)  # 返回一个unsampled中的随机项(取随机数作为索引)
#         batch_s[i] = _train[index]
#         batch_labels[i] = _labels[index]
#         _unsampled.remove(index)  # 将已经使用过的数据索引从列表中去除
#         '''print('batch_s[i]:')
#         print(batch_s[i])
#         print('batch_labels[i]')
#         print(batch_labels[i])'''
#     return batch_s, batch_labels, _unsampled


# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(data.Dataset):
    # 初始化
    def __init__(self, X_path, Y_path, n_steps, transform=None):
        self.X_path = X_path
        self.Y_path = Y_path
        self.transform = transform
        self.n_steps = n_steps
        self.len = np.loadtxt(Y_path)
        self.X_file = open(X_path, 'r')
        self.Y_file = open(Y_path, 'r')
        self.X_ = np.array([elem for elem in [row.split(',') for row in self.X_file]], dtype=np.float32)
        self.Y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in self.Y_file]], dtype=np.int32)  # Y_
        self.blocks = int(len(self.X_) / self.n_steps)
        self.X_split = np.array(np.split(self.X_, self.blocks))   # X_split
        self.X_file.close()
        self.Y_file.close()

    def __len__(self):
        return len(self.len)

    def __getitem__(self, item):
        x = self.X_split[item,:,:]
        y = self.Y_[item]

        return x, y - 1



# for step, (inputs, labels) in enumerate(dataloader):
#     print('inputs:',inputs)
#     print('inputs.size:', inputs.shape)
#     print('labels:', labels)
#     print('labels.size:', labels.shape)


# def one_hot(y_):
#     # One hot encoding of the network outputs
#     # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
#     y_ = y_.reshape(len(y_))
#     n_values = int(np.max(y_)) + 1
#     return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
#
# unsampled_indicies = list(range(0, len(X_train)))
# print('len(X_train):', len(X_train))
# # batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
# # 得到每个batch的数据和标签
# # batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indicies, batch_size)
# # batch_ys = one_hot(raw_labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TransformerOnlyDecoder(32, 36, 6, 36, 64, 6).to(device)
# print(model)
batch_size = 1
epoch = 5
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model_save_path = "output/model_trained.pkl"
train_loader_dataset = MyDataset(X_train_path, Y_train_path, 32)
train_loader = data.DataLoader(dataset=train_loader_dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
test_loader_dataset = MyDataset(X_test_path, Y_test_path, 32)
test_loader = data.DataLoader(dataset=test_loader_dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

def calc_acc(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.tensor(labels).long().squeeze(dim=1)
            net_output = model(inputs)
            pred = net_output.argmax(dim=1)
            total += len(labels)
            correct += (pred == labels).sum().item()
    acc = 100.0*correct/total
    return acc

# training
for e in range(epoch):
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # 梯度参数归零
        # forward + backward + optimize
        net_output = model(torch.tensor(inputs, dtype=torch.float))  # 前向传播 net_out.shape=[batch_size, classes]
        labels = torch.tensor(labels).long().squeeze(dim=1)
        loss = criterion(net_output, labels)  #计算loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        acc = calc_acc(model, test_loader)
        print('[Epoch: {}/{}][{}/{}], Train Loss: {:.4f}, Acc: {:.4f}]'.format(e+1, epoch, step*len(inputs), len(train_loader.dataset), loss, acc))
        # train_loss += float(loss.item())
        # pred = net_output.argmax(dim=1)
        # print('--------------------')
        # print('pred:', pred)
        # print('labels:', labels)
        # print('(pred==labels).sum().item():', (pred==labels).sum().item())
        # print('--------------------')



        # print('pred:', pred)
        # print('true:', labels)
        # print('step:', step)
        # num_correct += torch.eq(pred, labels).sum().float().item()
        # print('Epoch %d/%d, Loss: %.2f'%(e+1, epoch, loss))
        # print('Epoch: {}/{} [{}/{}] Train Loss: {:.4f}'.format(e+1, epoch, i*len(inputs), len(train_loader.dataset), loss.item()))
        # print('[Epoch: {}/{}, Train Loss: {:.4f}, Acc: {:.4f}]'.format(e+1, epoch, train_loss/(len(train_loader)), num_correct/(len(train_loader_dataset))))
#torch.save(model, model_save_path)
print('Finished Training')

# evaluation

# model = torch.load(model_save_path)
# for i, (inputs, labels) in enumerate(dataloader_test):
#     labels_test_out = model(inputs)
#     print('test_out:', labels_test_out)
