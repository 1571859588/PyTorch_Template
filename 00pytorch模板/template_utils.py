''' 导入包'''
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms
import torch.nn.functional as F
''' 以类的方式定义超参数'''


class argparse():
    # hidden_size是一个列表
    def __init__(self, csv_paths=None, hidden_size=None, lr_adjust=None, input_size=30, output_size=12, epochs=30,
                 original_lr=0.001,
                 patience=4, cuda_id=0):
        if hidden_size is None:
            hidden_size = [40, 20]
        self.device = torch.device("cuda:" + str(cuda_id) if torch.cuda.is_available() else "cpu")
        self.epochs, self.original_lr, self.patience = epochs, original_lr, patience
        self.hidden_size, self.input_size = hidden_size, input_size
        self.csv_paths = csv_paths
        self.lr_adjust = lr_adjust
        self.output_size = output_size
        # 集群
        # self.device, = [torch.device("cuda:"+str(cuda_num) if torch.cuda.is_available() else "cpu"), ]
        pass

    def check_cuda(self):
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        # 如果CUDA可用，打印出可用GPU的数量
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            print(f"CUDA is available. Number of GPUs: {num_gpus}")
        else:
            print("CUDA is not available.")
        # 根据CUDA的可用性选择运行设备的类型


''' 定义早停类(此步骤可以省略)'''


# 早停类（EarlyStopping）是一个在训练过程中用来避免过拟合的技术。它的基本思想是在验证集上的性能不再提高时停止训练，从而防止模型在训练集上过度训练。
class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        # 早停的耐心值，即连续多少个epoch没有提升就停止训练
        self.patience = patience
        # 是否打印信息
        self.verbose = verbose
        # 计数器，记录连续多少个epoch没有提升
        self.counter = 0
        # 最佳验证集损失
        self.best_score = None
        # 是否早停的标志
        self.early_stop = False
        # 目前为止的最小验证集损失
        self.val_loss_min = np.Inf
        # 提升阈值，如果损失减少小于这个值，则不计为提升
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # 如果当前验证集损失小于最小验证集损失减去提升阈值
        if self.best_score is None or val_loss < self.val_loss_min - self.delta:
            # 更新最佳验证集损失
            self.best_score = val_loss
            # 保存模型
            self.save_checkpoint(val_loss, model, path)
            # 重置计数器
            self.counter = 0
        else:
            # 否则，计数器加一
            self.counter += 1
            # 如果计数器达到了耐心值，设置早停标志为True
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, path):
        # 如果设置了打印信息，打印损失减少的信息
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型状态
        torch.save(model.state_dict(), path)
        # 更新最小验证集损失
        self.val_loss_min = val_loss


''' 绘图'''


def save_figure(train, valid,test, save_path='total_loss_figs/1.png'):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.figure(figsize=(12, 4))
    plt.plot(train[:], '-o', label="train")
    plt.plot(valid[:], '-o', label="valid")
    plt.plot(test[:], '-o', label="test")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    # plt.show()


''' 加载模型'''


# filepath为pth路径，model是需要导入的模型
def load_checkpoint(filepath, model, optimizer, device):
    if not os.path.exists(filepath):
        print("该模型不存在")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {filepath}, epoch {epoch}")
    print("hidden_size=",checkpoint['hidden_size'])
    return model, optimizer, epoch,checkpoint['hidden_size']


''' 保存模型'''


# model、optimizer要使用实例化的对象，而不是类名
def save_checkpoint(model, optimizer, hidden_size=0, filepath='./best_model/1.pth', epoch=None):
    # 检查文件夹是否存在
    if not os.path.exists(os.path.dirname(filepath)):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(os.path.dirname(filepath))
        print("文件夹已成功创建")
    else:
        print("文件夹已存在")

    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'hidden_size': hidden_size
    }
    torch.save(state, filepath)
    print(f"Model saved to {filepath}")


'''评估 flag=valid  或者 预测 flag=test '''


# 返回平均损失，某次batch的平均准确率，测试准确率（数组）,损失（数组），model,optimizer
# flag=valid / test / train
# 当flag=train时，需要添加optimizer,epoch,epochs参数
def evaluate(dataloader, model, device, flag, criterion, optimizer=None, epoch=None, epochs=None):
    valid_loss = []
    accuracys = []
    if flag == 'train':
        model.train()
    else:
        model.eval()

    for idx, (data_x, data_y) in enumerate(dataloader):
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        outputs = model(data_x)
        # print("outputs shape========",outputs.shape)
        # print("data_y shape========", data_y.shape)
        # print("outputs:",outputs)
        # print("data_y:",data_y)

        loss = criterion(outputs, data_y)
        valid_loss.append(loss.item())

        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, data_y)
        accuracys.append(accuracy)

        if flag == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print("epoch={}/{},{}/{} of train, loss={}, train_accuracy={:.4f}".format(
                    epoch, epochs, idx, len(dataloader), loss.item(), accuracy))
        else:
            if idx % 100 == 0:
                print("{}/{} of valid, loss={}, valid_accuracy={:.4f}".format(
                    idx, len(dataloader), loss.item(), accuracy))

    return np.average(valid_loss), np.average(accuracys), accuracys,valid_loss, model, optimizer


# calculate_accuracy:
# 假设你的模型正在处理一个分类问题，比如识别手写数字，有10个可能的类别（0到9）。模型对于每个输入样本会输出一个10维的向量，这个向量中的每个元素代表了模型认为该样本属于对应类别的概率。
# outputs张量的形状可能是(batch_size, num_classes)，其中batch_size是批量中的样本数量，num_classes是类别的数量。例如，如果批量大小为4，那么outputs可能看起来像这样：
# [
#  [0.1, 0.2, 0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # 示例1的输出
#  [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],  # 示例2的输出
#  [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.1, 0.0, 0.0],  # 示例3的输出
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.1]   # 示例4的输出
# ]
# 在这个例子中，第一行的最大值是0.5，对应的索引是2，这意味着模型预测第一个样本属于类别2。同样地，其他行的最大值索引分别表示模型对其他样本的预测类别。
# 当我们调用torch.max(outputs, 1)时，我们得到的是一个元组，其中第一个元素是每行的最大值，第二个元素是这些最大值的索引。我们通常只关心索引，因为它们代表了模型的预测，所以我们将索引赋值给predicted变量：
# _, predicted = torch.max(outputs, 1)
# 现在predicted张量看起来可能像这样：
# [2, 9, 6, 7]
# 这表示模型预测第一个样本属于类别2，第二个样本属于类别9，依此类推。
# torch.max(outputs, 1)的作用是在每一行（即沿着维度1）上找到最大值和对应的索引。

# 计算准确率 outputs 和 labels都是张量
def calculate_accuracy(outputs, labels):
    _, predicted_index = torch.max(outputs, 1)  # predicted_index 即 分类类别
    total = labels.size(0)
    correct = (predicted_index == labels).sum().item()
    accuracy = correct / total
    return accuracy


''' 定义自己的数据集Dataset,DataLoader'''


# 数据集默认就是在代码同级的
# 训练集、验证集、测试集=['./data_train.csv', './data_valid.csv', './data_test.csv']
# 且以上3个都是只有最后一列是标签
class Dataset_name(Dataset):
    def __init__(self, flag='train', csv_paths=None):
        self.label = None
        self.data = None
        if csv_paths is None:
            csv_paths = ['./data_train.csv', './data_valid.csv', './data_test.csv']
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.__load_data__(csv_paths)

        # 归一化
        self.transforms()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)

    def __load_data__(self, csv_paths: list):
        # 假设data.csv中的数据和标签是分开的，我们可以使用pandas来加载数据
        # 假设数据的第一列是特征，剩下的列是标签 训练数据存储在data_train.csv
        data, label = None, None
        if self.flag == 'train':
            data = pd.read_csv(csv_paths[0]).iloc[:, :-1].values
            label = pd.read_csv(csv_paths[0]).iloc[:, -1].values
            print(
                "data.shape:{}\nlabel.shape:{}\n"
                .format(data.shape, label.shape))
        elif self.flag == 'valid':
            # 假设我们还有验证和测试数据，它们分别存储在data_valid.csv和data_test.csv
            data = pd.read_csv(csv_paths[1]).iloc[:, :-1].values
            label = pd.read_csv(csv_paths[1]).iloc[:, -1].values
            print(
                "data.shape:{}\nlabel.shape:{}\n"
                .format(data.shape, label.shape))
        elif self.flag == 'test':
            data = pd.read_csv(csv_paths[2]).iloc[:, :-1].values
            label = pd.read_csv(csv_paths[2]).iloc[:, -1].values
            print(
                "data.shape:{}\nlabel.shape:{}\n"
                .format(data.shape, label.shape))
        self.data = data
        self.label = label

    def cal_mean_std(self):
        # 计算特征的均值和标准差
        features = self.data
        mean, std = 0, 0
        # 展平特征数组
        flattened_features = features.flatten()
        mean = np.mean(flattened_features, axis=0)
        std = np.std(flattened_features, axis=0)
        return mean, std

    def transforms(self):
        mean, std = self.cal_mean_std()
        self.data = self.data.float()
        self.label = self.label.float()
        self.data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))])(self.data)
