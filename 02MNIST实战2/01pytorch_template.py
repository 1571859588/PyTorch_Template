'''
只需要更改模型和训练方式即可

以下函数和类已定义好，可以直接调用，以下的'保存'路径均不用考虑是否存在问题
评估：evaluate(dataloader, model, device, flag, criterion, optimizer=None, epoch=None, epochs=None):
    返回平均损失，某次batch的平均准确率，测试准确率（数组）,model,optimizer  flag='valid' / 'test' / 'train'
自定义数据集：Dataset_name(flag='train', csv_paths=None))
超参数类：argparse(self, csv_paths, hidden_size=None, lr_adjust=None, input_size=30, output_size=12, epochs=30,
                 original_lr=0.001,
                 patience=4, cuda_id=0):仅argparse.device="cuda:cuda_id" 其余返回值都跟参数名称一样
早停类：EarlyStopping(patience=7, verbose=False, delta=0) verbose是否打印信息 其对象属性值early_stop为True时表示早停
保存模型（保存）：save_checkpoint(model, optimizer, epoch=None, filepath='./best_model/1.pth')
加载模型：load_checkpoint(filepath, model, optimizer, device):model, optimizer, epoch,hidden_size
画图（保存）：save_figure(train_loss, train_epochs_loss, valid_epochs_loss,save_path="./total_loss_figs/1.png")
'''
import torchvision.utils

''' 导入包'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 可以调用一些常见的函数，如非线性和池化等
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms, datasets

'''导入自定义模板工具函数及类'''
from template_utils import Dataset_name, argparse, evaluate, EarlyStopping, load_checkpoint, save_figure, \
    save_checkpoint, calculate_accuracy


# --------------------------------------------   一般只用改下面的实现  -------------------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    print("img shape", img.shape)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


''' 定义自己的模型'''


class Your_model_Name(nn.Module):
    # hidden_size是数组，另外两个是数字
    def __init__(self, input_size, hidden_size, output_size):
        super(Your_model_Name, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

        pass

    def forward(self, x):
        # print("x forward shape ",x.shape) # x=64x1x28x28
        # x=x.view(x.size(0),-1)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 横纵方向步长为2  # x=64x6x12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (2,2)也可以像这样写成2 # x=64x16x4x4
        x = x.view(-1, self.num_flat_features(x)) # x= 64x256

        x = F.relu(self.fc1(x))  # x= 64 x hidden_size[0]
        x = F.relu(self.fc2(x))  # x= 64 x hidden_size[1]
        x = self.fc3(x)   # x= 64 x output_size
        # print("x shape ",x.shape)  # x = 64 x 10
        # print("x ",x)
        return x

    def num_flat_features(self, x): # x=64x16x4x4
        # print("x1 shape",x.shape)
        size = x.size()[1:]  # 忽略第0维度，提取第1维度以及后面的维度
        # print('size ',size) #size=16x4x4

        num_features = 1
        for s in size:
            num_features *= s
        # print("num_features ",num_features)
        return num_features # num_features=256


def train(args, csv_paths):
    csv_files = os.listdir(csv_paths[0])
    csv_files.sort()

    train_total_acc = []
    valid_total_acc = []
    test_total_acc = []
    train_total_loss = []
    valid_total_loss = []
    test_total_loss = []
    '''实例化模型，设置loss，优化器等'''
    model = Your_model_Name(args.input_size, args.hidden_size, args.output_size).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.original_lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for csv_file in csv_files:

        # 这个数据集只是单个文件的，即单个被试的，如果需要多个的话则需要自行改成循环结构
        # 数据加载
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        # 显示一个batch的训练集图片
        dataiter = iter(train_dataloader)
        images, labels = dataiter.__next__()
        print("images shape", images.shape)
        # imshow(torchvision.utils.make_grid(images))
        # 假设你想显示批次中的第一张图像
        # image_to_show = images[0]  # 获取第一张图像
        # imshow(image_to_show)  # 显示第一张图像

        train_epochs_loss = []
        valid_epochs_loss = []
        train_epochs_accuracy = []
        valid_epochs_accuracy = []

        for epoch in range(args.epochs):
            # ===================train===============================
            avg_loss, avg_acc, _, _, model, optimizer = evaluate(dataloader=train_dataloader, model=model,
                                                                 device=args.device,
                                                                 flag='train',
                                                                 criterion=criterion, optimizer=optimizer,
                                                                 epoch=epoch + 1,
                                                                 epochs=args.epochs)
            train_epochs_loss.append(avg_loss)
            train_epochs_accuracy.append(avg_acc)

            # =====================valid============================
            # avg_loss, avg_acc, _,_,_ = evaluate(dataloader=valid_dataloader, model=model, device=args.device,
            #                                 flag='valid',
            #                                 criterion=criterion)
            # valid_epochs_loss.append(avg_loss)
            # valid_epochs_accuracy.append(avg_acc)

            # ==================early stopping======================
            early_stopping(train_epochs_loss[-1], model=model,
                           path='./early_stop_model/' + str(round(avg_acc, 4)) + '.pth')
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # ====================adjust lr========================
            if args.lr_adjust is not None:
                if epoch in args.lr_adjust.keys():
                    lr = args.lr_adjust[epoch]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Updating learning rate to {}'.format(lr))

        # ======================= 找到最佳模型 ===================================
        avg_loss, avg_acc, accs, test_losses, _, _ = evaluate(dataloader=test_dataloader, model=model,
                                                              device=args.device,
                                                              flag='test',
                                                              criterion=criterion)
        # ====================== 汇总 ====================================
        train_total_loss.append(np.average(train_epochs_loss))
        train_total_acc.append(np.average(train_epochs_accuracy))
        valid_total_loss.append(np.average(valid_epochs_loss))
        valid_total_acc.append(np.average(valid_epochs_accuracy))
        test_total_acc.append(avg_acc)
        test_total_loss.append(avg_loss)
        pre_acc = 0

        csv_path = './best_model/csvs/' + csv_file[:-4]
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        best_model_files = os.listdir(csv_path)
        best_model_files.sort()
        now_acc = np.average(test_total_acc)
        if len(best_model_files):
            pre_acc = float(best_model_files[-1][:-4])
        if now_acc > pre_acc:
            save_figure(train_epochs_accuracy, valid_epochs_accuracy, accs,
                        save_path='./figures/csvs_acc_figs/' + csv_file[:-4] + '/' + str(round(now_acc, 4)) + '.png')
            save_figure(train_epochs_loss, valid_epochs_loss, test_losses,
                        save_path='./figures/csvs_loss_figs/' + csv_file[:-4] + '/' + str(round(now_acc, 4)) + '.png')
            save_checkpoint(model, optimizer, hidden_size=args.hidden_size,
                            filepath='./best_model/csvs/' + csv_file[:-4] + '/' + str(round(now_acc, 4)) + '.pth')

    pre_acc = 0
    best_model_files = os.listdir('./best_model/total/')
    best_model_files.sort()
    now_acc = np.average(test_total_acc)
    if len(best_model_files):
        pre_acc = float(best_model_files[-1][:-4])
    if now_acc > pre_acc:
        save_figure(train_total_acc, valid_total_acc, test_total_acc,
                    save_path='./figures/total_acc_figs/' + str(round(now_acc, 4)) + '.png')
        save_figure(train_total_loss, valid_total_loss, test_total_loss,
                    save_path='./figures/total_loss_figs/' + str(round(now_acc, 4)) + '.png')
        save_checkpoint(model, optimizer, hidden_size=args.hidden_size,
                        filepath='./best_model/total/' + str(round(now_acc, 4)) + '.pth')


def main():
    ''' 设置随机种子'''
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ''' 设置超参数'''
    # 数据集默认就是在代码同级的
    # 这个路径和名字可以随便改，但是三者的顺序不能变动，一定是训练集、验证集、测试集 且都是只有最后一列是标签
    csv_paths = ['./01preprocessed_data/train_data/',
                 './01preprocessed_data/valid_data/',
                 './01preprocessed_data/test_data/']
    # 输入层
    input_size = 784
    # 输出层
    output_size = 10
    # 训练次数
    epochs = 10
    # 学习率
    original_lr = 0.001
    # 耐心
    patience = 4
    # gpu设备
    cuda_id = 0
    # 学习率调整 epoch：lr
    lr_adjust = {
        2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        10: 5e-7, 15: 1e-7, 20: 5e-8
    }
    # 寻找最优网络结构次数
    search_net_num = 5

    for i in range(search_net_num):
        # 隐藏层 从前往后
        hidden_size = [random.randint(100, 125), random.randint(80, 85)]
        args = argparse(hidden_size=hidden_size, input_size=input_size, output_size=output_size,
                        epochs=epochs,
                        original_lr=original_lr, lr_adjust=lr_adjust, patience=patience, cuda_id=cuda_id)
        train(args, csv_paths)


if __name__ == '__main__':
    main()
