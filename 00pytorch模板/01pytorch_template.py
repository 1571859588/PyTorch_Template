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
from torchvision import transforms,datasets

'''导入自定义模板工具函数及类'''
from template_utils import Dataset_name, argparse, evaluate, EarlyStopping, load_checkpoint, save_figure, \
    save_checkpoint, calculate_accuracy

# --------------------------------------------   一般只用改下面的实现  -------------------------------------------------------------------
''' 定义自己的模型'''


class Your_model_Name(nn.Module):
    # hidden_size是数组，另外两个是数字
    def __init__(self, input_size, hidden_size, output_size):
        super(Your_model_Name, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        # 可改
        self.fc1=nn.Linear(input_size,hidden_size[0])
        self.fc2=nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3=nn.Linear(hidden_size[1],output_size)


        pass

    def forward(self, x):
        # 下面可以改
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        train_file = csv_files[0] + csv_file
        valid_file = csv_files[1] + csv_file
        test_file = csv_files[2] + csv_file
        csvs = [train_file, valid_file, test_file]
        # 这个数据集只是单个文件的，即单个被试的，如果需要多个的话则需要自行改成循环结构
        # 数据加载
        train_dataset = Dataset_name(flag='train', csv_paths=csvs)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        valid_dataset = Dataset_name(flag='valid', csv_paths=csvs)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)
        test_dataset = Dataset_name(flag='test', csv_paths=csvs)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

        train_epochs_loss = []
        valid_epochs_loss = []
        train_epochs_accuracy = []
        valid_epochs_accuracy = []

        ''' 开始训练以及调整lr'''
        for epoch in range(args.epochs):
            # ===================train===============================
            avg_loss, avg_acc, _,_, model, optimizer = evaluate(dataloader=train_dataloader, model=model,
                                                              device=args.device,
                                                              flag='train',
                                                              criterion=criterion, optimizer=optimizer, epoch=epoch+1,
                                                              epochs=args.epochs)
            train_epochs_loss.append(avg_loss)
            train_epochs_accuracy.append(avg_acc)

            # =====================valid============================
            avg_loss, avg_acc, _,_,_ = evaluate(dataloader=valid_dataloader, model=model, device=args.device,
                                            flag='valid',
                                            criterion=criterion)
            valid_epochs_loss.append(avg_loss)
            valid_epochs_accuracy.append(avg_acc)

            # ==================early stopping======================
            early_stopping(train_epochs_loss[-1], model=model, path='./early_stop_model/'+str(round(avg_acc,4))+'.pth')
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
        avg_loss, avg_acc, accs,test_losses, _,_  = evaluate(dataloader=test_dataloader, model=model, device=args.device,
                                              flag='test',
                                              criterion=criterion)
        # ====================== 汇总 ====================================
        train_total_loss.append(np.average(train_epochs_loss))
        train_total_acc.append(np.average(train_epochs_accuracy))
        valid_total_loss.append(np.average(valid_epochs_loss))
        valid_total_acc.append(np.average(valid_epochs_accuracy))
        test_total_acc.append(avg_acc)
        test_total_loss.append(avg_loss)
        # ====================== 记录一个被试的所有epochs================

        pre_acc = 0

        csv_path='./best_model/csvs/'+csv_file[:-4]
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        best_model_files = os.listdir(csv_path)
        best_model_files.sort()
        now_acc = np.average(test_total_acc)
        if len(best_model_files):
            pre_acc = float(best_model_files[-1][:-4])
        if now_acc > pre_acc:
            save_figure(train_epochs_accuracy, valid_epochs_accuracy,accs,
                        save_path='./figures/csvs_acc_figs/'+csv_file[:-4]+ '/' + str(round(now_acc, 4)) + '.png')
            save_figure(train_epochs_loss, valid_epochs_loss,test_losses,
                        save_path='./figures/csvs_loss_figs/'+csv_file[:-4]+ '/' + str(round(now_acc, 4)) + '.png')
            save_checkpoint(model, optimizer, hidden_size=args.hidden_size,
                            filepath='./best_model/csvs/'+csv_file[:-4]+ '/' + str(round(now_acc, 4)) + '.pth')

    pre_acc = 0
    best_model_files = os.listdir('./best_model/total/')
    best_model_files.sort()
    now_acc = np.average(test_total_acc)
    if len(best_model_files):
        pre_acc = float(best_model_files[-1][:-4])
    if now_acc > pre_acc:
        save_figure( train_total_acc, valid_total_acc,test_total_acc,
                    save_path='./figures/total_acc_figs/' + str(round(now_acc, 4))+'.png')
        save_figure( train_total_loss, valid_total_loss,test_total_loss,
                    save_path='./figures/total_loss_figs/' + str(round(now_acc, 4))+'.png')
        save_checkpoint(model, optimizer, hidden_size=args.hidden_size,
                        filepath='./best_model/total/' + str(round(now_acc, 4))+'.pth')



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
    epochs = 1
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
        hidden_size = [random.randint(250, 260), random.randint(120, 130)]
        args = argparse(hidden_size=hidden_size, input_size=input_size, output_size=output_size,
                        epochs=epochs,
                        original_lr=original_lr, lr_adjust=lr_adjust, patience=patience, cuda_id=cuda_id)
        train(args, csv_paths)
if __name__=='__main__':
    main()