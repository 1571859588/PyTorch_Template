# 仓库简介

模板位于`00pytorch模板`目录中

其他都是实例，如`01pytorch模板实例1`

## 模板版本

目前版本：`1.0` 

## 目录结构

```
C:.
├─.idea
│  └─inspectionProfiles
├─pytorch模板
│  ├─00raw_data
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─01preprocessed_data
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─best_model
│  │  ├─csvs
│  │  └─total
│  ├─early_stop_model
│  └─figures
│      ├─csvs_acc_figs
│      ├─csvs_loss_figs
│      ├─total_acc_figs
│      └─total_loss_figs
└─pytorch模板实例1


```



## 贡献与更新

该模板由本人一人编写，仅用于开源，不可商用，商用必究

如有更好的意见，欢迎在`issues`提问，欢迎大家在`pull requests`更新模板



# 模板

## 简介

该`pytorch模板` 目录为一个整体，不能缺少任何一个模块

- 需要更改的部分：00，01
- 自动生成最佳模型及网络结构，并保存最佳模型到`best_model`中，并将过程中的训练和验证损失图像保存到`loss_fig` 中。文件名均以`准确率.pth` 或`准确率.png`命名



## 模板使用方法

只有00，01以数字开头的文件需要改，其他都是工具文件

执行顺序为数字顺序

### 目录结构

```
├─pytorch模板
│  │  00data_preprocess_template.py
│  │  01pytorch_template.py
│  │  readme.md
│  │  template_utils.py
│  │
│  ├─00raw_data
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─01preprocessed_data
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─best_model
│  │  ├─csvs
│  │  └─total
│  ├─early_stop_model
│  └─figures
│      ├─csvs_acc_figs
│      ├─csvs_loss_figs
│      ├─total_acc_figs
│      └─total_loss_figs
```



> cmd输入 `tree /f` 即可显示树形结构

### 各模块作用

#### 00data_preprocess_template.py 数据预处理

这个文件将`00raw_data`目录的原生数据集转为符合后续代码要求的数据集格式，即3个`csv`文件，分别为训练集、验证集和测试集，并将文件存放在`01preprocessed_data`目录里相应`train_data`，`valid_data` ，`test_data`目录

以上`csv`文件只有最后一列为标签，其余均为数据

#### 01pytorch_template.py 训练模型

该文件为模型定义和训练代码

该文件使用已经经过预处理的`01preprocessed_data`的数据集进行训练，验证和测试，并将经测试得到最高模型的保存到`best_model`目录中，并将训练和验证损失记录到`loss_fig`目录中。文件名均以`准确率.pth` 或`准确率.png`命名

#### template_utils.py 工具函数

```python
评估：evaluate(dataloader, model, device, flag, criterion, optimizer=None, epoch=None, epochs=None):
    返回平均损失，某次batch的平均准确率，测试准确率（数组）,model,optimizer  flag='valid' / 'test' / 'train'
自定义数据集：Dataset_name(flag='train', csv_paths=None))
超参数类：argparse(self, csv_paths, hidden_size=None, lr_adjust=None, input_size=30, output_size=12, epochs=30,
                 original_lr=0.001,
                 patience=4, cuda_id=0):仅argparse.device="cuda:cuda_id" 其余返回值都跟参数名称一样
早停类：EarlyStopping(patience=7, verbose=False, delta=0) verbose是否打印信息 其对象属性值early_stop为True时表示早停
保存模型（保存）：save_checkpoint(model, optimizer, epoch=None, filepath='./best_model/1.pth')
加载模型：load_checkpoint(filepath, model, optimizer, device):model, optimizer, epoch
画图（保存）：save_figure(train_loss, train_epochs_loss, valid_epochs_loss,save_path="./total_loss_figs/1.png")
```

#### 00raw_data （不一定）

这里考虑到每个数据集的存放方式不一致，可以不用严格按照这里的目录结构放好

#### 01preprocessed_data

需要严格将训练集、验证集和测试集存放到相应的文件夹中，并且按序从0放好，以更好地对应，且最后一列为标签

##### train_data

存放训练数据集

##### valid_data

存放验证集

##### test_data

存放测试集

#### best_model 

保存准确率最高的模型，以`准确率命名的pth`模型

##### csvs

该目录下会根据`01preprocessed_data` 里的文件保存对应训练集的最佳模型

##### total

该目录下会根据`01preprocessed_data` 里的文件保存对应训练集的最佳模型

#### early_stop_model 早停模型

记录下某个epoch最好的模型，文件名以`验证集平均准确率`为名字，与上面的`best_model` 不一样

#### figures 

以准确率命名的总损失和总准确率折线图，与上面的`best_model` 一一对应

##### csvs_acc_figs

该目录下会根据`01preprocessed_data` 里的文件保存对应的训练、验证、测试随着`epoch` 增加的准确率图像

##### csvs_loss_figs

该目录下会根据`01preprocessed_data` 里的文件保存对应的训练、验证、测试随着`epoch` 增加的损失图像

##### total_acc_figs

该目录下会根据`01preprocessed_data`  里所有文件保存对应的测试集随着`csvs` 增加，即随着被试训练后得到的准确率图像

##### total_loss_figs

该目录下会根据`01preprocessed_data`  里所有文件保存对应的测试集随着`csvs` 增加，即随着被试训练后得到的损失图像
