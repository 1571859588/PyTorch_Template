# 简介

该`pytorch模板` 目录为一个整体，不能缺少任何一个模块

- 需要更改的部分：00，01
- 自动生成最佳模型及网络结构，并保存最佳模型到`best_model`中，并将过程中的训练和验证损失图像保存到`loss_fig` 中。文件名均以`准确率.pth` 或`准确率.png`命名



# 模板使用方法

只有00，01以数字开头的文件需要改，其他都是工具文件

执行顺序为数字顺序

## 目录结构

├─pytorch模板
│  │  00data_preprocess_template.py ：数据预处理
│  │  01pytorch_template.py ： 训练模型
│  │  readme.md
│  │  template_utils.py ： 工具函数
│  │
│  ├─00raw_data（不一定）
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─01preprocessed_data
│  │  ├─test_data
│  │  ├─train_data
│  │  └─valid_data
│  ├─best_model ： 保存准确率最高的模型
│  └─figures ： 准确率最高模型的每个被试的损失和准确率
│      ├─total_acc_figs
│      └─total_loss_figs

> cmd输入 `tree /f` 即可显示树形结构

## 各模块作用

#### 00data_preprocess_template.py

这个文件将`00raw_data`目录的原生数据集转为符合后续代码要求的数据集格式，即3个`csv`文件，分别为训练集、验证集和测试集，并将文件存放在`01preprocessed_data`目录里相应`train_data`，`valid_data` ，`test_data`目录

以上`csv`文件只有最后一列为标签，其余均为数据



#### 01pytorch_template.py

该文件为模型定义和训练代码

该文件使用已经经过预处理的`01preprocessed_data`的数据集进行训练，验证和测试，并将经测试得到最高模型的保存到`best_model`目录中，并将训练和验证损失记录到`loss_fig`目录中。文件名均以`准确率.pth` 或`准确率.png`命名

#### template_utils.py

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

#### 00raw_data

这里考虑到每个数据集的存放方式不一致，可以不用严格按照这里的目录结构放好

#### 01preprocessed_data

需要严格将训练集、验证集和测试集存放到相应的文件夹中，并且按序从0放好，以更好地对应



#### best_model 

以准确率命名的pth模型

#### figures 

以准确率命名的总损失和总准确率折线图，与上面的`best_model` 一一对应
