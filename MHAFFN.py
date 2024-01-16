# FocusLab ZYB
# 2022/4/26 9:49

import numpy as np
import torch, gc
import torch.nn as nn
from utils_multisource import *
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time
from torch.nn.parameter import Parameter

#设备检测
# GPU limited
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class AFF(nn.Module):


    def __init__(self, channels=16, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)


        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_a, x_b, x_c):
        x_ab = x_a + x_b
        xl = self.local_att(x_ab)
        xg = self.global_att(x_ab)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo1 = x_a * wei + x_b * (1 - wei)

        x_abc = xo1 + x_c
        xl2 = self.local_att2(x_abc)
        xg2 = self.global_att2(x_abc)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo2 = xo1 * wei2 + x_c * (1 - wei)

        return (xo1+xo2)/2

class multi_CNN(nn.Module):
    def __init__(self, class_mum):
        super(multi_CNN, self).__init__()
        self.class_num = class_mum
        self.linearchannel = 16
        self.conv1d_a1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.conv1d_a2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.conv1d_a3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.conv1d_a4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.conv2d_b1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_b2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_b3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_b4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_c1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_c2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_c3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2d_c4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )

        self.fc_a = nn.Sequential(
            nn.LazyLinear(256*self.linearchannel),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.fc_b = nn.Sequential(
            nn.LazyLinear(256*self.linearchannel),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.fc_c = nn.Sequential(
            nn.LazyLinear(256*self.linearchannel),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.fc1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.fc2 = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.fc3 = nn.Sequential(
            nn.LazyLinear(self.class_num),
            nn.Softmax(dim=1),
        )






    def forward(self, input_a, input_b, input_c):
        AttentionFF = AFF()
        AttentionFF.to(device)

        x_a = self.conv1d_a1(input_a)
        x_a = self.conv1d_a2(x_a)
        x_a = self.conv1d_a3(x_a)
        x_a = self.conv1d_a4(x_a)
        x_a = x_a.view(x_a.shape[0], -1)
        x_a = self.fc_a(x_a)

        x_b = self.conv2d_b1(input_b)
        x_b = self.conv2d_b2(x_b)
        x_b = self.conv2d_b3(x_b)
        x_b = self.conv2d_b4(x_b)
        x_b = x_b.view(x_b.shape[0], -1)
        x_b = self.fc_b(x_b)
   
        x_c = self.conv2d_c1(input_c)
        x_c = self.conv2d_c2(x_c)
        x_c = self.conv2d_c3(x_c)
        x_c = self.conv2d_c4(x_c)
        x_c = x_c.view(x_c.shape[0], -1)
        x_c = self.fc_c(x_c)

        # x = torch.cat([x_a, x_b, x_c], 1)
        x_a = x_a.reshape([x_a.shape[0], self.linearchannel, -1])
        x_b = x_b.reshape([x_b.shape[0], self.linearchannel, -1])
        x_c = x_c.reshape([x_c.shape[0], self.linearchannel, -1])
        x = AttentionFF(x_a, x_b, x_c)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x0 = self.fc2(x)
        x = self.fc3(x0)


        self.x0 = x0
        return x

    def get_fea(self):
        return self.x0

def train_one_epoch(model, optimizer, loss_fn, training_loader, report_n):
    running_loss = 0.
    running_acc = 0.
    last_loss = 0.
    last_acc = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (x_a, x_b, x_c, y) in enumerate(training_loader):
        # Every data instance is an input + label pair
        input_a, input_b, input_c, labels = x_a, x_b, x_c, y    
        labels = labels.long().view(-1)  

        input_a = input_a.to(device)
        input_b = input_b.to(device)
        input_c = input_c.to(device)
        labels = labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model.forward(input_a, input_b, input_c)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        train_acc = accuracy(outputs, labels)

        # Gather data and report
        running_loss += loss.item()
        running_acc += train_acc.item()
        if i % report_n == (report_n-1):
            last_loss = running_loss / report_n # loss per batch
            last_acc = running_acc / report_n
            print('  batch {} loss: {} acc: {}'.format(i + 1, last_loss, last_acc))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            running_acc = 0.



    return last_loss, last_acc



if __name__ == "__main__":
    # path settings + parameter settings
    bsize = 32     # batch_size
    report_n = 30   # 训练x个batch后汇报一次
    EPOCHS = 500
    numclass = 14
    init_lr = 1e-4
    data_file_a = "../Datasets/...mat"  
    data_file_b = "../Datasets/...mat"  
    data_file_c = "../Datasets/...mat"  
    model_path = './Modelsave/...pth'  
    pic_path = './Modelsave/figure/...jpg' 
    tsne_path = './Modelsave/figure/...jpg'


    # 数据处理，导入，打包
    print('----------------Start Data Processing----------------')
    datapro = data_process()
    datapro.test_flag = False
    # 读取3个接收机数据
    datapro.datapath = data_file_a
    x_train_a, y_train, x_val_a, y_val, x_test_a, y_test = datapro.readdata_1d()
    print("datasets load successfully! "+"train shape:"+str(len(x_train_a))+" test shep:" + str(len(x_test_a)))
    datapro.datapath = data_file_b
    x_train_b, _, x_val_b, _, x_test_b, _ = datapro.readdata_2d()
    print("datasets load successfully! "+"train shape:"+str(len(x_train_b))+" test shep:" + str(len(x_test_b)))
    datapro.datapath = data_file_c
    x_train_c, _, x_val_c, _, x_test_c, _ = datapro.readdata_2d()
    print("datasets load successfully! "+"train shape:"+str(len(x_train_c))+" test shep:" + str(len(x_test_c)))
    # package the dataset
    train_ds = TensorDataset(torch.Tensor(x_train_a), torch.Tensor(x_train_b), torch.Tensor(x_train_c),
                             torch.Tensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=bsize, shuffle=True)
    val_ds = TensorDataset(torch.Tensor(x_val_a), torch.Tensor(x_val_b), torch.Tensor(x_val_c),
                           torch.Tensor(y_val))
    val_dl = DataLoader(val_ds, batch_size=bsize, shuffle=True)
    test_ds = TensorDataset(torch.Tensor(x_test_a), torch.Tensor(x_test_b), torch.Tensor(x_test_c),
                            torch.Tensor(y_test))
    test_dl = DataLoader(test_ds, batch_size=bsize, shuffle=False)




    # 网络
    model = multi_CNN(numclass)
    model = model.to(device)
    print(next(model.parameters()).device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optmizer = torch.optim.Adam(model.parameters(), lr=init_lr)


    # 训练过程
    def trainval():
        # Initializing in a separate cell so we can easily add more epochs to the same run
        epoch_number = 0

        best_vloss = 1_000_000.
        # writer = SummaryWriter(write_path)
        # 设置每50个epoch调整学习率，lr=0.1*lr
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optmizer, step_size=50, gamma=0.7)  # 设置学习率下降策略
        early_stopping = EarlyStopping(patience=100, verbose=True, delta=0.0001, path=model_path)  # 早停
        time_start = time.time()
        for epoch in range(EPOCHS):
            # 获取当前lr，新版本用 get_last_lr()函数，旧版本用get_lr()函数，具体看UserWarning
            print('EPOCH {}, lr {}:'.format(epoch_number + 1, scheduler_lr.get_last_lr()))
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)

            avg_loss, avg_acc = train_one_epoch(model=model,
                                                optimizer=optmizer,
                                                loss_fn=loss_fn,
                                                training_loader=train_dl,
                                                report_n=report_n)
            gc.collect()
            torch.cuda.empty_cache()
            # 调整学习率
            scheduler_lr.step()
            # We don't need gradients on to do reporting
            # 每个epoch train结束后，开始validation
            model.train(False)
            running_vloss = 0.0
            running_vacc = 0.0
            with torch.no_grad():
                for i, (vinput_a, vinput_b, vinput_c, vlabels) in enumerate(val_dl):
                    vlabels = vlabels.long().view(-1)  # 转换为维度为1的long类型
                    vinput_a = vinput_a.to(device)
                    vinput_b = vinput_b.to(device)
                    vinput_c = vinput_c.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = model.forward(vinput_a, vinput_b, vinput_c)
                    vloss = loss_fn(voutputs, vlabels)
                    vacc = accuracy(voutputs, vlabels)
                    running_vloss += vloss
                    running_vacc += vacc.item()

                avg_vloss = running_vloss / (i + 1)
                avg_vacc = running_vacc / (i + 1)
                print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
                print('Accuracy train {} valid {}'.format(avg_acc, avg_vacc))
                early_stopping(avg_vacc, model)
                if early_stopping.early_stop:
                    print("此时早停！")
                    break

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                # model_path = '../Modelsave_torch/figure/tx10cnn_t{}_e{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1
        time_end = time.time()  # train end
        trian_time = time_end - time_start
        print('Training time is: {0}'.format(trian_time))


    # 测试过程
    def test():
        print("---------------Test Stage----------------------")
        weight_optimized = torch.load(model_path)  # load 最优权重
        model.load_state_dict(weight_optimized)  # 最优权重赴给模型
        print("The best model weight loaded successfully")
        running_test_loss = 0.0
        running_test_acc = 0.0
        y_labels = []  # 原始label
        y_pre = []  # 输出结果
        y_feature = []  # tsne的特征
        time_start = time.time()
        model.train(False)
        with torch.no_grad():
            for i, (tinput_a, tinput_b, tinput_c, test_labels) in enumerate(test_dl):
                test_labels = test_labels.long().view(-1)  # 转换为维度为1的long类型
                tinput_a = tinput_a.to(device)
                tinput_b = tinput_b.to(device)
                tinput_c = tinput_c.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model.forward(tinput_a, tinput_b, tinput_c)
                test_fea = model.get_fea()  # 输出是提取到的特征
                test_loss = loss_fn(test_outputs, test_labels)
                test_acc = accuracy(test_outputs, test_labels)
                running_test_loss += test_loss
                running_test_acc += test_acc.item()
                # 拼接测试数据，准备混淆矩阵
                y_labels = np.append(y_labels, test_labels.cpu().numpy())
                test_pre = torch.argmax(test_outputs, dim=1)
                y_pre = np.append(y_pre, test_pre.cpu().numpy())
                # 拼接TSNE的特征
                if i == 0:
                    y_feature = test_fea.cpu().numpy()
                else:
                    y_feature = np.append(y_feature, test_fea.cpu().numpy(), axis=0)
            # 累计测试loss以及准确率
            avg_test_loss = running_test_loss / (i + 1)
            avg_test_acc = running_test_acc / (i + 1)

        print('Test Loss: {} Test Acc: {}'.format(avg_test_loss, avg_test_acc))
        time_end = time.time()
        test_time = time_end - time_start
        print('Testing time is: {0}'.format(test_time))

        # test and score
        scores(y_pre, y_labels)
        # 画图混淆矩阵
        cm_plot(y_labels, y_pre, pic=pic_path)
        plot_tsne(y_feature, y_labels, tsne_path)

    # trainval()
    test()





