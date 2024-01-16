# FocusLab ZYB
# 2022/4/14 20:10
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
import h5py
import math
from sklearn.manifold import TSNE
from sklearn.metrics import *

def addwgn(x, snr):   #输入为（sample,n,1）的复数信号，或者（sample,n,2）的IQ两通道实数信号
    if np.iscomplexobj(x):  # 判断是否复数对象
        y = np.zeros(shape=(np.append(x.shape, 2)), dtype=float)
        y[..., 0] = x.real
        y[..., 1] = x.imag
        addwgn(y, snr)
    # 处理IQ两通道信号
    sample_num = len(x)
    x_noise = x.copy()
    for i in range(sample_num):
        x_temp = x[i]
        signal_noise = np.zeros(shape=x_temp.shape, dtype=float)
        signal = x_temp[...,0] + 1j * x_temp[..., 1]
        signal_power = np.linalg.norm(signal) ** 2 / signal.size
        noise_power = signal_power / np.power(10, (snr / 10))
        noise = np.random.normal(loc=0.0, scale=noise_power, size=x_temp.shape)
        x_noise[i] = x_temp + noise
    return x_noise


def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵

    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(pic)
    plt.show()


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    acc = pred.eq(labels.data.view_as(pred)).sum()
    return acc/len(labels)

class NMSE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        bsize = len(predictions)
        predictions = predictions.view(bsize, -1)
        labels = labels.view(bsize, -1)
        error = predictions - labels
        fenzi = torch.sum(torch.pow(error, 2), dim=1)
        fenmu = torch.sum(torch.pow(labels, 2), dim=1)
        NMSE = torch.mean(fenzi / fenmu)
        return NMSE



class data_process():
    # 数据处理类
    # datapath需要自己定义，默认为空
    # norm_flag为归一化标志，默认True，归一化
    # test_flag为测试数据标志，为True时，每一类只读取50组样本
    def __init__(self):
        self.datapath = None
        self.testratio = 0.2
        self.valratio = 0.2
        self.norm_flag = True
        self.test_flag = False

    def split(self, X, Y, ratio):
        #split
        X1, X2, Y1, Y2 = [], [], [], []
        # X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=ratio, stratify=Y)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=424)
        for train_index, test_index in sss.split(X, Y):
            X1, X2 = X[train_index], X[test_index]
            Y1, Y2 = Y[train_index], Y[test_index]
        return X1, X2, Y1, Y2

    def noisemask(self, X, Y, ratio):
        # 抽取一定比例的样本做高斯白噪声
        xshape = X[0].shape
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio)
        for normal_index, noise_index in sss.split(X, Y):
            noiseshape = np.append(len(noise_index),xshape)
            noise_mask = np.random.normal(loc=0.0, scale=1, size=noiseshape)
            X[noise_index] = noise_mask

        return X, Y

    def readdata_1d(self):
        # read
        hdict = h5py.File(self.datapath, 'r')
        if self.test_flag:
            test_idx = []
            for i in range(0,len(np.transpose(hdict['label'])),8):
                test_idx.append(i)
            X_temp = np.transpose(hdict['data'][..., test_idx])
            Y = np.transpose(hdict['label'][..., test_idx])
        else:
            X_temp = np.transpose(hdict['data'][:])
            Y = np.transpose(hdict['label'][:])
        X = np.empty(shape=np.append(X_temp.shape, 2), dtype=float)
        X[..., 0] = X_temp['real']
        X[..., 1] = X_temp['imag']
        # 归一化
        if self.norm_flag:
            X_stand = X.copy()
            for i in range(len(X)):
                X_stand[i] = preprocessing.StandardScaler().fit_transform(X[i])
            X = X_stand.copy()
        X = np.transpose(X, [0, 2, 1])

        # 标签

        X_train, X_test, Y_train, Y_test = self.split(X, Y, ratio=self.testratio)
        X_train, X_val, Y_train, Y_val = self.split(X_train, Y_train, ratio=self.valratio)


        print(f"X train: {X_train.shape}, Y train: {Y_train.shape}")
        print(f"X test: {X_test.shape}, Y_test: {Y_test.shape}")

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def readdata_2d(self):
        # read
        hdict = h5py.File(self.datapath, 'r')
        if self.test_flag:
            test_idx = []
            for i in range(0,len(np.transpose(hdict['label'])),8):
                test_idx.append(i)
            X_temp = np.transpose(hdict['data'][..., test_idx])
            Y = np.transpose(hdict['label'][..., test_idx])
        else:
            X_temp = np.transpose(hdict['data'][:])
            Y = np.transpose(hdict['label'][:])
        X = np.empty(shape=np.append(X_temp.shape, 2), dtype=float)
        X[..., 0] = X_temp['real']
        X[..., 1] = X_temp['imag']
        # 归一化
        if self.norm_flag:
            ini_shape = X.shape
            X = X.reshape([len(X), -1, 2])
            X_stand = X.copy()
            for i in range(len(X)):
                X_stand[i] = preprocessing.StandardScaler().fit_transform(X[i])
            X = X_stand.copy()
            X = X.reshape(ini_shape)
        X = np.transpose(X, [0, 3, 1, 2])

        # 标签

        X_train, X_test, Y_train, Y_test = self.split(X, Y, ratio=self.testratio)
        X_train, X_val, Y_train, Y_val = self.split(X_train, Y_train, ratio=self.valratio)


        print(f"X train: {X_train.shape}, Y train: {Y_train.shape}")
        print(f"X test: {X_test.shape}, Y_test: {Y_test.shape}")

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation accuracy increase.'''
        if self.verbose:
            self.trace_func(
                f'Validation accuracy increased ({self.val_loss_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        # save_dict = {
        #     'model_state_dict': model.state_dict()
        # }
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_acc




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']#提前网络结构
    model.load_state_dict(checkpoint['model_state_dict'])#加载网络权重参数
    optimizer = torch.optim.Adam()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器参数
    for parameter in model.parameters():
        parameter.requires_grad=False
    model.eval()

    return model

def plot_tsne(features, labels, tsne_path):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='random')

    # class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    # latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features shape is:', tsne_features.shape)
    class_num = int(labels.max())+1   # 类别数



    cnames = {
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'beige': '#F5F5DC',
        'bisque': '#FFE4C4',
        'black': '#000000',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'floralwhite': '#FFFAF0',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'ghostwhite': '#F8F8FF',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'honeydew': '#F0FFF0',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'ivory': '#FFFFF0',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightcyan': '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen': '#90EE90',
        'lightgray': '#D3D3D3',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lightyellow': '#FFFFE0',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mintcream': '#F5FFFA',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navajowhite': '#FFDEAD',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#FAA460',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32'}
    # 随机生成color库
    np.random.seed(1024)
    color_tab = np.random.choice(list(cnames.values()), class_num, replace=False)

    fig, ax = plt.subplots()
    # for i in range(tsne_features.shape[0]):
    #     plt.text(tsne_features[i, 0], tsne_features[i, 1], str(labels[i]),
    #              color=plt.cm.Set1(labels[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    for i in range(tsne_features.shape[0]):
        plt.scatter(tsne_features[i, 0], tsne_features[i, 1],
                    s=5,
                    # c=color_tab[np.int(labels[i])].reshape(1, -1),
                    c=color_tab[int(labels[i])],
                    edgecolors='face'
                    )
    # 标号

    for i in range(class_num):
        idx = np.where(labels == i)
        tsen_feature_temp = tsne_features[idx]
        plt.text(tsen_feature_temp[..., 0].mean(), tsen_feature_temp[..., 1].mean(), str(i+1),
                 fontdict={'weight':'bold', 'size':15})

    # plt.grid()
    # plt.legend()
    plt.xticks([])
    plt.yticks([])
    # plt.title([])
    plt.savefig(tsne_path)
    plt.show()
    return fig

def scores(predict_label, target_label):
    target_data_list = target_label
    predict_label_list = predict_label
    acc = accuracy_score(target_data_list, predict_label_list)
    print("accuracy = ", acc)
    precision = precision_score(target_data_list, predict_label_list, average='macro')
    print("precision = ", precision)
    recall = recall_score(target_data_list, predict_label_list, average='macro')
    print("recall = ", recall)
    f1 = f1_score(target_data_list, predict_label_list, average='macro')
    print("f1 = ", f1)

    return acc, precision, recall, f1