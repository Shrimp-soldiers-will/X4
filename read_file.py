import numpy as np
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy import linalg as la
from scipy.optimize import leastsq
import csv
import copy

def read_csv(file_name2):
    with open (file_name2,'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]
    data = np.array(rows)
    return data

def read_file(file_name):
    data1 = np.fromfile(file_name, dtype=np.uint)
    data2 = np.fromfile(file_name, dtype=np.float32)
    axis1 = (data1[1])
    column1 = data1[-(6 + 2 * axis1)] - data1[0] + 1
    header_mat1 = np.zeros((2, column1))
    header_mat2 = np.zeros((4, column1))
    sigI = np.zeros((axis1, column1), dtype=np.float32)
    sigQ = np.zeros((axis1, column1), dtype=np.float32)
    q = 0
    for ii in range(0, len(data1), 6 + 2 * axis1):
        header_mat1[:, q] = data1[ii:ii + 2]
        header_mat2[:, q] = data2[ii + 2:ii + 6]
        sigI[:, q] = data2[ii + 6:ii + 6 + axis1]
        sigQ[:, q] = data2[ii + 6 + axis1:ii + 6 + 2 * axis1]
        q = q + 1
    FrameCounter = header_mat1[0, :]
    NumOfBins = header_mat1[1, 0]
    BinLen = header_mat2[0, 0]
    SamplinFrequency = header_mat2[1, 0]
    CarrierFrequency = header_mat2[2, 0]
    RangeOffset = header_mat2[3, 0]

    return FrameCounter, NumOfBins, BinLen, SamplinFrequency, CarrierFrequency, RangeOffset, sigI, sigQ


def average_filter(x,alpha,RangeAxis):
    # 自动显示动目标
    (b,a) = x.shape
    c = np.zeros((b,))
    y1 =[]
    for ii in range(0,a):
        t = x[:,ii]
        yy = t-c
        y1.append(yy)
        c = alpha * c + (1 - alpha) * x[:, ii]
    yk = np.asarray(y1)
    yk = yk.T
    return yk



def fir_bpf(x,fs,fs1,fp1,fp2,fs2,N,choice):
    #该函数采用blackman窗实现带通滤波
    #x为输入信号，fs，为采样频率
    #fs1,fp1分别为阻带上截至频率和通带上截至频率
    #fp2，fs2分别为通带下截止频率和阻带下截至频率
    #ps：输入时以上四个滤波参数按从大到小输入即可
    #20150615 by boat
    #求对应角频率
    if (choice=='bpf'):
        ws2 = fs2 * 2 / fs
        wp2 = fp2 * 2 / fs
        wp1 = fp1 * 2 / fs
        ws1 = fs1 * 2 / fs

#计算滤波器系数
        wc2 = (ws2 + wp2) / 2
        wc1 = (ws1 + wp1) / 2
        wp1 = [wc1, wc2]

        han_win = signal.firwin(N+1,[fp1,fp2],nyq=fs*0.5,pass_zero=False,window='hann',scale=False)
        han_win_fft = np.fft.fft(han_win)
        han_win1 = 20*np.log10(abs(han_win_fft))
        # plt.figure()
        # plt.subplot(211)
        # plt.plot(han_win)
        # plt.subplot(212)
        # plt.plot(han_win1)
        # plt.show()



        y = signal.lfilter(han_win,1.0,x)
        #具有线性相位，（滤波器阶数+1，截止频率，过渡区域的近似带宽，采用的窗，，采样频率）
        return y
    if (choice=='lpf'):
        ws1 = fs1 * 2 / fs
        wp1 = fp1 * 2 / fs
        B = min(ws1, wp1)
        wc1 = (ws1 + wp1) / 2
        [b,a] = signal.butter(N,wc1,btype='band')
        y = signal.lfilter(b,a,x)
        return y
def LMS(xn, M):
    # LMS
    len = np.size(xn)
    # M = 10
    mu = 0.02 # 步长

    e = np.zeros((1,len)) # 误差序列, e(k)表示第k次迭代时预期输出与实际输入的误差
    w = np.ones((M, 1)) / M # 每一行代表一个加权参量, 每一列代表 - 次迭代, 初始为0
    x = np.zeros((M, 1))
    y = copy.deepcopy(xn)

    for k in range(M,len+1):  # 第k次迭代

        x2 = xn[k::-1]
        x3 = x2[1:M+1]
        #x1 = np.reshape(x,[np.size(x),1])
        y_cash = np.dot(w.T,x3) # 滤波器的输出
        y[k-1] = np.dot(w.T,x3) # 滤波器的输出
        e_cash = xn[k-1] - y[k-1]
        e[:,k-1]=e_cash
        # 滤波器权值计算的迭代式
        #
        # w_cash2 =(np.dot(mu ,e[:,k-1])*x)
        # w_cash1 = np.repeat((np.dot(mu ,e[:,k-1])*x),M,axis=0)
        w_cash1 =np.reshape(mu *e_cash*x3,[M,1])
        #w_cash2 =  mu *e_cash*x3
        w = w + w_cash1
    yn = y
    return  yn










def remove_st(a, style):
    if (style is 'mean'):
        B = np.reshape(np.mean(a, 1), [np.size(a, 0), 1])
        B1 = np.tile(B, (1, np.size(a, 1)))
        X = a - B1
        return X
    elif (style is 'min'):
        B = np.reshape(np.min(a, 1), [np.size(a, 0), 1])
        B1 = np.tile(B, (1, np.size(a, 1)))
        X = a - B1
        return X


def MTI(x, n):
    r = np.size(x, 0)
    c = np.size(x, 1)
    if (n == 1):
        ze = np.zeros((r, 1))
        x1 = np.column_stack((ze, x))
        g = x1[:, 0:c]
        y = x - g
    else:
        y = MTI(x, n - 1)
    return y