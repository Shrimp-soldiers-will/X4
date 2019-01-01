import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import read_file
import math
import pickle
import pandas as pd
from statsmodels.tsa.stattools import acf
import copy



PRF =17
c=3e8

#csv_file = read_file.read_csv('xethru_sleep_20181219_170248.csv')

filename = 'xethru_baseband_iq_20181219_170248.dat'
FrameCounter, NumOfBins,BinLen,SamplingFrequency,CarrierFrequency,RangeOffset,sigI,\
sigQ = read_file.read_file(filename)

PowerDate = sigI**2+sigQ**2
AmpData = np.sqrt(PowerDate)
PhaseData = np.arctan(sigQ/sigI)
ComplexData = sigI+1j*sigQ
NumFrame = np.size(AmpData[0,:])


win_move = 20


## 估计呼吸频率
f_est = []                                     # 估计的呼吸频率
tt = []                                        # 当前时间
t_output = range(1,NumFrame)                        # 测试时间轴
globalState = []

# 初始化时间

N_win_initial = 25*PRF

# 快时间距离-多普勒矩阵
N_win_fast = 6*PRF
N_slow_FFt = 512
N_Fast_FFt = 128
rangeSlowMatrix_Fast = np.zeros((NumOfBins, N_Fast_FFt))
rangeDopplerMatrix_Fast = np.zeros((NumOfBins, N_Fast_FFt))
noiseRangeDopplerMap_Fast = np.zeros((NumOfBins, N_Fast_FFt))
alpha_fast = 0.6
# 慢时间距离-多普勒矩阵
N_win_Slow = 20*PRF
rangeSlowMatrix_Slow = np.zeros((NumOfBins, N_slow_FFt))
rangeDopplerMatrix_Slow = np.zeros((NumOfBins, N_slow_FFt))
noiseRangeDopplerMap_Slow = np.zeros((NumOfBins, N_slow_FFt))
alpha_slow = 0.85
# 状态配置存储单元
localStateSlow = np.zeros((NumOfBins, 1))
localStateFast = np.zeros((NumOfBins, 1))
globalState = np.zeros((NumOfBins, 1))
# 状态标志信息
initialization = -1
# 没运动
nomovement = 0
# 运动
movement = 1
# 呼吸
resperation = 2
# 记录当前记录的雷达数据帧数
clock_count = 0



