from __future__ import division
import glob  # 搜索路径
import time
import pywt
import python_speech_features
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import soundfile


def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


def dct(x):
    N = len(x)
    X = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for k in range(N):
        X[k] = np.sqrt(2 / N) * np.sum(C[k] * np.multiply(x, np.cos((2 * ts + 1) * k * np.pi / 2 / N)))
    return X


def idct(X):
    N = len(X)
    x = np.zeros(N)
    ts = np.array([i for i in range(N)])
    C = np.ones(N)
    C[0] = np.sqrt(2) / 2
    for n in range(N):
        x[n] = np.sqrt(2 / N) * np.sum(np.multiply(np.multiply(C[ts], X[ts]), np.cos((2 * n + 1) * np.pi * ts / 2 / N)))
    return x


def melbankm(p, NFFT, fs, fl, fh, w=None):
    """
    计算Mel滤波器组
    :param p: 滤波器个数
    :param n: 一帧FFT后的数据长度
    :param fs: 采样率
    :param fl: 最低频率
    :param fh: 最高频率
    :param w: 窗(没有加窗，无效参数)
    :return:
    """
    bl = 1125 * np.log(1 + fl / 700)  # 把 Hz 变成 Mel
    bh = 1125 * np.log(1 + fh / 700)
    B = bh - bl  # Mel带宽
    y = np.linspace(0, B, p + 2)  # 将梅尔刻度等间隔
    Fb = 700 * (np.exp(y / 1125) - 1)  # 把 Mel 变成Hz
    W2 = int(NFFT / 2 + 1)
    df = fs / NFFT
    freq = [int(i * df) for i in range(W2)]  # 采样频率值
    bank = np.zeros((p, W2))
    for k in range(1, p + 1):
        f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for i in range(1, W2):
            if n1 <= i <= n0:
                bank[k - 1, i] = (i - n1) / (n0 - n1)
            elif n0 < i <= n2:
                bank[k - 1, i] = (n2 - i) / (n2 - n0)
            elif i > n2:
                break
        # plt.plot(freq, bank[k - 1, :], 'r')
    # plt.savefig('images/mel.png')
    return bank


def greyList(n):
    """
    生成格雷编码序列
    参考：https://www.jb51.net/article/133575.htm
    :param n: 长度
    :return: 范围 2 ** n的格雷序列
    """

    def get_grace(list_grace, n):
        if n == 1:
            return list_grace
        list_before, list_after = [], []
        for i in range(len(list_grace)):
            list_before.append('0' + list_grace[i])
            list_after.append('1' + list_grace[-(i + 1)])
        return get_grace(list_before + list_after, n - 1)

    # get_grace生成的序列是二进制字符串，转化为10进制数
    return [int(i, 2) for i in get_grace(['0', '1'], n)]


def wavePacketRec(s, wname):
    """
    小波包重构
    :param s: 小波包分解系数，(a,d)间隔
    :param wname: 小波名
    :return: 各个分量的重构,注意这里重构的长度可能稍微大于原来的长度，如果要保证一样长，去除尾段的即可
    """
    out = []
    for i, l in enumerate(s):
        # 利用该数字去判断是a,d分量类型
        magic = i
        for j in range(int(np.log2(len(s)))):
            if magic % 2 == 0:
                # 为a分量时
                l = pywt.waverec([l, None], wname)
            else:
                # 为d分量时
                l = pywt.waverec([None, l], wname)
            magic = magic // 2
        out.append(l)

    grey_order = greyList(int(np.log2(len(s))))
    return np.array(out)[grey_order]


def wavePacketDec(s, jN, wname):
    """
    小波包分解，分解得到的是最后一层的结果，并非小波系数
    :param s: 源信号
    :param jN: 分解层数
    :param wname: 小波名
    :return:
        out: 小波包系数,是len为2**jN的list,以a,d相互间隔的输出
    """
    assert jN > 0 and isinstance(jN, int), 'Please take a positive integer as jN'
    out = []
    # 第一层分解
    a, d = pywt.dwt(s, wname)
    out.append(a)
    out.append(d)

    # 执行第2-jN次分解
    for level in range(1, jN):
        tmp = []
        for i in range(2 ** level):
            a, d = pywt.dwt(out[i], wname)
            tmp.append(a)
            tmp.append(d)
        out = tmp
    return out


#############################################################################################################
def WPMFCC(x, fs, p, frameSize, inc, jN, wname, nFFT=512, n_dct=12):
    """
    利用小波包-MFCC的方式提取特征，处理流程参考：
    陈静，基于小波包变换和MFCC的说话人识别特征参数，语音技术，2009，DOI:10.16311/j.audioe.2009.02.017
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :param jN: 小波包分解尺度
    :param wname: 小波包名
    :param nFFT: FFT点数
    :param n_dct: DCT阶数
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.9375], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hamming(frameSize))
    # 分层计算小波包能量谱线并合并
    enp = np.zeros((xx.shape[0], nFFT // 2 + 1))
    for i, l in enumerate(xx):
        # 进行小波包分解
        wpcoeff = wavePacketDec(l, jN, wname)
        wp_l = wavePacketRec(wpcoeff, wname)[:, :frameSize]
        # 计算谱线能量
        en = np.abs(np.fft.rfft(wp_l, nFFT)) ** 2 / nFFT
        # 谱线合并
        enp[i] = np.sum(en, axis=0)
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nFFT, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(enp, bank.T)
    # 计算DCT倒谱
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc


#############################################################################################################


def wavedec(s, jN, wname):
    ca, cd = [], []
    a = s
    for i in range(jN):
        a, d = pywt.dwt(a, wname)
        ca.append(a)
        cd.append(d)
    return ca, cd


############################################## 泥嚎  这里才正式开始#############################################
def getmfcc(file):  # fetch_index_label
    '''转换音乐文件格式并且提取其特征'''

    '''./data/music\\50 Cent - Ready For War.mp3'''

    items = file.split('.')
    file_format = items[-1].lower()  # 获取歌曲格式 mp3
    file_name = file[: -(len(file_format) + 1)]  # 获取歌曲名称
    # 把mp3格式的数据转化为wav

    try:
        '''提取wav格式歌曲特征'''

        fs, data, bits = wavfile.read(file)

        wmfcc = WPMFCC(data, fs, 12, 400, 200, 3, 'db7')
        print(wmfcc.shape)
        d_mfcc_feat = delta(wmfcc, 1)
        print('d_mfcc_feat.shape:', end='')
        print(d_mfcc_feat.shape)
        d_mfcc_feat2 = delta(wmfcc, 2)
        print('d_mfcc_feat2.shape:', end='')
        print(d_mfcc_feat2.shape)
        feature = np.hstack((wmfcc, d_mfcc_feat, d_mfcc_feat2))
        print('feature.shape:', end='')
        print(feature.shape)
        mm = np.transpose(feature)
        mf = np.mean(mm, axis=1)
        result = mf
        return result
    except Exception as msg:
        print(msg)


if __name__ == "__main__":
    """
    读取语音文件
    2020-2-26   Jie Y.  Init
    这里的wavfile.read()函数修改了里面的代码，返回项return fs, data 改为了return fs, data, bit_depth
    如果这里报错，可以将wavfile.read()修改。
    :param formater: 获取数据的格式，为sample时，数据为float32的，[-1,1]，同matlab同名函数. 否则为文件本身的数据格式
    指定formater为任意非sample字符串，则返回原始数据。
    :return: 语音数据data, 采样率fs，数据位数bits
    """
    data,sr = soundfile.read("CN-Celeb/CN-Celeb_flac/data/id00000/singing-01-001.flac")
    results = WPMFCC(data, sr, 12, 400, 200, 3, 'db7')
    print(results.shape, results)