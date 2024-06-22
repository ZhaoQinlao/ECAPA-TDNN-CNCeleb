# 介绍

此项目针对 CN-Celeb 数据集所开发，框架采用 `ECAPA-TDNN + AAM-Softmax`。

## 性能

|   训练数据   |    测试数据     | Augment | EER (%) | minDCF (0.01) | 阈值|
|:--------:|:-----------:|:-------:|:-------:|:-------------:|:---:|
| CN-2-dev | CN-1-trials |   No    |  11.7   |    0.4999     |待测试|
| CN-2-dev | CN-1-trials |   Yes   |  9.9   |    0.4276     |0.180|
| CN-2-dev | StarRail |   No   |  18.13   |    0.7229     |0.547|
| StarRail | StarRail |   No   |  0.578   |    0.0388     |0.168|

# Quick Start

## 准备工作

### 数据
* CN-Celeb 1  [[点此下载]](http://openslr.org/82/)
* CN-Celeb 2  [[点此下载]](http://openslr.org/82/)
> CN-Celeb 原始数据是 flac 格式，考虑到转换格式又要占磁盘空间，就直接读取 flac 格式进行训练了。

### 元数据准备
* 用build_datalist.py生成train.csv
* 数据集中提供的trial.lst以wav作为后缀，疑似存在问题，用dataset.py中的create_cnceleb_trails函数生成新的trial.lst

### 数据增广
* 下载rirs noises数据集 [[点此下载]](http://openslr.org/28/)
* 下载musan数据集 [[点此下载]](http://openslr.org/17/)
* 解压数据集到`augmented_data\`路径下
* 使用dataprep.py对rirs noise进行预处理
* 运行时添加`--augmentation`选项，如下
```
python trainECAPAModel.py --augmentation
```

### 环境

```
conda create -n cnceleb python=3.12.3
conda activate cnceleb
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```
> pip 安装不了可以试试下面其他的源
> * 清华：https://pypi.tuna.tsinghua.edu.cn/simple/
> * 阿里云：http://mirrors.aliyun.com/pypi/simple/
> * 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
> * 华中理工大学：http://pypi.hustunique.com/
> * 山东理工大学：http://pypi.sdutlinux.org/
> * 豆瓣：http://pypi.douban.com/simple/


## 训练
1. 在 `trainECAPAModel.py` 配置好对应路径
2. 激活 conda 环境 `conda activate cnceleb`
3. 运行 `python trainECAPAModel.py`
4. 提供backend，link_method, backbone等多个选项，详情见help

## 测试
1. 在主程序中设置定义 `initial_model` 路径
2. 运行`python trainECAPAModel.py --eval`

## 结果分析
使用增广后的测试结果保存于`score_label.pkl`中，可使用以下代码加载
```
with open('score_label.pkl', 'rb') as f:
    ids_dict, ids_true, ids_false, revues_dict, revues_true, revues_false = pickle.load(f)
```

## Demo
1. 从[release](https://github.com/ZhaoQinlao/ECAPA-TDNN-CNCeleb/releases)中下载预训练权重
2. 使用命令`gradio demo_with_gradio.py`启动脚本，在浏览器中打开对应链接

## Acknowledge

本项目基于 [PunkMale/ECAPA-TDNN-CNCeleb](https://github.com/PunkMale/ECAPA-TDNN-CNCeleb)和[TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) 修改，并参考了 [Lantian Li/Sunine](https://gitlab.com/csltstu/sunine)。
