import os

os.environ["TMPDIR"] = "gradio/tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import argparse
import datetime
import torch
import warnings

from DemoModel import DemoModel
from tools import init_args

import gradio as gr




#################  你需要修改的一些路径  #################
cn1_root = "CN-Celeb/CN-Celeb_flac"
cn2_dev = "CN-Celeb/CN-Celeb2_flac/data"
train_list_path = "CN-Celeb/CN-Celeb2_flac/train_lst.csv"
trials_path = "CN-Celeb/CN-Celeb_flac/eval/lists/new_trials.lst"
save_path = "exps/exp1"
device = "cuda:0"
max_epoch = 80
batch_size = 64
eval_step = 5
initial_model = ""
######################################################

parser = argparse.ArgumentParser(description="ECAPA_trainer")

## 设置模型后端
parser.add_argument('--backend', type=str, default='ASP', help='选择模型后端，ASP或Query')

## 设置主干连接方式
parser.add_argument('--link_method', type=str, default='Summed', help='选择layer1、2、3、4的连接方式，Default/GRU/Summed')

## 设置主干部分使用的模型
parser.add_argument('--backbone', type=str, default='Res2Block',
                    help='设置主干部分使用的模型，Res2Block/Res2BlockB/Res2BlockA')

## Training Settings
parser.add_argument(
    "--num_frames", type=int, default=200, help="输入语音长度，200帧为2秒"
)
parser.add_argument("--max_epoch", type=int, default=max_epoch, help="训练多少个epoch")
parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size")
parser.add_argument("--n_cpu", type=int, default=4, help="DataLoader时使用多少核心")
parser.add_argument("--test_step", type=int, default=1, help="跑几个epoch测试一下性能")
parser.add_argument("--lr", type=float, default=0.001, help="学习率")
parser.add_argument("--lr_decay", type=float, default=0.9, help="学习率衰减率")
parser.add_argument("--device", type=str, default=device, help="训练设备")

## 训练、测试路径、模型保存路径
parser.add_argument("--train_list", type=str, default=train_list_path, help="训练列表")
parser.add_argument("--train_path", type=str, default=cn2_dev, help="训练数据路径")
parser.add_argument("--eval_list", type=str, default=trials_path, help="测试trails")
parser.add_argument("--eval_path", type=str, default=cn1_root, help="测试数据路径")
parser.add_argument("--save_path", type=str, default=save_path, help="模型保存路径")

## 设置embedding维度和margin loss超参数
parser.add_argument(
    "--C", type=int, default=1024, help="Channel size for the speaker encoder"
)
parser.add_argument("--m", type=float, default=0.2, help="Loss margin in AAM softmax")
parser.add_argument("--s", type=float, default=30, help="Loss scale in AAM softmax")
parser.add_argument("--n_class", type=int, help="Number of speakers")

## 运行模式
parser.add_argument("--eval", dest="eval", action="store_true", help="训练还是测试")
parser.add_argument(
    "--resume", dest="resume", action="store_true", help="是否恢复之前的训练"
)
parser.add_argument(
    "--initial_model", type=str, default=initial_model, help="从哪个模型继续"
)

train_start_time = datetime.datetime.now()
## 初始化、设置模型和打分文件保存路径
warnings.simplefilter("ignore")  # 忽略警告
torch.multiprocessing.set_sharing_strategy("file_system")
args = parser.parse_args()
args = init_args(args)



##---------------------------demo 从这里开始---------------------------------##

## demo用参数
args.initial_model = "exps/exp1/model/epoch_64_acc_88.pth" # 这里放预训练的模型
args.n_class = 1996  # 与训练用的cn-celeb2 speaker数量一致


## load the model
if gr.NO_RELOAD:  # 该部分仅初始化一次
    model = DemoModel(**vars(args))
    model.load_parameters(args.initial_model)
    print("Model {} 已加载!".format(args.initial_model))
    model.eval()


def detect_same_speaker(embed_file, eval_file, threshold=0.5):
    """
    Detects if the two audio files are from the same speaker
    Args:
        embed_file: path to the audio file to be
        eval_file: path to the audio file to be evaluated
    Returns:
        int: similarity score
    """
    score = model.compare(embed_file, eval_file)
    return float(score), score > threshold

def get_embeddings(file):
    return model.embed(file)

def detect_same_speaker_with_embeddings(embeddings1,embeddings2): 
    score = model.compare_with_embeddings(embeddings1, embeddings2)
    return float(score)

demo = gr.Interface(
    fn=detect_same_speaker,
    inputs=[
        gr.Audio(type="filepath", label="Embed File"),
        gr.Audio(type="filepath", label="Eval File"),
        gr.Slider(minimum=-1, maximum=1, value=0.5, label="Threshold"),
    ],
    outputs=[gr.Textbox(label="Similarity Score"), gr.Textbox(label="Same Speaker")],
)

demo.launch(show_error=True)
