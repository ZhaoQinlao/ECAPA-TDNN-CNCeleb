#this is a modified version of the original ECAPA model

import argparse
import glob
import datetime
import torch
import warnings

from torch.utils.data import DataLoader
import soundfile
from ECAPAModel import ECAPAModel
from dataset import CNCeleb
from tools import *

import torch
import soundfile
import sys
import time

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import AAMsoftmax
from model import ECAPA_TDNN
from tools import *
from dataset import create_cnceleb_trails



class DemoModel(ECAPAModel):
    def __init__(self, **kwargs):
        super(DemoModel, self).__init__(**kwargs)

    def embed(self,file) -> list[torch.Tensor]:
        '''
        Embeds the audio file into a speaker embedding
        Args:
            file: path to the audio file
        Returns:
            list[torch.Tensor]: embed1 and embed2
        '''
        self.eval()
        audio, _ = soundfile.read(file)
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(self.device)

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')

        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = numpy.stack(feats, axis=0).astype(float)

        data_2 = torch.FloatTensor(feats).to(self.device)
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings = [embedding_1, embedding_2]

        return embeddings
    

    def compare(self, file1, file2):
        embeddings1 = self.embed(file1)
        embeddings2 = self.embed(file2)
        embedding_11, embedding_12 = embeddings1
        embedding_21, embedding_22 = embeddings2
        # Compute the scores
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()
        return score