'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


import math
import torch
import torchaudio

import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class SelfAttentionBranch(nn.Module):
    def __init__(self, C, num_heads, dropout=0.1):
        super(SelfAttentionBranch, self).__init__()

        E = C // 8
        self.layer_norm = nn.LayerNorm(C)

        # self.Wq = nn.Linear(C, C)
        # self.Wk = nn.Linear(C, C)
        # self.Wv = nn.Linear(C, C)
        self.Wq = nn.Conv1d(C, E, kernel_size=1)
        self.Wk = nn.Conv1d(C, E, kernel_size=1)
        self.Wv = nn.Conv1d(C, E, kernel_size=1)

        self.self_attention = nn.MultiheadAttention(E, num_heads, dropout=dropout)
        self.Wo = nn.Conv1d(E, C, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape: [batch_size, feature_dimension, sequence_length]
        # 转换为 [batch_size, sequence_length, feature_dimension] 适应LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        # 转换为 [batch_size, feature_dimension, sequence_length] 适应Conv1d
        x = x.permute(0, 2, 1)

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # 转换为 [sequence_length, batch_size, feature_dimension]
        q = q.permute(2, 0, 1)
        k = k.permute(2, 0, 1)
        v = v.permute(2, 0, 1)
        attn_output, _ = self.self_attention(q, k, v)
        # 转换回 [batch_size, feature_dimension, sequence_length]
        attn_output = attn_output.permute(1, 2, 0)
        attn_output = self.Wo(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output


class DepthWiseConvAndSE(nn.Module):
    def __init__(self, C):  # C大多数情况输入为1024
        super(DepthWiseConvAndSE, self).__init__()
        self.dw_conv = nn.Conv1d(2 * C, 2 * C, kernel_size=3, groups=2 * C, padding=1)
        self.fc = nn.Linear(2 * C, C)
        self.se_block = SEModule(2 * C)

    def forward(self, Y_A, Y_R):
        Y_C = torch.cat((Y_A, Y_R), dim=1)  # Y_A, Y_R维度均为[64, 1024, 202]，输出Y_C维度为[64, 2048, 202]
        Y_D = self.dw_conv(Y_C)  # 输出Y_D维度为[64, 2048, 202]
        Y_D = self.se_block(Y_D)  # 输出Y_D维度为[64, 2048, 202]
        Y_C = Y_C.permute(0, 2, 1)
        Y_D = Y_D.permute(0, 2, 1)
        Y_Merge = self.fc(Y_C + Y_D)
        Y_Merge = Y_Merge.permute(0, 2, 1)
        return Y_Merge


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class Bottle2neckB(nn.Module):

    def __init__(self, inplanes, planes, kernel_size1=None, kernel_size2=None, dilation=None, scale=8):
        super(Bottle2neckB, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        conv3s = []
        conv1s = []
        bn3s = []
        bn1s = []
        num_pad1 = math.floor(kernel_size1 / 2) * dilation
        num_pad2 = math.floor(kernel_size2 / 2) * dilation
        for i in range(self.nums):
            conv3s.append(nn.Conv1d(width, width, kernel_size=kernel_size1, dilation=dilation, padding=num_pad1))
            conv1s.append(nn.Conv1d(width, width, kernel_size=kernel_size2, dilation=dilation, padding=num_pad2))
            bn3s.append(nn.BatchNorm1d(width))
            bn1s.append(nn.BatchNorm1d(width))
        self.conv3s = nn.ModuleList(conv3s)
        self.conv1s = nn.ModuleList(conv1s)
        self.bn3s = nn.ModuleList(bn3s)
        self.bn1s = nn.ModuleList(bn1s)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp3 = spx[i]
                sp1 = spx[i]
            else:
                sp3 = sp3 + spx[i]
                sp1 = sp1 + spx[i]
            sp3 = self.conv3s[i](sp3)
            sp1 = self.conv1s[i](sp1)
            sp3 = self.relu(sp3)
            sp1 = self.relu(sp1)
            sp3 = self.bn3s[i](sp3)
            sp1 = self.bn1s[i](sp1)
            if i == 0:
                out = sp3 + sp1
            else:
                out = torch.cat((out, sp3 + sp1), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class Bottle2neckA(nn.Module):  # 很多重复，怕改了会崩，就直接复制粘贴了，欸嘿

    def __init__(self, inplanes, planes, kernel_size1=None, kernel_size2=None, dilation=None, scale=8):
        super(Bottle2neckA, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        conv3s = []
        conv1s = []
        bn3s = []
        bn1s = []
        num_pad1 = math.floor(kernel_size1 / 2) * dilation
        num_pad2 = math.floor(kernel_size2 / 2) * dilation
        for i in range(self.nums):
            conv3s.append(nn.Conv1d(width, width, kernel_size=kernel_size1, dilation=dilation, padding=num_pad1))
            conv1s.append(nn.Conv1d(width, width, kernel_size=kernel_size2, dilation=dilation, padding=num_pad2))
            bn3s.append(nn.BatchNorm1d(width))
            bn1s.append(nn.BatchNorm1d(width))
        self.conv3s = nn.ModuleList(conv3s)
        self.conv1s = nn.ModuleList(conv1s)
        self.bn3s = nn.ModuleList(bn3s)
        self.bn1s = nn.ModuleList(bn1s)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)
        self.attention_branch = SelfAttentionBranch(C=inplanes, num_heads=8, dropout=0.1)
        self.merge = DepthWiseConvAndSE(planes)

    def forward(self, x):
        residual = x
        out_R = self.conv1(x)
        out_R = self.relu(out_R)
        out_R = self.bn1(out_R)

        spx = torch.split(out_R, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp3 = spx[i]
                sp1 = spx[i]
            else:
                sp3 = sp3 + spx[i]
                sp1 = sp1 + spx[i]
            sp3 = self.conv3s[i](sp3)
            sp1 = self.conv1s[i](sp1)
            sp3 = self.relu(sp3)
            sp1 = self.relu(sp1)
            sp3 = self.bn3s[i](sp3)
            sp1 = self.bn1s[i](sp1)
            if i == 0:
                out_R = sp3 + sp1
            else:
                out_R = torch.cat((out_R, sp3 + sp1), 1)

        out_R = torch.cat((out_R, spx[self.nums]), 1)

        out_R = self.conv3(out_R)
        out_R = self.relu(out_R)
        out_R = self.bn3(out_R)

        out_R = self.se(out_R)
        out_A = self.attention_branch(x)
        out = self.merge(out_A, out_R)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class Query(nn.Module):
    def __init__(self, channels, embed_dim, num_heads, hidden_dim, num_layers=2):
        super(Query, self).__init__()
        print('Using backend Query')
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.randn(1, embed_dim, embed_dim))  # 调整 query 的维度

        self.linear_k = nn.Linear(channels, embed_dim)
        self.linear_v = nn.Linear(channels, embed_dim)

        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, 12)

    def forward(self, inputs):
        # inputs: [batch_size, channels, time_steps]
        batch_size = inputs.size(0)

        key = self.linear_k(inputs.permute(2, 0, 1))  # [time_steps, batch_size, embed_dim]
        value = self.linear_v(inputs.permute(2, 0, 1))  # [time_steps, batch_size, embed_dim]

        query = self.query.repeat(batch_size, 1, 1)  # [batch_size, query_size, embed_dim]
        query = query.permute(1, 0, 2)  # [query_size, batch_size, embed_dim]

        for mha, norm, ffn in zip(self.mha_layers, self.norm_layers, self.ffn_layers):
            attn_output, _ = mha(query, key, value)  # [query_size, batch_size, embed_dim]
            query = query + attn_output  # Residual connection
            query = norm(query.permute(1, 0, 2))  # [batch_size, query_size, embed_dim]
            query = query.permute(1, 0, 2)  # [query_size, batch_size, embed_dim]
            query = ffn(query.permute(1, 2, 0)).permute(2, 0, 1)  # [query_size, batch_size, embed_dim]

        output = self.output_layer(query.permute(1, 2, 0))  # [batch_size, 12, query_size]
        output = output.view(batch_size, -1)

        return output
    
class Query2(nn.Module):
    def __init__(self, channels, query_dim, embed_dim, num_heads, hidden_dim, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.query = nn.Parameter(torch.randn(1, query_dim, embed_dim))  # 调整 query 的维度
        
        self.linear_k = nn.Linear(channels, embed_dim)
        self.linear_v = nn.Linear(channels, embed_dim)
        
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(query_dim, 12)
        
    def forward(self, inputs):
        # inputs: [batch_size, channels, time_steps]
        batch_size = inputs.size(0)
        
        key = self.linear_k(inputs.permute(2, 0, 1))  # [time_steps, batch_size, embed_dim]
        value = self.linear_v(inputs.permute(2, 0, 1))  # [time_steps, batch_size, embed_dim]
        
        query = self.query.repeat(batch_size, 1, 1)  # [batch_size, query_size, embed_dim]
        query = query.permute(1, 0, 2)  # [query_size, batch_size, embed_dim]
        
        for mha, norm, ffn in zip(self.mha_layers, self.norm_layers, self.ffn_layers):
            attn_output, _ = mha(query, key, value)  # [query_size, batch_size, embed_dim]
            query = query + attn_output  # Residual connection
            query = norm(query.permute(1, 0, 2))  # [batch_size, query_size, embed_dim]
            query = query.permute(1, 0, 2)  # [query_size, batch_size, embed_dim]
            query = query + ffn(query.permute(1,0,2)).permute(1, 0, 2)  # [query_size, batch_size, embed_dim]

        output = self.output_layer(query.permute(1, 2, 0))  # [batch_size, 12, query_size]
        output = output.view(batch_size, -1)
        
        return output


class ECAPA_TDNN(nn.Module):

    def __init__(self, C, feature_extractor, backend, link_method, backbone):
        super(ECAPA_TDNN, self).__init__()

        self.feature_extractor = feature_extractor
        self.backend = backend
        self.link_method = link_method
        self.backbone = backbone

        if self.feature_extractor == 'Fbank':
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20,
                                                    f_max=7600, window_fn=torch.hamming_window, n_mels=80), )

            self.specaug = FbankAug()  # Spec augmentation
            self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)

        elif self.feature_extractor == 'WPMFCC':
            self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)

        elif self.feature_extractor == 'MFCC':
            self.torchfbank = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, dct_type=2, norm='ortho')
            self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
           
        # 两个训练阶段：1）wavlm frozon 2）wavlm learning
        elif self.feature_extractor in ['wavlm1','wavlm2']:
            self.wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
            if self.feature_extractor == 'wavlm1':
                for name, layer in self.named_parameters():
                    if 'original' not in name:
                        layer.requires_grad_(False)
            self.conv1 = nn.Conv1d(768, C, kernel_size=5, stride=1, padding=2)
                
                
        else:
            raise Exception('Feature extractor name error, check your backbone name')

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)


        if self.backbone == 'Res2Block':
            self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
            self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
            self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        elif self.backbone == 'Res2BlockB':
            self.layer1 = Bottle2neckB(C, C, kernel_size1=3, kernel_size2=1, dilation=2, scale=8)
            self.layer2 = Bottle2neckB(C, C, kernel_size1=3, kernel_size2=1, dilation=3, scale=8)
            self.layer3 = Bottle2neckB(C, C, kernel_size1=3, kernel_size2=1, dilation=4, scale=8)
        elif self.backbone == 'Res2BlockA':
            self.layer1 = Bottle2neckA(C, C, kernel_size1=3, kernel_size2=1, dilation=2, scale=8)
            self.layer2 = Bottle2neckA(C, C, kernel_size1=3, kernel_size2=1, dilation=3, scale=8)
            self.layer3 = Bottle2neckA(C, C, kernel_size1=3, kernel_size2=1, dilation=4, scale=8)
        else:
            raise Exception('Backbone name error, check your backbone name')

        if self.link_method == 'GRU':
            self.gru1 = nn.GRU(input_size=C, hidden_size=C, num_layers=1, batch_first=False)
            self.gru2 = nn.GRU(input_size=C, hidden_size=C, num_layers=1, batch_first=False)

        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        # backend
        if self.backend == 'ASP':
            self.attention = nn.Sequential(
                nn.Conv1d(4608, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Tanh(),  # I add this layer
                nn.Conv1d(256, 1536, kernel_size=1),
                nn.Softmax(dim=2),
            )
        elif self.backend == 'Query':
            self.query = Query(channels=1536, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=2)
        elif self.backend == 'Query2':
            self.query = Query2(channels=1536, query_dim=16, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=2)
        else:
            raise Exception('Backend name error, check your backend name')

        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug):
        if self.feature_extractor == 'Fbank':
            with torch.no_grad():
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if aug == True:
                    x = self.specaug(x)

        elif self.feature_extractor == 'MFCC':
            x = self.torchfbank(x)

        elif self.feature_extractor == 'WPMFCC':
            x = x.permute(0,2,1)

        elif self.feature_extractor == 'wavlm1':
            x = self.wavlm(x).last_hidden_state.permute(0,2,1)
        elif self.feature_extractor == 'wavlm2':
            x = self.wavlm(x).last_hidden_state.permute(0,2,1)



        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        if self.link_method == 'Default':
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        elif self.link_method == 'Summed':
            x1 = self.layer1(x)
            x2 = self.layer2(x + x1)
            x3 = self.layer3(x + x1 + x2)
        elif self.link_method == 'GRU':
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

            x1 = x1.permute(2, 0, 1)
            x2 = x2.permute(2, 0, 1)
            x1, _ = self.gru1(x1)
            x2, _ = self.gru2(x2)
            x1 = x1.permute(1, 2, 0)
            x2 = x2.permute(1, 2, 0)
        else:
            raise Exception('link_method name error, check your link_method name')

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]  # batch channel time

        if self.backend == 'ASP':
            global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                                  torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)),
                                 dim=1)  # [batch, 3*channel, t]

            w = self.attention(global_x)
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
            x = torch.cat((mu, sg), 1)
        elif self.backend == 'Query':
            x = self.query(x)
        elif self.backend == 'Query2':
            x = self.query(x)

        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
