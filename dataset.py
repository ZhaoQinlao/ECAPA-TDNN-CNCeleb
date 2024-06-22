'''
Datasets
'''

import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import glob
from WPMFCC import WPMFCC


class CNCeleb(Dataset):
    def __init__(self, train_list, train_path, num_frames, augmentation, feature_extractor, musan_path,rir_path, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        self.augmentation = augmentation
        self.feature_extractor = feature_extractor
        if os.path.exists(train_list):
            print('load {}'.format(train_list))
            df = pd.read_csv(train_list)
            speaker_int_labels = []
            utt_paths = []
            for (utt_path, label) in zip(df["utt_paths"].values, df["utt_spk_int_labels"].values):
                if utt_path[-4:]=='flac' or  utt_path[-3:]=='wav':
                    utt_paths.append(utt_path)
                    speaker_int_labels.append(label)
        else:
            utt_tuples, speakers = findAllUtt(train_path, extension='flac', speaker_level=1)
            utt_tuples = np.array(utt_tuples, dtype=str)
            utt_paths = utt_tuples.T[0]
            speaker_int_labels = utt_tuples.T[1].astype(int)
            speaker_str_labels = []
            for i in speaker_int_labels:
                speaker_str_labels.append(speakers[i])

            csv_dict = {"speaker_str_label": speaker_str_labels,
                        "utt_paths": utt_paths,
                        "utt_spk_int_labels": speaker_int_labels
                        }
            df = pd.DataFrame(data=csv_dict)
            try:
                df.to_csv(train_list)
                print(f'Saved data list file at {train_list}')
            except OSError as err:
                print(f'Ran in an error while saving {train_list}: {err}')

        # Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

        # Load data & labels
        self.data_list = utt_paths
        self.data_label = speaker_int_labels
        self.n_class = len(np.unique(self.data_label))
        print("find {} speakers".format(self.n_class))
        print("find {} utterance".format(len(self.data_list)))

    def __getitem__(self, index):
        audio, sr = sf.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        # Data Augmentation
        if self.augmentation:
            augtype = random.randint(0,5)
            if augtype == 0:   # Original
                audio = audio
            elif augtype == 1: # Reverberation
                audio = self.add_rev(audio)
            elif augtype == 2: # Babble
                audio = self.add_noise(audio, 'speech')
            elif augtype == 3: # Music
                audio = self.add_noise(audio, 'music')
            elif augtype == 4: # Noise
                audio = self.add_noise(audio, 'noise')
            elif augtype == 5: # Television noise
                audio = self.add_noise(audio, 'speech')
                audio = self.add_noise(audio, 'music')

        # WPMFCC
        if self.feature_extractor == 'WPMFCC':
            audio = WPMFCC(audio, 16000, 80, 400, 160, 3, 'db7', 512, 80)
            #audio = WPMFCC(audio[0], 16000, 12, 400, 200, 3, 'db7')

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
    
    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = sf.read(rir_file)
        rir         = np.expand_dims(rir.astype(float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]
    
    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = sf.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio],axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio


def findAllUtt(dirName, extension='flac', speaker_level=1):
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)

    # speaker_dict:{speaker_str_label:speaker_int_label}
    # utt_tuple:(utt_path,speaker_int_label)
    speaker_dict = {}
    utt_tuples = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speaker_str_label = root[prefixSize:].split(os.sep)[0]
            if speaker_str_label not in speaker_dict.keys():
                speaker_dict[speaker_str_label] = len(speaker_dict)
            speaker_int_label = speaker_dict[speaker_str_label]
            for filename in filtered_files:
                utt_path = os.path.join(root, filename)
                utt_tuples.append((utt_path, speaker_int_label))

    outSpeakers = [None]*len(speaker_dict)
    for key, index in speaker_dict.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(utt_tuples)))

    # return [(utt_path:speaker_int_label), ...], [id00012, id00031, ...]
    return utt_tuples, outSpeakers


def create_cnceleb_trails(cnceleb_root, trails_path, extension='flac'):
    enroll_lst_path = os.path.join(cnceleb_root, "eval/lists/enroll.lst")
    raw_trl_path = os.path.join(cnceleb_root, "eval/lists/trials.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in tqdm(enroll_lst, desc='speaker mapping', mininterval=2, ncols=50):
        path = os.path.splitext(item[1])
        spk2wav_mapping[item[0]] = path[0] + '.{}'.format(extension)
    trials = np.loadtxt(raw_trl_path, str)

    with open(trails_path, "w") as f:
        for item in tqdm(trials, desc='handle trials', mininterval=2, ncols=50):
            enroll_path = os.path.join(cnceleb_root, "eval", spk2wav_mapping[item[0]])
            test_path = os.path.join(cnceleb_root, "eval", item[1])
            test_path = os.path.splitext(test_path)[0] + '.{}'.format(extension)
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))


def test():
    cn1_root = 'CN-Celeb/CN-Celeb_flac'
    cn2_dev = 'CN-Celeb/CN-Celeb2_flac/data'
    musan_path = "augmented_data/musan_split"
    rirs_path = "augmented_data/RIRS_NOISES/simulated_rirs"
    train_list_path = 'augmented_data/cn2_train_list.csv'
    dataset = CNCeleb(train_list_path, cn1_root, 200, False, 'Fbank', musan_path, rirs_path)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for idx, batch in enumerate(loader):
        data, label = batch
        print('data:', data.shape, data)
        print('label', label.shape, label)
        break

    if not os.path.exists(musan_path) or not os.path.exists(rirs_path):
        print('no musan data or rirs data, skip test augmentation')
    else:
        data = data.numpy()
        data = data[0]
        data = np.expand_dims(data, 0)
        print(data.shape)
        dataset.add_rev(data)
        dataset.add_noise(data, 'speech')
        dataset.add_noise(data, 'music')
        dataset.add_noise(data, 'noise')
        dataset.add_noise(data, 'speech')
        dataset.add_noise(data, 'music')
        print('test augmentation done')

    print('done')


if __name__ == "__main__":
    # cn1_root = '/home2/database/sre/CN-Celeb-2022/task1/cn_1'
    # cn2_dev = '/home2/database/sre/CN-Celeb-2022/task1/cn_2/data'
    # train_list_path = 'data/cn2_train_list.csv'
    # dataset = CNCeleb(train_list_path, cn1_root, 200)
    # loader = DataLoader(dataset, batch_size=5, shuffle=True)
    # for idx, batch in enumerate(loader):
    #     data, label = batch
    #     print('data:', data.shape, data)
    #     print('label', label.shape, label)
    #     break
    test()