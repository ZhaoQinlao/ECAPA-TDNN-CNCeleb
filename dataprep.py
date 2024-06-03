import glob
import os
from scipy.io import wavfile

from argparse import ArgumentParser

def split_musan(args):

    files = glob.glob('%s/musan/*/*/*.wav'%args.save_path)

    audlen = 16000*5
    audstr = 16000*3

    for idx,file in enumerate(files):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0,len(aud)-audlen,audstr):
            wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])

        print(idx,file)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    args.save_path = 'augmented_data'
    split_musan(args)
