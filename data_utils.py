import os
import random
from typing import Union, Optional
import librosa
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import seed_worker
from perturber import AudioPerturbation
def genCustom_list(data_path, fake=True):
    d_meta = {}
    data_list = []
    file_list = os.listdir(data_path)
    for line in file_list:
        key = os.path.join(data_path, line)
        data_list.append(key)
        if fake:
            d_meta[key] = 0
        else:
            d_meta[key] = 1
    real_database = './data/SONAR_dataset/real_samples/' 
    file_list = os.listdir(real_database) #
    file_list = random.sample(file_list, len(data_list))

    for line in file_list:
        key = os.path.join(real_database, line)
        data_list.append(key)
        d_meta[key] = 1

    return d_meta, data_list

def getASVSpoof2019_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    data_list0 = []
    data_list1 = []

    if is_train:
        with open(os.path.join(dir_meta,'ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.train.trn.txt'), "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            key = os.path.join(dir_meta, 'ASVspoof2019_LA_train', 'flac', key + '.flac')
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        with open(os.path.join(dir_meta,'ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.eval.trl.txt'), "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            key = os.path.join(dir_meta,'ASVspoof2019_LA_eval','flac',key+'.flac')
            if label == "bonafide":
                data_list1.append(key)
            elif label == "spoof":
                data_list0.append(key)
            else:
                raise ValueError

            d_meta[key] = 1 if label == "bonafide" else 0

        if len(data_list0) <= len(data_list1):
            data_list1 = random.sample(data_list1, len(data_list0))
        else:
            data_list0 = random.sample(data_list0, len(data_list1))
        file_list = data_list0 + data_list1
        return d_meta, file_list
    else:
        with open(os.path.join(dir_meta,'ASVspoof2019_LA_cm_protocols','ASVspoof2019.LA.cm.dev.trl.txt'), "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            key = os.path.join(dir_meta, 'ASVspoof2019_LA_dev', 'flac', key + '.flac')
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def get_custom_loader(seed: int,
                      batch_size: int,
                      dataset: str):

    if dataset == 'openai':
        database_path = './data/SONAR_dataset/OpenAI/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'flashspeech':
        database_path = './data/SONAR_dataset/FlashSpeech/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'voicebox':
        database_path = './data/SONAR_dataset/VoiceBox/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'xtts':
        database_path = './data/SONAR_dataset/xTTS/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'naturalspeech3':
        database_path = './data/SONAR_dataset/NaturalSpeech3/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'valle':
        database_path = './data/SONAR_dataset/VALLE/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'prompttts2':
        database_path = './data/SONAR_dataset/PromptTTS2/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'audiogen':
        database_path = './data/SONAR_dataset/AudioGen/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)
    elif dataset == 'seedtts':
        database_path = './data/SONAR_dataset/SeedTTS/'
        d_label_custom, file_custom = genCustom_list(database_path, fake=True)

    dataset = AudioDataset(list_IDs=file_custom,
                               labels=d_label_custom,
                               transform=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)
    print("no. custom files:", len(file_custom))

    return data_loader


def get_in_the_wild_loader(database_path: str,
                           seed:int,
                           batch_size: int,
                           use_name=False,
                           names_list=None):

    import csv
    file = os.path.join(database_path, 'meta.csv')
    d_meta = {}
    file_list = []
    data_list0 = []
    data_list1 = []

    name_group = {}
    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            key, name, label = line
    
            if label == 'bona-fide':
                data_list1.append(os.path.join(database_path,key))
                d_meta[os.path.join(database_path, key)] = 1
            else:
                data_list0.append(os.path.join(database_path, key))
                d_meta[os.path.join(database_path, key)] = 0
            if name not in name_group:
                name_group[name] = []
            name_group[name].append(os.path.join(database_path,key))

    if use_name:
        data_list0 = []
        data_list1 = []

        for i in names_list:
            name_data = name_group[i]
            for j in name_data:
                if d_meta[j] == 1:
                    data_list1.append(j)
                else:
                    data_list0.append(j)



    if len(data_list0) <= len(data_list1):
        data_list1 = random.sample(data_list1,len(data_list0))
    else:
        data_list0 = random.sample(data_list0,len(data_list1))
    file_list = data_list0 + data_list1
    dataset = AudioDataset(list_IDs=file_list,
                           labels=d_meta,
                           transform=False)

    gen = torch.Generator()
    gen.manual_seed(seed)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)
    print("no. in-the-wild files:", len(file_list))

    return data_loader

def genWavefake_list(data_path, is_train=False, is_eval=False):
    d_meta = {}
    data_list0 = []
    data_list1 = []

    ## Get wavefake
    folders = ['ljspeech_melgan',
               'ljspeech_parallel_wavegan',
               'ljspeech_multi_band_melgan',
               'ljspeech_full_band_melgan',
               'ljspeech_waveglow',
               'ljspeech_hifiGAN']

    for i in range(len(folders)):
        file_list = os.listdir(os.path.join(data_path, folders[i]))
        if is_train:
            file_list = file_list[:int(0.7 * len(file_list))]
        elif is_eval:
            file_list = file_list[int(0.8 * len(file_list)):]
        else:
            file_list = file_list[int(0.7 * len(file_list)):int(0.8 * len(file_list))]
        # elif few_shot:
        #     file_list = file_list[i*20:(i+1)*20]
        for line in file_list:
            key = os.path.join(data_path, folders[i], line)
            data_list0.append(key)
            d_meta[key] = 0

    # Get LJSpeech
    real_datapath = './data/LJSpeech-1.1/wavs/'
    file_list = os.listdir(real_datapath)
    if is_train:
        file_list = file_list[:int(0.7 * len(file_list))]
    elif is_eval:
        file_list = file_list[int(0.8 * len(file_list)):]
    else:
        file_list = file_list[int(0.7 * len(file_list)):int(0.8 * len(file_list))]

    for line in file_list:
        key = os.path.join(real_datapath, line)
        data_list1.append(key)
        d_meta[key] = 1

    if is_train:

        random.shuffle(data_list0)
        data_list0 = data_list0[:len(data_list1)]
        data_list = data_list0 + data_list1
        random.shuffle(data_list)

    else:
        random.shuffle(data_list0)
        data_list0 = data_list0[:len(data_list1)]
        # Combine and shuffle the final dataset
        data_list = data_list0 + data_list1
        random.shuffle(data_list)

    return d_meta, data_list

def get_ASVSpoof2019_loader(database_path: str,
        seed: int,
        batch_size: int):

    d_label_trn, file_train = getASVSpoof2019_list(database_path, is_train=True, is_eval=False)
    print("no. training files:", len(file_train))

    train_set = AudioDataset(list_IDs=file_train,
                                labels=d_label_trn,
                                transform=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = getASVSpoof2019_list(database_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = AudioDataset(list_IDs=file_dev,
                                labels=d_label_dev,
                                transform=False)
    dev_loader = DataLoader(dev_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)


    d_label_eval, file_eval = getASVSpoof2019_list(database_path, is_train=False, is_eval=True)
    print("no. test files:", len(file_eval))

    eval_set = AudioDataset(list_IDs=file_eval,
                                labels=d_label_eval,
                                transform=False)
    eval_loader = DataLoader(eval_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader
def get_libri_loader(database_path: str,
        seed: int,
        batch_size: int):

    d_label_trn, file_train = genLibriSeVoc_list(database_path, is_train=True, is_eval=False)
    print("no. training files:", len(file_train))

    train_set = AudioDataset(list_IDs=file_train,
                                labels=d_label_trn,
                                transform=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = genLibriSeVoc_list(database_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = AudioDataset(list_IDs=file_dev,
                                labels=d_label_dev,
                                transform=False)
    dev_loader = DataLoader(dev_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)


    d_label_eval, file_eval = genLibriSeVoc_list(database_path, is_train=False, is_eval=True)
    print("no. test files:", len(file_eval))

    eval_set = AudioDataset(list_IDs=file_eval,
                                labels=d_label_eval,
                                transform=False)
    eval_loader = DataLoader(eval_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def genLibriSeVoc_list(data_path, is_train=False, is_eval=False):
    d_meta = {}
    data_list0 = []
    data_list1 = []
    data_list = []
    folders = ['diffwave',  'gt',  'melgan',  'parallel_wave_gan',  'wavegrad',  'wavenet',  'wavernn']
    if is_train:
        file = os.path.join(data_path, 'train.txt')
    elif is_eval:
        file = os.path.join(data_path, 'test.txt')
    else:
        file = os.path.join(data_path, 'dev.txt')

    for folder in folders:
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if folder == 'gt':
                    key = os.path.join(data_path, folder, line)
                    data_list1.append(key)
                    d_meta[key] = 1
                else:
                    line = line.replace(".wav", "_gen.wav")
                    key = os.path.join(data_path, folder, line)
                    data_list0.append(key)
                    d_meta[key] = 0

    if is_train:
        # random.shuffle(data_list0)
        # budget = 1
        # data_list = data_list0[:int(budget*len(data_list0))] + data_list1[:int(budget*len(data_list1))]
        # data_list0 = data_list0[:len(data_list1)]
        data_list = data_list0 + data_list1
        random.shuffle(data_list)

    else:
        random.shuffle(data_list0)
        data_list0 = data_list0[:len(data_list1)]  # Sampling to match label 1 count
        # Combine and shuffle the final dataset
        data_list = data_list0 + data_list1
        random.shuffle(data_list)

    return d_meta, data_list

def get_wavefake_loader(database_path: str,
                        seed: int,
                        batch_size: int,
                        pert_method: Optional[str] = None,
                        pert_level: Optional[Union[int, float]] = None):

    d_trn_label, trn_file = genWavefake_list(database_path, is_train=True)
    trn_set = AudioDataset(list_IDs=trn_file,
                           labels=d_trn_label,
                           transform=True)
    print("no. wavefake train files:", len(trn_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(trn_set,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=gen)

    d_dev_label, dev_file = genWavefake_list(database_path)
    dev_set = AudioDataset(list_IDs=dev_file,
                                labels=d_dev_label,
                                transform=False)

    print("no. wavefake dev files:", len(dev_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    dev_loader = DataLoader(dev_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)

    d_eval_label, eval_file = genWavefake_list(database_path, is_eval=True)
    eval_set = AudioDataset(list_IDs=eval_file,
                            labels=d_eval_label,
                            transform=False,
                            pert_method=pert_method,
                            pert_level=pert_level
                            )

    print("no. wavefake eval files:", len(eval_file))

    gen = torch.Generator()
    gen.manual_seed(seed)
    eval_loader = DataLoader(eval_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=gen)

    return train_loader, dev_loader, eval_loader

def pad(waveform, max_len=64000):

    waveform_shape = waveform.shape

    if len(waveform_shape) == 1:
        waveform = waveform.unsqueeze(0)

    channels, time_len = waveform.shape

    if time_len >= max_len:
        return waveform[:, :max_len]

    pad_length = max_len - time_len

    padded_waveform = F.pad(waveform, (0, pad_length))

    return padded_waveform

class AudioDataset(Dataset):
    def __init__(self, list_IDs, labels, sample_rate=16000, transform=False, pert_level=None, pert_method=None):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.cut = 64000
        self.transform = transform
        self.sample_rate = sample_rate
        self.pert_level = pert_level
        self.pert_method = pert_method
        self.perturber = AudioPerturbation(sample_rate=self.sample_rate)

    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        key = self.list_IDs[index]
        waveform, sr = torchaudio.load(str(key))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        if self.pert_method == "gaussian_noise":
            pert_waveform = self.perturber.gaussian_noise(waveform,self.pert_level)
        elif self.pert_method == "background_noise":
            pert_waveform = self.perturber.background_noise(waveform,self.pert_level)
        elif self.pert_method == "smooth":
            pert_waveform = self.perturber.smooth(waveform,self.pert_level)
        elif self.pert_method == "echo":
            pert_waveform = self.perturber.echo(waveform,self.pert_level)
        elif self.pert_method == "high_pass":
            pert_waveform = self.perturber.highpass(waveform,self.pert_level)
        elif self.pert_method == "low_pass":
            pert_waveform = self.perturber.lowpass(waveform,self.pert_level)
        elif self.pert_method == "pitch_shift":
            pert_waveform = self.perturber.pitch_shift(waveform,self.pert_level)
        elif self.pert_method == "time_stretch":
            pert_waveform, sr = self.perturber.time_stretch(waveform,self.pert_level)
        elif self.pert_method == "quantization":
            pert_waveform = self.perturber.quantization(waveform,self.pert_level)
        elif self.pert_method == "opus":
            pert_waveform = self.perturber.opus(waveform,self.pert_level*1000)
        elif self.pert_method == "mp3":
            pert_waveform = self.perturber.mp3(waveform,self.pert_level)
        elif self.pert_method == "encodec":
            pert_waveform = self.perturber.encodec(waveform,self.pert_level)
        else:
            pert_waveform = waveform

        pert_waveform = pad(pert_waveform, max_len=self.cut)
        y = self.labels[key]

        return pert_waveform.squeeze().numpy(), y, key