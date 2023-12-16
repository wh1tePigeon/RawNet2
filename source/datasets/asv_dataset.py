import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path
import collections
import torchaudio
from torch.utils.data import Dataset
from source.utils.util import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from source.utils.parse_config import ConfigParser
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

ASVMeta = collections.namedtuple('ASVMeta',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    def __init__(self, data_dir, protocols_path, config_parser: ConfigParser):
        self._data_dir = data_dir
        self.protocols_path = protocols_path

        self.data_meta = []

        for line in open(protocols_path).readlines():
            info = line.strip().split(' ')
            sys_id = 0
            if info[3] != '-':
                sys_id = 1
            key = 1
            if info[-1] == 'spoof':
                key = 0
            path = os.path.join(data_dir, info[1] + '.flac')
            self.data_meta.append(ASVMeta(speaker_id=info[0],
                                          file_name=info[1],
                                          path=path,
                                          sys_id=sys_id,
                                          key=key))


    def __getitem__(self, ind):
        path = self.data_meta[ind].path
        audio, _ = torchaudio.load(path)
        audio = audio.squeeze()
        max = 64000
        while audio.shape[-1] < max:
            audio = audio.repeat(2)
        audio = audio[:max]

        target = self.data_meta[ind].key
        meta = self.data_meta[ind]
        return {
            "audio": audio,
            "bonafied": target,
            "meta": meta
        }
    
    def __len__(self):
        return len(self.data_meta)
