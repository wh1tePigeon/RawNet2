import argparse
import json
import os
from pathlib import Path
import collections
import torch
import torchaudio
from tqdm import tqdm

import source.model as module_model
from source.trainer import Trainer
from source.utils import ROOT_PATH
from source.utils.object_loading import get_dataloaders
from source.utils.parse_config import ConfigParser
from source.model.RawNet.rawnet import RawNet

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "test_model" / "main.pth"


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = RawNet(config["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(args.resume))
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    data_dir_path = args.data_dir
    with torch.no_grad():
        #batch = 
        for audio_name in os.listdir(data_dir_path):
            path = os.path.join(data_dir_path, audio_name)
            audio, _ = torchaudio.load(path)
            audio = audio.squeeze()
            max = 64000
            while audio.shape[-1] < max:
                audio = audio.repeat(2)
            audio = audio[:max]

            pred = model(audio)["pred"]
            print(pred)
            


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="source/configs/config_kaggle.json",
        type=str,
        help="Path to config which was used for training",
    )
    args.add_argument(
        "-dd",
        "--data_dir",
        default="test_data",
        type=str,
        help="data path",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="test_model/main.pth",
        type=str,
        help="Path to checkpoint",
    )

    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)