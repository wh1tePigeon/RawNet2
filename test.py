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

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "test_model" / "main.pth"


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    data_dir_path = config["data_dir"]
    with torch.no_grad():
        for audio_name in os.listdir(data_dir_path):
            path = os.path.join(data_dir_path, audio_name)
            audio, _ = torchaudio.load(path)
            audio = audio.squeeze()
            max = 64000
            while audio.shape[-1] < max:
                audio = audio.repeat(2)
            audio = audio[:max]

            pred = model(audio)["pred"].detach().cpu().numpy()
            print(pred)
            #verdict = "bona-fide" if abs(pred[0]) > abs(pred[1]) else "spoofed"
            #logger.info(f"{audio_name}:\nverdict: {verdict}\t\tpredictions: {pred}\n")

        #logger.info("Testing has ended.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="source/configs/config_test.json",
        type=str,
        help="Config path used to test",
    )
    args.add_argument(
        "-dd",
        "--data_dir",
        default="/home/comp/as/RawNet2/test_data",
        type=str,
        help="data path",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="/home/comp/as/RawNet2/test_model/main.pth",
        type=str,
        help="Path to checkpoint",
    )

    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    #args = args.parse_args()

    config = ConfigParser.from_args(args)
    main(config)