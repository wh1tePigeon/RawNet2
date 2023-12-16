import argparse
import collections
import warnings

import numpy as np
import torch
import torch.nn as nn

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import source.metric as module_metric
import source.model as module_arch
from source.trainer import Trainer
from source.utils import prepare_device, get_logger
from source.utils.object_loading import get_dataloaders
from source.utils.parse_config import ConfigParser
from source.model.RawNet.rawnet import RawNet

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path='source/configs', config_name='main_config')
def main(cfg: DictConfig):
    logger = get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(cfg.data)

    # build model architecture, then print to console
    model = RawNet(cfg.arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg.n_gpu)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    weight = torch.tensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    metrics = [
        instantiate(metric_dict)
        for metric_dict in cfg.metrics
    ]


    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer, trainable_params)
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=cfg.trainer.get("len_epoch", None)
    )

    trainer.train()



if __name__ == "__main__":
    main()
