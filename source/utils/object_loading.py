from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

from hydra.utils import instantiate
from source.datasets.asv_dataset import ASVDataset

def get_dataloaders(cfg):
    dataloaders = {}
    for split, params in cfg.items():
        num_workers = params.get("num_workers", 1)
        drop_last = False

        # create and join datasets
        datasets = []
        datasets.append(ASVDataset(params.datasets.data_dir, params.datasets.protocols_path))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=None,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
