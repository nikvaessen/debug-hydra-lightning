import functools
import time

import lightning
import hydra
import torch

from torchdata.datapipes.iter import *

from torch.utils.data import DataLoader


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.fc1(x)


def get_sample(seed: int):
    rng = torch.random.manual_seed(seed)

    return torch.rand((1000,), generator=rng)


def construct_datapipe(shard_workers: bool = False):
    dp = IterableWrapper(range(0, 10_000))
    dp = Cycler(dp)
    dp = Shuffler(dp)

    if shard_workers:
        dp = ShardingFilter(dp)

    dp = Mapper(dp, get_sample)

    return dp


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg):
    fabric = lightning.Fabric(
        accelerator="gpu", devices=cfg.devices, num_nodes=cfg.num_nodes
    )
    fabric.launch()

    network = Network()
    network, opt = fabric.setup(network, torch.optim.Adam(network.parameters()))

    train_dataloader = fabric.setup_dataloaders(
        DataLoader(construct_datapipe(cfg.shard_workers), num_workers=cfg.num_workers)
    )

    # dataloader is infinite
    for x in train_dataloader:
        y = network(x)
        y_target = torch.randint(0, 9, size=(1,), device=fabric.device)
        loss = torch.nn.functional.cross_entropy(y, y_target)
        print(loss)

        fabric.backward(loss)
        opt.step()


if __name__ == "__main__":
    main()
