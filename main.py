import pathlib

import wandb
import lightning
import hydra
import torch

from lightning.fabric.loggers import TensorBoardLogger
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


def construct_datapipe(shard_workers: bool = False, is_infinite: bool = False):
    dp = IterableWrapper(range(0, 10_000))

    if is_infinite:
        dp = Cycler(dp)

    dp = Shuffler(dp)

    if shard_workers:
        dp = ShardingFilter(dp)

    dp = Mapper(dp, get_sample)

    return dp


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg):
    # init wandb and tensorboard logger
    logger = TensorBoardLogger(pathlib.Path.cwd(), name="tensorboard", version="")

    fabric = lightning.Fabric(
        accelerator="gpu",
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        precision=cfg.precision,
        loggers=logger,
    )

    fabric.launch()

    if fabric.is_global_zero:
        tensorboard_dir = pathlib.Path.cwd() / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True, parents=True)
        wandb.tensorboard.patch(root_logdir=str(tensorboard_dir))
        wandb.init(project="debug")

    network = Network()
    network, opt = fabric.setup(network, torch.optim.Adam(network.parameters()))

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        DataLoader(
            construct_datapipe(cfg.shard_workers, is_infinite=True),
            num_workers=cfg.num_workers,
        ),
        DataLoader(
            construct_datapipe(cfg.shard_workers, is_infinite=False),
            num_workers=cfg.num_workers,
        ),
    )

    # dataloader is infinite
    train_iter = iter(train_dataloader)
    iteration = 0
    while True:
        iteration += 1

        if iteration % 100 == 0:
            val_loss_list = []
            for x in val_dataloader:
                with torch.no_grad():
                    loss = step(fabric, network, x)
                    val_loss_list.append(loss.cpu())
            fabric.log("val_loss", torch.mean(torch.tensor(val_loss_list)))

        x = next(train_iter)
        loss = step(fabric, network, x)

        fabric.log("loss", loss)
        fabric.backward(loss)
        opt.step()


def step(fabric, network, x):
    y = network(x)
    y_target = torch.randint(0, 9, size=(1,), device=fabric.device)
    loss = torch.nn.functional.cross_entropy(y, y_target)

    return loss


if __name__ == "__main__":
    main()
