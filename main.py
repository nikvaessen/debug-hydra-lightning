import time

import lightning
import hydra
import torch

from torch.utils.data import Dataset, DataLoader


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.fc1(x)


class Data(Dataset):
    def __len__(self):
        return 10_000

    def __getitem__(self, item):
        return torch.rand((1000,))


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg):
    fabric = lightning.Fabric(
        accelerator="gpu", devices=cfg.devices, num_nodes=cfg.num_nodes
    )
    fabric.launch()

    network = Network()
    network, opt = fabric.setup(network, torch.optim.Adam(network.parameters()))

    train_dataloader = fabric.setup_dataloaders(DataLoader(Data(), num_workers=4))

    # keep alive
    while True:
        for x in train_dataloader:
            y = network(x)
            y_target = torch.randint(0, 9, size=(1,), device=fabric.device)
            loss = torch.nn.functional.cross_entropy(y, y_target)
            print(loss)

            fabric.backward(loss)
            opt.step()


if __name__ == "__main__":
    main()
