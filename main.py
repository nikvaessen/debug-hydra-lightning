import argparse
import os.path
import time

import lightning
import torch


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.fc1(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default=2)
    parser.add_argument("--bug", action="store_true")
    args = parser.parse_args()

    fabric = lightning.Fabric(accelerator="gpu", devices=args.devices, strategy="ddp")

    fabric.launch()

    network = Network()
    network, opt = fabric.setup(network, torch.optim.Adam(network.parameters()))

    produce_bug = args.bug
    if os.path.exists("network.ckpt"):
        if produce_bug:
            ckpt = torch.load("network.ckpt")
        else:
            ckpt = fabric.load("network.ckpt")

        network.load_state_dict(ckpt["network"])
    else:
        if fabric.is_global_zero:
            fabric.save("network.ckpt", {"network": network.state_dict()})
            fabric.print("network.ckpt now exists, run this script again.")
        exit()

    while True:
        fabric.print("sleeping")
        time.sleep(1)


if __name__ == "__main__":
    main()
