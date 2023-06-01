import time

import lightning
import hydra
import torch


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg):
    fabric = lightning.Fabric(
        accelerator="gpu", devices=cfg.devices, num_nodes=cfg.num_nodes
    )
    fabric.launch()

    # use gpu so process shows up in nvidia-smi
    tensor = torch.arange(0, 100_000, device=fabric.device)

    # keep alive
    while True:
        time.sleep(1)
