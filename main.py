import time

import lightning
import hydra
import torch


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg):
    fabric = lightning.Fabric(devices=cfg.devices, num_nodes=cfg.num_nodes)
    fabric.launch()

    # use gpu so process shows up in nvidia-smi
    tensor = torch.tensor([0]).to(fabric.device)

    # keep alive
    while True:
        time.sleep(1)
