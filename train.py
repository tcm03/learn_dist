import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from nnetwork import MLP
from data import CustomDataset
import logging

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(filename)s: %(lineno)s - %(funcName)s - %(levelname)s - %(message)s"
)

NUM_EPOCHS = 2

def gen_data(n: int) -> CustomDataset:
    ts_list = []
    labels = []
    for _ in range(n):
        ts_list.append(torch.randn(100))
        labels.append(torch.randint(high = 100, size = (1,)))
    return CustomDataset(ts_list, labels)

if __name__ == "__main__":

    torch.cuda.set_device(torch.cuda.current_device())
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend = "nccl", rank = rank, world_size = world_size)

    model = MLP()
    fsdp_model = FullyShardedDataParallel(
        model,
        device_id = torch.cuda.current_device(),
        auto_wrap_policy = size_based_auto_wrap_policy,
        cpu_offload = CPUOffload(offload_params = True)
    )
    optim = torch.optim.Adam(fsdp_model.parameters(), lr = 0.0001)
    custom_dataset = gen_data(100)
    dataloader = DataLoader(custom_dataset, batch_size = 10, shuffle = True)
    device = next(fsdp_model.parameters()).device
    logging.debug(f"device: {device}")
    for i in range(NUM_EPOCHS):
        print(f"Epoch {i+1}/{NUM_EPOCHS}")
        fsdp_model.train()
        for batch_idx, (ts_batch, label_batch) in enumerate(dataloader):
            print(f"Processing batch {batch_idx} ...")
            ts_batch = ts_batch.to(device)
            label_batch = label_batch.to(device)
            optim.zero_grad()
            logits = fsdp_model(ts_batch)
            logging.debug(f"logits.shape: {logits.shape}")
            logging.debug(f"logits.device: {logits.device}")
            logging.debug(f"label_batch.shape: {label_batch.shape}")
            logging.debug(f"label_batch.device: {label_batch.device}")
            loss = F.cross_entropy(logits, label_batch)
            print(f"loss = {loss.item()}")
            loss.backward()
            optim.step()
