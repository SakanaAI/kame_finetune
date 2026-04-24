import os

import torch


def set_mpi_env_vars():
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

    master_addr = os.environ.get("HOSTNAME", None)
    if master_addr is None:
        raise ValueError("HOSTNAME environment variable is not set")

    master_port = "29500"

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(world_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    torch.cuda.set_device(local_rank)  # set default device

    return world_size, world_rank, local_rank
