import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import dist_vis  # Your library
from dist_vis import sanity_check_with_torch
import os


if __name__ == "__main__":

    # For single-node multi-GPU training, we can set these manually
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # If these are not set by a launcher, we'll set them for single-node testing
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'  # Main process is rank 0
    
    # Standard PyTorch distributed setup
    dist.init_process_group(backend="nccl")
    
    # Now it's safe to call sanity check
    sanity_check_with_torch()
    
    # Enable visualization (the one-liner!)
    dist_vis.enable()
    
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    model = torch.nn.Linear(10, 10).to(device)
    model = DDP(model)

    # Training code with all_reduce
    tensor = torch.randn(10).to(device)
    dist.all_reduce(tensor)
    print(f"Rank {rank}: {tensor}")

    dist.destroy_process_group()