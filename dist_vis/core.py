import torch
import torch.distributed as dist
import time
import json
import os
from functools import wraps

# Store original function
_orig_all_reduce = None

def _log_all_reduce(tensor, *args, **kwargs):
    """Wrapper for all_reduce that logs details."""
    rank = dist.get_rank()
    start_time = time.time()
    result = _orig_all_reduce(tensor, *args, **kwargs)  # Call original
    end_time = time.time()

    # Log details
    log_entry = {
        "call": "all_reduce",
        "rank": rank,
        "start_time": start_time,
        "end_time": end_time,
        "data_size": tensor.element_size() * tensor.nelement(),  # Bytes
        "shape": str(tensor.shape),  # Convert to string for JSON
    }

    # Ensure directory exists
    log_dir = "dist_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Write to per-rank log file
    log_file = os.path.join(log_dir, f"log_rank_{rank}.jsonl")
    with open(log_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")  # JSON Lines format

    return result

def enable(log_dir="dist_logs"):
    """Enable monkeypatching for distributed visualization."""
    global _orig_all_reduce
    if _orig_all_reduce is None:  # Only patch once
        _orig_all_reduce = dist.all_reduce
        dist.all_reduce = _log_all_reduce
    # Optionally override default log directory
    globals()["log_dir"] = log_dir
    print(f"Distributed visualization enabled. Logs will be saved to {log_dir}")

def disable():
    """Revert monkeypatching."""
    global _orig_all_reduce
    if _orig_all_reduce is not None:
        dist.all_reduce = _orig_all_reduce
        _orig_all_reduce = None
        print("Distributed visualization disabled")

def sanity_check_with_torch():
    print("✅ Sanity check with torch starting...")
    print("torch.__version__:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("Num GPUs:", torch.cuda.device_count())
    else:
        print("No CUDA available")
    
    # Check if distributed is initialized before trying to access distributed info
    is_initialized = dist.is_initialized()
    print("Distributed initialized:", is_initialized)
    
    if is_initialized:
        print("World size:", dist.get_world_size())
        print("Rank:", dist.get_rank())
    else:
        print("Distributed not initialized")

    print("✅ Sanity check with torch passed")



