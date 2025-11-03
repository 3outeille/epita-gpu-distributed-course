import os
import datetime
import torch
import torch.distributed as dist

'''
To run the file, use:
    torchrun --nproc_per_node 2 main_send_recv.py                                                                   
    
When giving --nproc_per_node=N to torchrun, we get:
    LOCAL_RANK: 0..N-1
When giving --nodes=M to torchrun, we get:
    WORLD_SIZE: M * N
'''

# Same environment variables as above
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
backend = "gloo"
device = torch.device("cpu", local_rank)

# This block assumes a fresh run (as if this were a separate script)
dist.init_process_group(
    rank=global_rank,
    world_size=world_size,
    backend=backend,
)

# local_rank = dist.get_rank()
assert local_rank == global_rank, "LOCAL_RANK and RANK should match in this setup"

# Synchronous send / recv
if dist.get_rank() == 0:
    # Assumes world_size of 2.
    objects = ["foo", 12, {1: 2}] # any picklable object
else:
    objects = [None, None, None]

print(f"\nSend/Recv - Global Rank {global_rank} (Local Rank {local_rank}): sending/receiving objects -> {objects}")

if dist.get_rank() == 0:
    # Assumes world_size of 2.
    dist.send_object_list(objects, dst=1, device=device)
else:
    dist.recv_object_list(objects, src=0, device=device)
    print(f"\nSend/Recv - Global Rank {global_rank} (Local Rank {local_rank}): received objects -> {objects}")