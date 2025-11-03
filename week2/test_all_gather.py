"""
torchrun --nproc_per_node=2 test_all_gather.py
# To debug
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 test_all_gather.py
"""
import torch 
import torch.distributed as dist
import os

from rich.traceback import install
install()  # Installe le gestionnaire de traceback Rich

global_rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# permet de lancer torchrun
dist.init_process_group(backend="gloo", rank=local_rank, world_size=world_size)

device = torch.device("cpu")
# local_rank = dist.get_rank()
assert(local_rank == dist.get_rank())
tensor = torch.arange(local_rank, local_rank+2, dtype=torch.int64, device=device)
print(f"Tensor of {local_rank} : {tensor}\n")

tensor_list = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor) # concat + broadcast to all processes in tensor_list

print(f"Tensor_list : {tensor_list}\n")