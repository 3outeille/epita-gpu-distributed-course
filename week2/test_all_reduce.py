"""
torchrun --nproc_per_node=2 test_all_reduce.py
# To debug
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 test_all_reduce.py
"""
import torch
import torch.distributed as dist
from rich.traceback import install

# Enable pretty-printing of traceback
install()


dist.init_process_group(backend="gloo")

device = torch.device("cpu")
local_rank = dist.get_rank()
x = torch.arange(local_rank, local_rank+2, dtype=torch.int64, device=device)
print(f"x={x} of rank {local_rank}")

dist.barrier() # Si on est parano

dist.all_reduce(x, op=dist.ReduceOp.SUM)
assert torch.equal(x, torch.tensor([1, 3]))