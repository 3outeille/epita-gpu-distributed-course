"""
torchrun --nproc_per_node=2 column_parallel_forward_backward.py
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 column_parallel_forward_backward.py
"""
import torch
import torch.distributed as dist
import lovely_tensors as lt;lt.monkey_patch()

dist.init_process_group(backend="gloo")
device = torch.device("cpu")

local_rank = dist.get_rank()
print(f"Local rank: {local_rank}")

world_size = dist.get_world_size()

X = torch.arange(start=0, end=8, step=1, device=device, requires_grad=True, dtype=torch.float32).reshape(4, 2)
X.retain_grad()
X_ref = X.clone()
X_ref.retain_grad()
print(X)

W = torch.arange(start=10, end=50, step=10, device=device, requires_grad=True, dtype=torch.float32).reshape(2, 2).transpose(0, 1)
W.retain_grad()
W_ref = W.clone()
W_ref.retain_grad()
print(W)

W_col = W[:, local_rank].contiguous()

y_col = X @ W_col

Y = [torch.zeros_like(y_col, dtype=y_col.dtype, device=device) for _ in range(world_size)]
dist.all_gather(Y, y_col)

Y = torch.stack(Y, dim=1)

Y_ref = X_ref @ W_ref
Y_ref.retain_grad()
assert torch.equal(Y, Y_ref), "Matrix multiplication result is incorrect"
print("PASS Forward ✅")

# ==== Backward ====
# Ref
L_ref = Y_ref.sum()
L_ref.retain_grad()
L_ref.backward()

# compute backward manually

L = Y.sum()

# L.grad => torch.tensor(1.)
# dL/dY = Y.grad => torch.ones_like(X)
# dL/dX <=> X.grad
# dL/dW <=> W.grad
# dY/dW = X
# dY/dX = W

Y_grad = torch.ones_like(Y)
# dL/dX = dL/dY @ dY/dX = Y_grad @ W.T
X_grad = Y_grad @ W.T
# dL/dW = dL/dY @ dY/dW = Y_grad @ X.T
W_grad = Y_grad @ X.T

# all_reduce
assert torch.equal(X_grad, X_ref.grad), "X.grad is incorrect"
assert torch.equal(W_grad, W_ref.grad), "W.grad is incorrect"
print("PASS Backward✅")