"""
torchrun --nproc_per_node=2 column_parallel_forward_backward.py
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 column_parallel_forward_backward.py
"""
import torch
import torch.distributed as dist
import lovely_tensors as lt;lt.monkey_patch()

def split_tensor(tensor, dim):
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    return torch.chunk(tensor, world_size, dim)[local_rank].contiguous()

def column_parallel_linear_forward(X, W_col):
    Y_col = X @ W_col
    return Y_col

def column_parallel_linear_backward(Y_col_grad, X, W_col):
    """
    Y = X @ W
    # Applying chain rule:
    dL/dX = dL/dY @ dY/dX
    dL/dW = dL/dY @ dY/dW

    dY/dX = W
    dY/dW = X

    dL/dX = dL/dY @ W
    dL/dW = dL/dY @ X
    """


    # 1. Compute dL/dX = dL/dY_col @ W_col.T
    #    Each rank computes partial gradient, then all_reduce to sum contributions
    X_grad = Y_col_grad @ W_col.T
    dist.all_reduce(X_grad, op=dist.ReduceOp.SUM)
    
    # 2. Compute dL/dW_col = X.T @ dL/dY_col
    #    Each rank computes gradient for its column slice
    W_col_grad = X.T @ Y_col_grad
    
    return X_grad, W_col_grad

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    device = torch.device("cpu")

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Baseline
    X_ref = torch.arange(start=0, end=8, step=1, device=device, requires_grad=True, dtype=torch.float32).reshape(4, 2)
    X_ref.retain_grad()
    W_ref = torch.arange(start=10, end=50, step=10, device=device, requires_grad=True, dtype=torch.float32).reshape(2, 2).transpose(0, 1)
    W_ref.retain_grad()
    Y_ref = X_ref @ W_ref
    Y_ref.retain_grad()

    L_ref = Y_ref.sum()
    L_ref.retain_grad()
    L_ref.backward()
    
    # In column parallel, X is replicated (not split), W is split along columns
    X = X_ref.clone()
    X.retain_grad()
    W = W_ref.clone()
    W.retain_grad()

    # Column parallel forward
    W_col = split_tensor(W, dim=1)
    Y_col = column_parallel_linear_forward(X, W_col)

    assert torch.equal(Y_col, split_tensor(Y_ref, dim=1)), "Matrix multiplication result is incorrect"
    print("Both forward pass are matching ✅")

    # Column parallel backward
    Y_col_grad = torch.ones_like(Y_col)  # dL/dY_col (gradient from loss)
    X_grad, W_col_grad = column_parallel_linear_backward(Y_col_grad, X, W_col)

    W_ref_col_grad = split_tensor(W_ref.grad, dim=1)

    assert torch.equal(X_grad, X_ref.grad), "X.grad is incorrect"
    assert torch.equal(W_col_grad, W_ref_col_grad), "W.grad is incorrect"
    print("Both backward pass are matching ✅")