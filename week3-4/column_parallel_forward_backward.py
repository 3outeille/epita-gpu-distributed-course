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
    Y_col = X @ W_col.t()
    return Y_col

def column_parallel_linear_backward(Y_col_grad, X, W_col):
    """

    #Given: Y = X @ W
    # We get the derivatives: 
    dY/dX = W
    dY/dW = X

    # Applying chain rule:
    dL/dX = dL/dY @ dY/dX = dL/dY @ W
    dL/dW = dL/dY @ dY/dW = dL/dY @ X
    """
    # 1. Compute dL/dX = dL/dY_col @ W_col
    #    Each rank computes partial gradient, then all_reduce to sum contributions
    X_grad = Y_col_grad @ W_col
    dist.all_reduce(X_grad, op=dist.ReduceOp.SUM)
    
    # 2. Compute dL/dW_col = dL/dY_col @ X
     #    Each rank computes gradient for its column slice
    W_col_grad = Y_col_grad.t() @ X
    
    return X_grad, W_col_grad

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    device = torch.device("cpu")

    # Baseline
    X_ref = torch.arange(4 * 2, device=device, dtype=torch.float32, requires_grad=True).reshape(4, 2)
    W_ref = torch.arange(1, 5, device=device, dtype=torch.float32, requires_grad=True).reshape(2, 2) * 10
    X_ref.retain_grad()
    W_ref.retain_grad()    
    Y_ref = X_ref @ W_ref.t()
    L_ref = Y_ref.sum()
    L_ref.backward()

    # Column parallel
    X = X_ref.clone()
    W = W_ref.clone()
    
    # We will transpose for matrix multiplication. As a result, we need to split row-wise
    Y_local = column_parallel_linear_forward(X, split_tensor(W, dim=0))
    
    torch.testing.assert_close(Y_local, split_tensor(Y_ref, dim=1), rtol=1e-5, atol=1e-5)
    print("Both forward pass are matching ✅")

    Y_local_grad = torch.ones_like(Y_local)
    
    X_grad, W_grad = column_parallel_linear_backward(Y_local_grad, X, split_tensor(W, dim=0))

    torch.testing.assert_close(X_grad, X_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(W_grad, split_tensor(W_ref.grad, dim=0), rtol=1e-5, atol=1e-5)
    
    print("Both backward pass are matching ✅")