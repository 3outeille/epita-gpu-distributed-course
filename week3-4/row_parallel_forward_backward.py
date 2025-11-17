"""
torchrun --nproc_per_node=2 row_parallel_forward_backward.py
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 row_parallel_forward_backward.py
"""
import torch
import torch.distributed as dist
import lovely_tensors as lt;lt.monkey_patch()

def split_tensor(tensor, dim):
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    return torch.chunk(tensor, world_size, dim)[local_rank].contiguous()

def row_parallel_linear_forward(X_row, W_row):
    Y = X_row @ W_row.t()
    dist.all_reduce(Y, op=dist.ReduceOp.SUM)
    return Y

def row_parallel_linear_backward(Y_grad, X_row, W_row, use_all_gather=True):
    """
    #Given: Y = X @ W
    # We get the derivatives: 
    dY/dX = W
    dY/dW = X

    # Applying chain rule:
    dL/dX = dL/dY @ dY/dX = dL/dY @ W
    dL/dW = dL/dY @ dY/dW = dL/dY @ X
    """
    # 1. Compute dL/dX_row = dL/dY @ dY/dX = dL/dY @ W_row
    X_local_grad = Y_grad @ W_row
    if use_all_gather:
        X_grad = [torch.zeros_like(X_local_grad) for _ in range(dist.get_world_size())]
        dist.all_gather(X_grad, X_local_grad)
        X_grad = torch.cat(X_grad, dim=1)
    else:
        X_grad = X_local_grad

    # 2. Compute dL/dW_row = dL/dY @ dY/dW = dL/dY @ X_row
    W_grad = Y_grad.t() @ X_row

    return X_grad, W_grad

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    device = torch.device("cpu")

    X_ref = torch.arange(4 * 2, device="cuda", dtype=torch.float32, requires_grad=True).reshape(4, 2)
    W_ref = torch.arange(1, 5, device="cuda", dtype=torch.float32, requires_grad=True).reshape(2, 2) * 10
    
    X_ref.retain_grad()
    W_ref.retain_grad()
    Y_ref = X_ref @ W_ref.t()
    L_ref = Y_ref.sum()
    L_ref.backward()

    # Row parallel
    X = X_ref.clone()
    W = W_ref.clone()
    Y = row_parallel_linear_forward(split_tensor(X, dim=1), split_tensor(W, dim=1))    
    torch.testing.assert_close(Y, Y_ref, rtol=1e-5, atol=1e-5)
    print("Both forward pass are matching ✅")

    Y_grad = torch.ones_like(Y)
    X_grad, W_grad = row_parallel_linear_backward(Y_grad, X, split_tensor(W, dim=1))

    torch.testing.assert_close(X_grad, X_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(W_grad, W_ref.grad, rtol=1e-5, atol=1e-5)
    print("Both backward pass are matching ✅")