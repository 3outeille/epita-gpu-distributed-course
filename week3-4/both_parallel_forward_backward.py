
"""
torchrun --nproc_per_node=2 both_parallel_forward_backward.py
debugpy-run -p 1234 -m torch.distributed.run -- --nproc_per_node=2 both_parallel_forward_backward.py
"""
import torch
import torch.distributed as dist
import lovely_tensors as lt;lt.monkey_patch()
from column_parallel_forward_backward import split_tensor, column_parallel_linear_forward, column_parallel_linear_backward
from row_parallel_forward_backward import row_parallel_linear_forward, row_parallel_linear_backward

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    device = torch.device("cpu")

    X_ref = torch.arange(4 * 2, device=device, dtype=torch.float32, requires_grad=True).reshape(4, 2)
    W_ref_layer1 = torch.arange(1, 5, device=device, dtype=torch.float32, requires_grad=True).reshape(2, 2) * 10
    W_ref_layer2 = torch.arange(1, 5, device=device, dtype=torch.float32, requires_grad=True).reshape(2, 2)

    X_ref.retain_grad()
    W_ref_layer1.retain_grad()
    W_ref_layer2.retain_grad()
    
    X = X_ref.clone()
    W_layer1 = W_ref_layer1.clone()
    W_layer2 = W_ref_layer2.clone()
    
    # Forward
    Y_ref_linear1 = X_ref @ W_ref_layer1.t()
    Y_ref_linear1.retain_grad()

    # We will transpose for matrix multiplication. As a result, we need to split row-wise
    Y_local_linear1 = column_parallel_linear_forward(X, split_tensor(W_layer1, dim=0))

    torch.testing.assert_close(Y_local_linear1, split_tensor(Y_ref_linear1, dim=1), rtol=1e-5, atol=1e-5)
    
    Y_local_linear2 = row_parallel_linear_forward(Y_local_linear1, split_tensor(W_ref_layer2, dim=1))
    Y_ref_linear2 = Y_ref_linear1 @ W_ref_layer2.t()
    torch.testing.assert_close(Y_local_linear2, Y_ref_linear2, rtol=1e-5, atol=1e-5)
    
    # Backward
    Y_ref_linear2.sum().backward()
    
    grad_Y = torch.ones_like(Y_ref_linear2)
    grad_X_linear2, grad_W_linear2 = row_parallel_linear_backward(grad_Y, Y_local_linear1, split_tensor(W_layer2, dim=1), use_all_gather=False)

    torch.testing.assert_close(grad_X_linear2, split_tensor(Y_ref_linear1.grad, dim=1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_W_linear2, split_tensor(W_ref_layer2.grad, dim=1), rtol=1e-5, atol=1e-5)

    grad_X, grad_W = column_parallel_linear_backward(grad_X_linear2, X, split_tensor(W_layer1, dim=0))

    torch.testing.assert_close(grad_X, X_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_W, split_tensor(W_ref_layer1.grad, dim=0), rtol=1e-5, atol=1e-5)

    print("Both forward and backward pass are matching ✅")