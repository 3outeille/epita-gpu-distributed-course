import torch

from torch.nn.functional import gelu

X = torch.randn(4, 2, device="cuda", dtype=torch.float32)
W = torch.randn(2, 2, device="cuda", dtype=torch.float32)

W_0, W_1 = W.chunk(2, dim=1)
# Column linear
y_col_1 =  torch.cat([gelu(X @ W_0), gelu(X @ W_1)], dim=1)
y_col_2 = gelu(torch.cat([X @ W_0, X @ W_1], dim=1))

torch.testing.assert_close(y_col_1, y_col_2, rtol=1e-5, atol=1e-5) # All match

# Row linear
X_0, X_1 = X.chunk(2, dim=1)
W_0, W_1 = W.chunk(2, dim=0)
y_row_1 = gelu(X_0 @ W_0) +  gelu(X_1 @ W_1)
y_row_2 = gelu(X_0 @ W_0 + X_1 @ W_1)

torch.testing.assert_close(y_row_1, y_row_2, rtol=1e-5, atol=1e-5) # Mismatch