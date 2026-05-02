# -*- coding: utf-8 -*-

import torch
from set_transformer_1.model import SetTransformer

# 1. Create model
model = SetTransformer(
    dim_input=16,
    num_outputs=1,
    dim_output=10
)

# 2. Create a single input batch
# shape = (batch_size, set_size, dim_input)
X = torch.randn(1, 20, 16)   # 1 set, 20 elements, 16 features

# 3. Run once
output = model(X)

print(output.shape)