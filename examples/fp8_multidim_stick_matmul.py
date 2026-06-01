# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.spyre
from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    spyre_empty_with_layout,
)
from torch_spyre._C import copy_tensor

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

M, K, N = 128, 4096, 1024

mat_a = torch.randn((M, K), dtype=torch.float16)
mat_b = torch.eye(K, N, dtype=torch.float16)

# Quantize mat_b to FP8 on CPU, then upload with the FP8_MULTI_DIM_STICK layout.
mat_b_fp8_cpu = mat_b.to(torch.float8_e4m3fn)

# Standard 3D STL with FP8_MULTI_DIM_STICK: device_size=[N/128, K, 128],
# stride_map=[128, K_stride, 1].  generate_dci expands this to the correct
# 4D DMA transfer layout internally.
eps = 128  # elems_per_stick for SEN143_FP8
fp8_stl = SpyreTensorLayout(
    [N // eps, K, eps],
    [eps, N, 1],
    DataFormats.SEN143_FP8,
    ElementArrangement.FP8_MULTI_DIM_STICK,
)

mat_b_dev = spyre_empty_with_layout(
    mat_b_fp8_cpu.size(),
    mat_b_fp8_cpu.stride(),
    torch.float8_e4m3fn,
    fp8_stl,
)
copy_tensor(mat_b_fp8_cpu, mat_b_dev, non_blocking=False)

mat_a_s = mat_a.to(DEVICE)
scale_a = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)
scale_b = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)


def qfp8ch_scaled_mm(a, b_fp8, sa, sb):
    q_a = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
    out = torch.ops.aten._scaled_mm(
        q_a, b_fp8, sa, sb, bias=None, out_dtype=torch.float16
    )
    return out


compiled_mm = torch.compile(qfp8ch_scaled_mm)
spyre_result = compiled_mm(mat_a_s, mat_b_dev, scale_a, scale_b).cpu()

cpu_result = mat_a @ mat_b

max_delta = torch.abs(spyre_result - cpu_result).max()

BLOCK_SIZE = 16

print(f"spyre_result dtype: {spyre_result.dtype}")
print(f"spyre_result shape: {spyre_result.shape}")


def format_block(tensor_slice):
    lines = []
    for row in tensor_slice.tolist():
        lines.append(" ".join(f"{val:8.4f}" for val in row))
    return "\n".join(lines)


for row_start in range(0, M, BLOCK_SIZE):
    for col_start in range(0, N, BLOCK_SIZE):
        row_end = row_start + BLOCK_SIZE
        col_end = col_start + BLOCK_SIZE

        print("-" * 78)
        print(
            f"BLOCK SUITE: Rows [{row_start}:{row_end}], Cols [{col_start}:{col_end}]"
        )
        print("-" * 78)
        print("")

        block_a = mat_a[row_start:row_end, :BLOCK_SIZE]
        block_b = mat_b[:BLOCK_SIZE, col_start:col_end]
        block_cpu = cpu_result[row_start:row_end, col_start:col_end]
        block_spyre = spyre_result[row_start:row_end, col_start:col_end]

        print(f"mat_a [{row_start}:{row_end}, :16]:")
        print(format_block(block_a))
        print("")

        print(f"mat_b [:16, {col_start}:{col_end}]:")
        print(format_block(block_b))
        print("")

        print(f"cpu_result [{row_start}:{row_end}, {col_start}:{col_end}]:")
        print(format_block(block_cpu))
        print("")

        print(f"spyre_result [{row_start}:{row_end}, {col_start}:{col_end}]:")
        print(format_block(block_spyre))
        print("\n\n")


print(f"Max delta Compiled Spyre vs. CPU: {max_delta}")
