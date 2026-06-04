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

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

M, K, N = 128, 128, 128

mat_a = torch.rand(M, K, dtype=torch.float16) * 0.01
mat_b = torch.rand(K, N, dtype=torch.float16) * 0.01

mat_a_s = mat_a.to(DEVICE)
mat_b_s = mat_b.to(DEVICE)
scale_a = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)
scale_b = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)


def qfp8ch_scaled_mm(a, b, sa, sb):
    q_a = torch.ops.spyre.quantize_fp8_with_scale(a, sa)
    q_b = torch.ops.spyre.quantize_weight_fp8_with_scale(b, sb)
    out = torch.ops.aten._scaled_mm(
        q_a, q_b, sa, sb, bias=None, out_dtype=torch.float16
    )
    return out


compiled_mm = torch.compile(qfp8ch_scaled_mm)
spyre_result = compiled_mm(mat_a_s, mat_b_s, scale_a, scale_b).cpu()

cpu_result = (mat_a.to(torch.float8_e4m3fn) @ mat_b.to(torch.float8_e4m3fn)).to(
    torch.float16
)
max_delta = torch.abs(spyre_result - cpu_result).max()
print(f"spyre_result: {spyre_result}")
print(f"cpu_result: {cpu_result}")

print(f"Max delta Compiled Spyre vs. CPU: {max_delta}")
