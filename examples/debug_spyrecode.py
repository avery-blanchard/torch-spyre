import torch
import torch_spyre._inductor.propagate_named_dims as pnd
from torch.spyre import SpyreTensorLayout, DataFormats

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(3)

x = torch.rand(12, 1024, dtype=torch.float16)
i = torch.tensor((0, 1, 0), dtype=torch.int32)


def kernel(x, i):
    return x[i]


# CPU reference
ref = kernel(x, i)

# Device run]
stl = SpyreTensorLayout(
    device_size=[12, 16, 64],
    stride_map=[1024, 64, 1],
    device_dtype=DataFormats.SEN169_FP16,
)
x_dev = x.to("spyre", device_layout=stl)
i_dev = i.to("spyre")


result = torch.compile(kernel)(x_dev, i_dev).cpu()
print("cpu", ref)
print("dev", result)
diff = torch.abs(ref - result)
print(f"max abs diff: {diff.amax().item()}")

torch.testing.assert_close(
    result,
    ref,
    equal_nan=True,
    atol=0.01,
    rtol=0.01,
    msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
)
print("PASSED")
