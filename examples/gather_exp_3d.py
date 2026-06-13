import torch
import torch_spyre._inductor.propagate_named_dims as pnd

declare_tensor_dim = pnd.declare_tensor_dim
name_tensor_dims = pnd.name_tensor_dims

torch.manual_seed(3)

x = torch.rand(4, 64, 1024, dtype=torch.float16)
i = torch.randint(0, 4, (3,), dtype=torch.int32)


def kernel(x, i):
    return x[i, :].exp()


# CPU reference
ref = kernel(x, i)

# Device run
x_dev = x.to("spyre")
i_dev = i.to("spyre")


result = torch.compile(kernel)(x_dev, i_dev).cpu()

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
