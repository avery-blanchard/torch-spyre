import torch

torch.manual_seed(3)

M = 5
K = 64
N = 512
x = torch.rand(M, K, N, dtype=torch.float16)
i = torch.tensor([2, 4, 0], dtype=torch.int32)


# CPU reference
def kernel(x, i):
    return x[i].exp()


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
