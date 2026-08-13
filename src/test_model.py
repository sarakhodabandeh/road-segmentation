import torch

from .unet import UNet


# Use Apple GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


print("Device:", device)


# Create model
model = UNet(num_classes=11)
model = model.to(device)


# Create fake input
x = torch.randn(
    4,
    3,
    360,
    480,
    device=device
)


# Forward pass
with torch.no_grad():
    output = model(x)


print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)