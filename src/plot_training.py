import matplotlib.pyplot as plt


epochs = list(range(1, 21))

train_loss = [
    1.2170,
    0.7226,
    0.6413,
    0.5620,
    0.5229,
    0.4834,
    0.4605,
    0.4444,
    0.4078,
    0.3929,
    0.3932,
    0.3564,
    0.3448,
    0.3352,
    0.3218,
    0.3120,
    0.3028,
    0.2820,
    0.2722,
    0.2842,
]

val_loss = [
    1.0106,
    0.6852,
    0.8041,
    0.5978,
    1.1782,
    0.4615,
    0.4806,
    0.4839,
    0.3916,
    0.4555,
    0.4410,
    0.3844,
    0.3736,
    0.3911,
    0.4688,
    0.3715,
    0.3888,
    0.2996,
    0.3647,
    0.3023,
]

val_miou = [
    0.2318,
    0.3339,
    0.2965,
    0.3751,
    0.2528,
    0.4103,
    0.4003,
    0.4097,
    0.4361,
    0.4195,
    0.4248,
    0.4657,
    0.4712,
    0.4817,
    0.4390,
    0.4763,
    0.4836,
    0.5141,
    0.5074,
    0.5274,
]


# -----------------------------------------
# Loss curve
# -----------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/training_loss.png",
    dpi=200
)

plt.close()


# -----------------------------------------
# Validation mIoU
# -----------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    val_miou,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Mean IoU")
plt.title("Validation Mean IoU")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/validation_miou.png",
    dpi=200
)

plt.close()


print("Training curves saved successfully!")
print("Saved: outputs/training_loss.png")
print("Saved: outputs/validation_miou.png")