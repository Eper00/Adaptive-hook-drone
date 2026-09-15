import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "results/identification/adaptive_velocity_identification.npz"
MODEL_PATH = "results/identification/velocity_increment_model.pth"

DT = 1.0 / 48.0

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MLP
# ============================================================

class VelocityIncrementMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.Tanh(),

            nn.Linear(64, 3)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Load dataset
# ============================================================

print(f"Loading dataset: {DATASET_PATH}")

data = np.load(DATASET_PATH)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


# ============================================================
# Prepare target
# ============================================================

X_torch = torch.tensor(X)
Y_torch = torch.tensor(Y)

# ------------------------------------------------------------
# The network learns:
#
#     delta_v = v_next - v_current
#
# Therefore:
#
#     v_next = v_current + delta_v
# ------------------------------------------------------------

velocity = X_torch[:, 0:3]

delta_velocity = Y_torch - velocity


print("\nVelocity increment statistics:")

print(
    "Mean:",
    delta_velocity.mean(dim=0).numpy()
)

print(
    "Std:",
    delta_velocity.std(dim=0).numpy()
)

print(
    "Min:",
    delta_velocity.min(dim=0).values.numpy()
)

print(
    "Max:",
    delta_velocity.max(dim=0).values.numpy()
)


# ============================================================
# Normalize input
# ============================================================

X_mean = X_torch.mean(dim=0)
X_std = X_torch.std(dim=0)

# Prevent division by zero
X_std[X_std < 1e-6] = 1.0

X_normalized = (
    (X_torch - X_mean)
    / X_std
)


# ============================================================
# Dataset
# ============================================================

dataset = TensorDataset(
    X_normalized,
    delta_velocity
)

train_size = int(
    TRAIN_RATIO * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Create model
# ============================================================

input_dim = X.shape[1]

model = VelocityIncrementMLP(
    input_dim=input_dim
).to(DEVICE)


print("\nModel:")
print(model)

print(f"\nDevice: {DEVICE}")


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()

        # Predict velocity increment
        delta_prediction = model(X_batch)

        loss = criterion(
            delta_prediction,
            Y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * X_batch.size(0)
        )

    train_loss /= len(train_dataset)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            delta_prediction = model(X_batch)

            loss = criterion(
                delta_prediction,
                Y_batch
            )

            val_loss += (
                loss.item()
                * X_batch.size(0)
            )

    val_loss /= len(val_dataset)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train: {train_loss:.8f} | "
            f"Val: {val_loss:.8f}"
        )


# ============================================================
# Save model
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "input_dim":
        input_dim,

    "x_mean":
        X_mean.numpy(),

    "x_std":
        X_std.numpy(),

    "dt":
        DT,
}


torch.save(
    checkpoint,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)


