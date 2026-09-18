import numpy as np
import casadi as ca
import torch
import torch.nn as nn


class VelocityIncrementMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


def load_checkpoint(MODEL_PATH):
    """Load the trained PyTorch checkpoint."""
    print(f"\nLoading velocity increment MLP:\n{MODEL_PATH}")
    return torch.load(MODEL_PATH, map_location="cpu", weights_only=False)


def load_pytorch_model(checkpoint):
    """Create the PyTorch MLP and load its trained weights."""
    model = VelocityIncrementMLP(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_weights(model):
    """Extract Linear layer weights and biases."""
    weights = []
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy().astype(np.float64)
            b = layer.bias.detach().cpu().numpy().astype(np.float64)
            weights.append((W, b))
    return weights


def build_casadi_network(x, model, weights):
    """Build the neural network using CasADi operations."""
    h = x
    linear_idx = 0

    for layer in model.network:
        if isinstance(layer, nn.Linear):
            W, b = weights[linear_idx]
            h = ca.mtimes(ca.DM(W), h) + ca.DM(b)
            linear_idx += 1
        elif isinstance(layer, nn.Tanh):
            h = ca.tanh(h)

    return h


def create_casadi_mlp(checkpoint, model):
    """Create a CasADi function equivalent to the PyTorch MLP."""
    input_dim = checkpoint["input_dim"]
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float64)
    weights = extract_weights(model)

    x = ca.MX.sym("x", input_dim)
    x_normalized = (x - ca.DM(x_mean)) / ca.DM(x_std)
    delta_v = build_casadi_network(x_normalized, model, weights)

    return ca.Function(
        "velocity_increment_mlp",
        [x],
        [delta_v],
        ["x"],
        ["delta_v"]
    )


def load_casadi_model(MODEL_PATH):
    """Load the trained model and convert it to CasADi."""
    checkpoint = load_checkpoint(MODEL_PATH)
    model = load_pytorch_model(checkpoint)
    return create_casadi_mlp(checkpoint, model)


def test_models(MODEL_PATH):
    mlp_cas = load_casadi_model(MODEL_PATH)
    checkpoint = load_checkpoint(MODEL_PATH)
    model = load_pytorch_model(checkpoint)

    x = np.random.uniform(-0.5, 0.5, 9).astype(np.float32)

    # CasADi
    y_cas = np.array(
        mlp_cas(x.astype(np.float64))
    ).flatten()

    # PyTorch – normalize exactly as in CasADi
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    x_norm = (x - x_mean) / x_std

    with torch.no_grad():
        y_torch = model(torch.from_numpy(x_norm)).numpy().flatten()

    # Compare
    print("\nInput:")
    print(x)

    print("\nNormalized input:")
    print(x_norm)

    print("\nCasADi:")
    print(y_cas)

    print("\nPyTorch:")
    print(y_torch)

    print("\nDifference:")
    print(y_cas - y_torch)

    print("\nMax absolute difference:")
    print(np.max(np.abs(y_cas - y_torch)))
