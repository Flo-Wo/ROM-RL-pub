import numpy as np
from pathlib import Path

# file to save initial condition and boundary condition for the navier
# stokes equation -> used for the env registry
path_to_target = Path(__file__).resolve().parent


def getInitialCondition(X):
    u = np.random.uniform(-5, 5) * np.ones_like(X)
    v = np.random.uniform(-5, 5) * np.ones_like(X)
    p = np.random.uniform(-5, 5) * np.ones_like(X)
    return u, v, p


def getActionRef():
    return 2.0 * np.ones(1000)


def get_desired_states():
    u_target = np.load(path_to_target / "target.npz")["u"]
    v_target = np.load(path_to_target / "target.npz")["v"]
    desired_states = np.stack([u_target, v_target], axis=-1)
    return desired_states


# Set up boundary conditions here
boundary_condition = {
    "upper": ["Controllable", "Dirchilet"],
    "lower": ["Dirchilet", "Dirchilet"],
    "left": ["Dirchilet", "Dirchilet"],
    "right": ["Dirchilet", "Dirchilet"],
}
