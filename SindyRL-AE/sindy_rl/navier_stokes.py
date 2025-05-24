from pde_control_gym.src import NSReward
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from pde_control_gym.src.environments2d.navier_stokes2D import NavierStokes2D
from pathlib import Path
from sindy_rl.navierStokesData import (
    get_desired_states,
    getActionRef,
    getInitialCondition,
    boundary_condition,
)


# adapter pattern for the controlgym burgers PDE example to be used in SindyRL
class NavierStokesControlEnv(NavierStokes2D):
    def __new__(self, config=None):
        config = config or {}
        config["reset_init_condition_func"] = getInitialCondition
        config["boundary_condition"] = boundary_condition
        config["U_ref"] = get_desired_states()
        config["action_ref"] = getActionRef()
        config["reward_class"] = NSReward(0.1)
        config["action_dim"] = 1
        env_full = gym.make("PDEControlGym-NavierStokes2D", **config)
        return FlattenObservation(env_full)

    def reset(self, **kwargs):
        # return observation and info
        return super(NavierStokesControlEnv, self).reset(
            seed=kwargs.get("seed", None), options=kwargs.get("options", None)
        )
