from gymnasium.spaces.box import Box
from gymnasium import Env
import numpy as np
import torch
from copy import deepcopy


class BasePolicy:
    """Parent class for policies"""

    def __init__(self):
        raise NotImplementedError

    def compute_action(self, obs):
        """given observation, output action"""
        raise NotImplementedError


class FixedPolicy(BasePolicy):
    """
    Deterministic policy that provides feedforward control
    from a prescribed sequence of actions
    """

    def __init__(self, fixed_actions):
        self.fixed_actions = fixed_actions
        self.n_step = 0
        self.n_acts = len(fixed_actions)

    def compute_action(self, obs):
        u = self.fixed_actions[self.n_step % self.n_acts]
        self.n_step += 1
        return u


class RLlibPolicyWrapper(BasePolicy):
    """
    Wraps an RLlib algorithm into a BasePolicy class.

    Provide the same interface as the controlgym agents.
    """

    def __init__(self, algo, mode="algo"):
        self.algo = algo
        self.mode = mode

    def compute_action(self, obs, explore=False):
        res = self.algo.compute_single_action(obs, explore=explore)
        if self.mode == "policy":
            res = res[0]
        return res

    def run_navierStokes(self, env: Env, seed: int = None):
        torch.manual_seed(seed=seed)
        observation, info = env.reset(seed=seed)
        rewards_rollout = []

        n_steps = env.nt
        state_traj = np.zeros((observation.shape[0], n_steps + 1))
        action_traj = np.zeros(n_steps)
        state_traj[:, 0] = observation

        # CONTROLLER ACTIVATED
        for t in range(n_steps):
            action = self.compute_action(observation, explore=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            state_traj[:, t + 1] = deepcopy(observation)
            action_traj[t] = deepcopy(action)
            if terminated or truncated:
                break
            rewards_rollout.append(reward)
        env.state_traj = state_traj
        return rewards_rollout, action_traj

    def run(
        self,
        env: Env,
        state: np.ndarray[float] = None,
        seed: int = None,
        reset: bool = True,
        num_zero_controls: int = 0,
    ) -> tuple[float, float]:
        """Execute a full-sweep of the controller within the environment.

        Parameters
        ----------
        env : Env
            PDE-like environment, more general a gymnasium environment.
        state : np.ndarray[float], optional
            Initial state to start with, by default None. If None, initialized
            by the environment itself, during ``.reset()``.
        seed : int, optional
            Random seed, by default None.
        reset : bool, optional
            Reset the environment, if needed, before the controller starts.
            Default is true.
        num_zero_controls :  int, optional
            Number of controls with zero value to let the env evolve before the
            controller is activated. Default is zero.

        Returns
        -------
        float
            Total reward along the trajectory.
        """
        # reset the environment
        torch.manual_seed(seed=seed)
        observation, info = env.reset(seed=seed, state=state)
        # if reset:
        #     observation, info = env.reset(seed=seed, state=state)
        # observation = env.C @ env.state
        info = {"state": env.state}
        # run the simulated trajectory and calculate the h2 cost
        rewards_rollout = []
        projected_rewards_rollout = []

        state_traj = np.zeros((env.n_state, env.n_steps + num_zero_controls + 1))
        action_traj = np.zeros((env.n_action, env.n_steps + num_zero_controls))
        state_traj[:, 0] = info["state"]

        # LET THE ENV EVOLVE WITHOUT CONTROLS
        for t in range(num_zero_controls):
            action = np.zeros_like(self.compute_action(observation, explore=False))
            observation, reward, terminated, truncated, info = env.step(action)
            state_traj[:, t + 1] = deepcopy(info["state"])
            action_traj[:, t] = deepcopy(action)
            if terminated or truncated:
                break
            rewards_rollout.append(reward)
            projected_rewards_rollout.append(
                env.projected_reward(env.C @ info["state"], action)
            )

        # CONTROLLER ACTIVATED
        cache_n_steps = env.n_steps
        env.n_steps = cache_n_steps + num_zero_controls
        for t in range(env.n_steps):
            action = self.compute_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            state_traj[:, num_zero_controls + t + 1] = deepcopy(info["state"])
            action_traj[:, num_zero_controls + t] = deepcopy(action)
            if terminated or truncated:
                break
            rewards_rollout.append(reward)
            projected_rewards_rollout.append(
                env.projected_reward(env.C @ info["state"], action)
            )
        env.state_traj = state_traj
        env.n_steps = cache_n_steps
        return rewards_rollout, projected_rewards_rollout, action_traj

    def run_surrogate(
        self,
        env: Env,
        state: np.ndarray[float] = None,
        seed: int = None,
        n_steps: int = None,
    ) -> tuple[float, float]:
        """Same wrapper as above, but we only work with the partially observed states"""
        torch.manual_seed(seed=seed)
        observation, info = env.reset(seed=seed, state=state)
        # if reset:
        #     observation, info = env.reset(seed=seed, state=state)
        # observation = env.obs
        # run the simulated trajectory and calculate the h2 cost
        total_reward = []
        if not n_steps:
            n_steps = env.real_env.n_steps
        state_traj = np.zeros((env.obs_dim, n_steps + 1))
        action_traj = np.zeros((env.act_dim, n_steps))
        state_traj[:, 0] = observation
        for t in range(n_steps):
            action = self.compute_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            # Here we can only work with the partially observed state
            state_traj[:, t + 1] = deepcopy(observation)
            action_traj[:, t] = deepcopy(action)
            if terminated or truncated:
                print("terminated")
                break
            total_reward.append(reward)
        env.state_traj = state_traj
        return total_reward, action_traj


class RandomPolicy(BasePolicy):
    """
    A random policy
    """

    def __init__(self, action_space=None, low=None, high=None, seed=0):
        """
        Inputs:
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        """

        if action_space:
            self.action_space = action_space
        else:
            # NOTE: we need an explicit lowering here
            self.action_space = Box(
                low=np.float32(low), high=np.float32(high)
            )  # action_space
        self.action_space.seed(seed)
        self.magnitude = 1.0

    def compute_action(self, obs):
        """
        Return random sample from action space

        Inputs:
            obs: ndarray (unused)
        Returns:
            Random action
        """
        return self.magnitude * self.action_space.sample()

    def set_magnitude_(self, mag):
        self.magnitude = mag


class SparseEnsemblePolicy(BasePolicy):
    """
    Sparse ensemble dictionary model of the form
    Y = \\Theta(X) \\Xi
    where the labels Y are control values depending on the states u(x)
    """

    def __init__(self, optimizer, feature_library, min_bounds=None, max_bounds=None):

        # bounds for the action space
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        self.optimizer = optimizer
        self.feature_library = feature_library

    def compute_action(self, obs):
        ThetaX = self.feature_library.transform(obs)
        u = self.optimizer.coef_ @ ThetaX

        # clip action
        if self.min_bounds is not None:
            u = np.clip(u, self.min_bounds, self.max_bounds)
        return np.array(u, dtype=np.float32)

    def _init_features(self, X_concat):
        """compute Theta(X)"""
        X = self.feature_library.reshape_samples_to_spatial_grid(X_concat)
        self.ThetaX = self.feature_library.fit_transform(X)
        return self.ThetaX

    def fit(self, data_trajs, action_trajs):
        """Fit ensemble models"""
        X_concat = np.concatenate(data_trajs)
        Y_concat = np.concatenate(action_trajs)
        ThetaX = self._init_features(X_concat)
        self.optimizer.fit(ThetaX, Y_concat)
        return self.optimizer.coef_list

    def get_coef_list(self):
        """'
        Get list of model coefficients.

        (Wrapper for pysindy optimizer `coef_list` attribute.)
        """
        return self.optimizer.coef_list

    def set_mean_coef_(self, valid=False):
        """
        Set the model coefficients to be the ensemble mean.

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble mean coefficients
        """
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.optimizer.coef_ = np.mean(coef_list, axis=0)
        return self.optimizer.coef_

    def set_median_coef_(self, valid=False):
        """
        Set the model coefficients to be the ensemble median.

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble median coefficients
        """
        coef_list = np.array(self.get_coef_list())
        if valid:
            coef_list = coef_list[self.safe_idx]
        self.optimizer.coef_ = np.median(coef_list, axis=0)

        return self.optimizer.coef_

    def set_idx_coef_(self, idx):
        """
        Set the model coefficients to be the `idx`-th ensemble coefficient

        Inputs:
            `valid': (bool) whether to only perform this on validated models.
        Outputs:
            `coef_`: the ensemble `idx`-th ensemble coefficient
        """
        self.optimizer.coef_ = self.optimizer.coef_list[idx]
        return self.optimizer.coef_

    def print(self, input_features=None, precision=3):
        """
        Analagous to SINDy model print function
        Inputs:
            input_features: (list)
                List of strings for each state/control feature
            precision: (int)
                Floating point precision for printing.
        """
        lib = self.feature_library
        feature_names = lib.get_feature_names(input_features=input_features)
        coefs = self.optimizer.coef_
        for idx, eq in enumerate(coefs):
            print_str = f"u{idx} = "

            for c, name in zip(eq, feature_names):
                c_round = np.round(c, precision)
                if c_round != 0:
                    print_str += f"{c_round:.{precision}f} {name} + "

            print(print_str[:-2])


class OpenLoopSinusoidPolicy(BasePolicy):
    """Feedforward control outputting a sinewave"""

    def __init__(self, dt=1, amp=1, phase=0, offset=0, f0=1, k=1):
        """
        Amp * sin(freq * t - phase) + offset
        """
        self.amp = amp  # amplitude
        self.phase = phase  # phase
        self.offset = offset  # offset
        self.f0 = f0  # fundamental frequency
        self.k = k  # wave number

        self.dt = dt  # used for updating the time
        self.t = 0

        self.freq = 2 * np.pi * self.k / self.f0

    def compute_action(self, obs):
        """
        Return deterministic sine output

        Inputs:
            obs: ndarray (unused)
        Returns:
            Sinusoidal output depending on the number of calls
            to the policy.
        """
        self.t += self.dt

        u = self.amp * np.sin(self.freq * self.t - self.phase) + self.offset
        return np.array([u])


class OpenLoopSinRest(OpenLoopSinusoidPolicy):
    """Feedforward Sine wave for some amount of time, then do nothing.
    Used for generating data for Hydrogym environments and geting the decay
    response.
    """

    def __init__(self, t_rest, **kwargs):
        super().__init__(**kwargs)
        self.t_rest = t_rest

    def compute_action(self, obs):
        """
        Return deterministic sine output, then
        do nothing

        Inputs:
            obs: ndarray (unused)
        Returns:
            Sinusoidal output depending on the number of calls
            to the policy.
        """
        u = super().compute_action(obs)

        if self.t >= self.t_rest:
            u = u * 0.0
        return u


class OpenLoopRandRest(RandomPolicy):
    """
    Feedforward random actions, then null actions after some
    amount of time.
    """

    def __init__(self, steps_rest, **kwargs):
        super().__init__(**kwargs)
        self.steps_rest = steps_rest
        self.n_steps = 0

    def compute_action(self, obs):
        self.n_steps += 1
        u = super().compute_action(obs)

        if self.n_steps >= self.steps_rest:
            u = 0.0 * u
        return u


class SwitchPolicy(BasePolicy):
    """
    A wrapper that switches between generic policies
    """

    def __init__(self, policies):
        self.policies = policies
        self.policy = policies[0]

    def switch_criteria(self):
        """Determine when to swap between policies"""
        pass

    def compute_action(self, obs):
        policy = self.switch_criteria()
        return policy.compute_action(obs)


class SwitchAfterT(SwitchPolicy):
    """Switch between 2 policies after some amount of time"""

    def __init__(self, t_switch, policies):
        super().__init__(policies)
        self.t_switch = t_switch
        self.n_steps = 0

    def switch_criteria(self):
        """switch policies after some amount of time"""
        self.n_steps += 1

        if self.n_steps < self.t_switch:
            policy_idx = 0
        else:
            policy_idx = 1

        return self.policies[policy_idx]


class SignPolicy(BasePolicy):
    """Wrapper for creating a symmetric bang-bang controller from a given policy"""

    def __init__(self, policy, mag=1.0, thresh=0):
        self.wrapper = policy
        self.mag = mag
        self.thresh = thresh

    def compute_action(self, obs):
        # compute action from policy
        action = self.wrapper.compute_action(obs)

        # compute whether the action meets a threshold
        mask = np.abs(action) < self.thresh

        return np.sign(action) * self.mag * mask

    def set_mean_coef_(self):
        self.wrapper.set_mean_coef_()
