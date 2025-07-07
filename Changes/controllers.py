
from utilities import dss_sim, rep_mat, uptria2vec, push_vec
import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize, basinhopping
from scipy.optimize import Bounds
from numpy.linalg import lstsq
import math
import warnings
from tabulate import tabulate

# Controller selector
def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, mode):
    if mode == 'manual':
        return action_manual
    elif mode == 'nominal':
        return ctrl_nominal.compute_action(t, observation)
    else:
        return ctrl_benchmarking.compute_action(t, observation)

# Optimal Predictive Controller (MPC)
class ControllerOptimalPredictive:
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init=[],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=5,
                 pred_step_size=0.1,
                 sys_rhs=None,
                 sys_out=None,
                 state_sys=None,
                 buffer_size=20,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 run_obj_struct='quadratic',
                 run_obj_pars=[],
                 observation_target=[],
                 state_init=[],
                 obstacle=[],
                 seed=1):

        np.random.seed(seed)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.mode = mode
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        self.Nactor = Nactor
        self.pred_step_size = pred_step_size
        self.action_min = np.array(ctrl_bnds[:, 0])
        self.action_max = np.array(ctrl_bnds[:, 1])
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor)

        if len(action_init) == 0:
            self.action_init = self.action_min / 10
        else:
            self.action_init = action_init

        self.action_curr = self.action_init
        self.action_sqn_init = rep_mat(self.action_init, 1, self.Nactor)
        self.state_init = state_init
        self.action_buffer = np.zeros([buffer_size, dim_input])
        self.observation_buffer = np.zeros([buffer_size, dim_output])
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys
        self.buffer_size = buffer_size
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = min(Ncritic, buffer_size - 1)
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.run_obj_struct = run_obj_struct
        self.run_obj_pars = run_obj_pars
        self.observation_target = observation_target
        self.accum_obj_val = 0
        self.N_CTRL = N_CTRL()

        # Critic dimensionality based on structure
        if self.critic_struct == 'quad-nomix':
            self.dim_critic = dim_output + dim_input
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        else:
            raise NotImplementedError("Only 'quad-nomix' critic structure is supported in this setup.")

    def reset(self, t0):
        self.action_curr = self.action_init
        self.action_sqn_init = rep_mat(self.action_init, 1, self.Nactor)
        self.action_buffer = np.zeros([self.buffer_size, self.dim_input])
        self.observation_buffer = np.zeros([self.buffer_size, self.dim_output])
        self.ctrl_clock = t0
        self.critic_clock = t0

    def receive_sys_state(self, state):
        self.state_sys = state

    def upd_accum_obj(self, observation, action):
        self.accum_obj_val += self.run_obj(observation, action) * self.sampling_time

    def run_obj(self, observation, action):
        # Simple quadratic cost to penalize deviation from target and excessive control effort
        obs_error = observation - self.observation_target
        return np.dot(obs_error, obs_error) + 0.1 * np.dot(action, action)

    def _actor_cost(self, action_sqn, observation):
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        state = self.state_sys
        observation_sqn[0, :] = observation

        for k in range(1, self.Nactor):
            state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k - 1, :])
            observation_sqn[k, :] = self.sys_out(state)

        J = 0
        for k in range(self.Nactor):
            J += self.gamma**k * self.run_obj(observation_sqn[k, :], my_action_sqn[k, :])
        return J

    def _actor_optimizer(self, observation):
        method = 'SLSQP'
        options = {'maxiter': 50, 'disp': False}
        my_action_sqn_init = self.action_sqn_init.reshape(-1,)
        bounds = Bounds(self.action_sqn_min.reshape(-1,), self.action_sqn_max.reshape(-1,), keep_feasible=True)

        try:
            result = minimize(lambda x: self._actor_cost(x, observation),
                              my_action_sqn_init,
                              method=method,
                              bounds=bounds,
                              options=options)
            return result.x[:self.dim_input]
        except Exception as e:
            print(f"Optimizer failed: {e}")
            return self.action_curr

    def compute_action(self, t, observation):
        time_in_sample = t - self.ctrl_clock
        if time_in_sample >= self.sampling_time:
            self.ctrl_clock = t
            if self.mode == "MPC":
                action = self._actor_optimizer(observation)
            elif self.mode == "N_CTRL":
                action = self.N_CTRL.pure_loop(observation)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            self.action_curr = action
        return self.action_curr

# Simple constant controller
class N_CTRL:
    def __init__(self):
        pass

    def pure_loop(self, observation):
        # Returns small forward velocity and no rotation
        v = 0.2
        w = 0.0
        return [v, w]
