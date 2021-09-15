import numpy as np

from . import register_env
from .half_cheetah import HalfCheetahEnv
import pickle
import os


@register_env("cheetah-highdim")
class HalfCheetahHighDimEnv(HalfCheetahEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        n_params = 18
        self.lo, self.hi = np.array([-1.0] * n_params), np.array([1.0] * n_params)
        data = pickle.load(open(f'{os.getcwd()}/HCstate-state.pkl', 'rb'))
        self.s_mean, self.s_std = data["mean"], data["std"]
        self._task = task
        np.random.seed(1)
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]
        super(HalfCheetahHighDimEnv, self).__init__()

    def step(self, action):
        xposbefore = np.copy(self.get_body_com("torso"))
        action = np.clip(action, *self.action_bounds)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        reward_ctrl = -0.05 * np.sum(np.square(action))
        norm_next_obs = (next_obs - self.s_mean) / self.s_std
        reward_run = np.sum(norm_next_obs * self._goal)

        reward = reward_ctrl + reward_run
        done = False
        reward_state = norm_next_obs

        infos = dict(
            reward_ctrl=reward_ctrl,
            norm_next_obs=norm_next_obs,
        )
        return (next_obs, reward, done, infos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reward(self, info, goal):
        reward_ctrl, norm_next_obs = info["reward_ctrl"], info["norm_next_obs"]
        reward_run = np.sum(norm_next_obs * goal)
        return reward_ctrl + reward_run, False

    def sample_tasks(self, num_tasks):
        tasks = [np.random.uniform(self.lo, self.hi) for _ in range(num_tasks)]
        return tasks

    def get_train_goals(self, n_train_tasks):
        return [task for task in self.tasks[:n_train_tasks]]

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task
        self.reset()

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
        size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    @property
    def action_bounds(self):    
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T
