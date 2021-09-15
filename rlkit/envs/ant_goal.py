import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env("ant-goal")
class AntGoalEnv(MultitaskAntEnv):
    def __init__(
        self, task={}, n_tasks=2, randomize_tasks=True, sparse=True, **kwargs
    ):
        self.sparse = sparse
        np.random.seed(1)
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        dist = np.linalg.norm(xposafter[:2] - self._goal)
        goal_reward = -np.sum(
            np.abs(xposafter[:2] - self._goal)
        ) + 4.0  # make it happy, not suicidal

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.square(action / scaling).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.05
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        # reward = goal_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        sparse_goal_reward = goal_reward
        if dist > 0.8:
            sparse_goal_reward = -np.sum(np.abs(self._goal)) + 4.0
        sparse_reward = sparse_goal_reward - ctrl_cost - contact_cost + survive_reward
        reward = sparse_reward
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                sparse_reward=sparse_reward,
                goal_forward=goal_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                xposafter=xposafter,
                done=done,
            ),
        )

    def reward(self, info, goal):
        reward_ctrl, reward_contact = info["reward_ctrl"], info["reward_contact"]
        reward_survive, xposafter = info["reward_survive"], info["xposafter"]
        done = info["done"]
        dist = np.linalg.norm(xposafter[:2] - goal)
        goal_reward = -np.sum(np.abs(xposafter[:2] - goal)) + 4.0
        sparse_goal_reward = goal_reward
        if dist > 0.8:
            sparse_goal_reward = -np.sum(np.abs(goal)) + 4.0
        sparse_reward = sparse_goal_reward + reward_ctrl + reward_contact + \
            reward_survive
        reward = goal_reward + reward_ctrl + reward_contact + reward_survive
        # return reward, sparse_reward, done
        return sparse_reward, done

    def sample_tasks(self, num_tasks):
        radius = 2.0
        angles = np.linspace(0, np.pi, num=num_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        np.random.shuffle(goals)
        goals = goals.tolist()
        tasks = [{"goal": goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    @property
    def action_bounds(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T
