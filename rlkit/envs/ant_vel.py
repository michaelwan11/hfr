import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env("ant-vel")
class AntVelEnv(MultitaskAntEnv):
    # Note that goal here refers to goal velocity
    def __init__(
        self, task={}, n_tasks=2, randomize_tasks=True, **kwargs
    ):
        np.random.seed(3)
        super(AntVelEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        xposbefore = np.copy(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")
        comvel = (xposafter[0] - xposbefore[0]) / self.dt
        forward_reward = -np.abs(comvel - self._goal) + 1.0
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and \
                state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                comvel=comvel,
                state=state,
            ),
        )

    def reward(self, info, goal):
        comvel = info["comvel"]
        forward_reward = -np.abs(comvel - goal) + 1.0
        reward_ctrl = info["reward_ctrl"]
        reward_contact = info["reward_contact"]
        reward_survive = info["reward_survive"]
        state = info["state"]
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and \
                state[2] <= 1.0
        reward = forward_reward + reward_ctrl + reward_contact + reward_survive
        done = not notdone
        return reward, done

    def sample_tasks(self, num_tasks):
        tasks = np.random.uniform(0.0, 3.0, (num_tasks, ))
        tasks = [{"goal": goal} for goal in tasks]
        return tasks

    def _get_obs(self):
        """
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.get_body_xmat("torso").flat,
                self.get_body_com("torso"),
            ]
        ).reshape(-1)
        """
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    @property
    def action_bounds(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

@register_env("ant-vel-sparse")
class AntVelSparseEnv(MultitaskAntEnv):
    # Note that goal here refers to goal velocity
    def __init__(
        self, task={}, n_tasks=2, randomize_tasks=True, **kwargs
    ):
        np.random.seed(3)
        self.goal_radius = 0.5
        super(AntVelSparseEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        xposbefore = np.copy(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")
        comvel = (xposafter[0] - xposbefore[0]) / self.dt
        # forward_reward = -np.abs(comvel - self._goal) + 1.0
        forward_reward = -np.abs(comvel - self._goal)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.05
        forward_reward = self.sparsify_rewards(forward_reward)
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and \
                state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                comvel=comvel,
                state=state,
            ),
        )

    def reward(self, info, goal):
        comvel = info["comvel"]
        # forward_reward = -np.abs(comvel - goal) + 1.0
        forward_reward = -np.abs(comvel - goal)
        forward_reward = self.sparsify_rewards(forward_reward)
        reward_ctrl = info["reward_ctrl"]
        reward_contact = info["reward_contact"]
        reward_survive = info["reward_survive"]
        state = info["state"]
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and \
                state[2] <= 1.0
        reward = forward_reward + reward_ctrl + reward_contact + reward_survive
        done = not notdone
        return reward, done

    def sample_tasks(self, num_tasks):
        tasks = np.random.uniform(-1.5, 1.5, (num_tasks, ))
        tasks = [{"goal": goal} for goal in tasks]
        return tasks

    def sparsify_rewards(self, r):
        """
        if r < -self.goal_radius:
            r = -2
            # r = -1
        r = r + 2
        """
        if r > -self.goal_radius:
            r = r + 1
        # r = r + 1
        return r

    def _get_obs(self):
        """
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.get_body_xmat("torso").flat,
                self.get_body_com("torso"),
            ]
        ).reshape(-1)
        """
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    @property
    def action_bounds(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
