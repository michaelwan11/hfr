import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env("point-robot")
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=True, n_tasks=2):

        if randomize_tasks:
            np.random.seed(2)
            goals = [
                [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)]
                for _ in range(n_tasks)
            ]
        else:
            # some hand-coded goals for debugging
            goals = [
                np.array([10, -10]),
                np.array([10, 10]),
                np.array([-10, 10]),
                np.array([-10, -10]),
                np.array([0, 0]),
                np.array([7, 2]),
                np.array([0, 4]),
                np.array([-6, 9]),
            ]
            goals = [g / 10.0 for g in goals]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        """ reset goal AND reset the agent """
        self._goal = self.goals[idx]
        self.reset()

    def get_train_goals(self, n_train_tasks):
        return self.goals[:n_train_tasks]

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1.0, 1.0, size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(x=x + self._goal[0], y=y + self._goal[1])

    def reward(self, info, goal):
        x, y = info["x"], info["y"]
        reward = -(((x - goal[0]) ** 2 + (y - goal[1]) ** 2) ** 0.5)
        return reward

    def viewer_setup(self):
        print("no viewer")
        pass

    def render(self):
        print("current state:", self._state)

@register_env("four-corners")
class FourCorners(PointEnv):
    def __init__(self, randomize_tasks=True, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius
        self.length = 1.0
        self.goals = [[-self.length, -self.length], [self.length, self.length], [self.length, -self.length], [-self.length, self.length]]

        # self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        """ zero out rewards when outside the goal radius """
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        x, y = ob[0], ob[1]
        if reward >= -self.goal_radius:
            reward = 0.
            done = True
        else:
            reward = -1.
            if self._goal[0] == -self.length and self._goal[1] == -self.length:
                if x < -0.1 and x > -0.5 and y < -0.1 and y > -0.5:
                    reward = -3.
            elif self._goal[0] == self.length and self._goal[1] == self.length:
                if x > 0.1 and x < 0.5 and y > 0.1 and y < 0.5:
                    reward = -3.
            elif self._goal[0] == self.length and self._goal[1] == -self.length:
                if x > 0.1 and x < 0.5 and y < -0.1 and y > -0.5:
                    reward = -3.
            else:
                if x < -0.1 and x > -0.5 and y > 0.1 and y < 0.5:
                    reward = -3.
        d.update({"sparse_reward": reward, "success": reward == 0.})
        return ob, reward, done, d

    def reward(self, info, goal):
        x, y = info["x"], info["y"]
        x_diff, y_diff = x - goal[0], y - goal[1]
        reward = - (x_diff ** 2 + y_diff ** 2) ** 0.5
        done = False
        if reward >= -self.goal_radius:
            reward = 0.
            done = True
        else:
            reward = -1.
            if goal[0] == -self.length and goal[1] == -self.length:
                if x < -0.1 and x > -0.5 and y < -0.1 and y > -0.5:
                    reward = -3.
            elif goal[0] == self.length and goal[1] == self.length:
                if x > 0.1 and x < 0.5 and y > 0.1 and y < 0.5:
                    reward = -3.
            elif goal[0] == self.length and goal[1] == -self.length:
                if x > 0.1 and x < 0.5 and y < -0.1 and y > -0.5:
                    reward = -3.
            else:
                if x < -0.1 and x > -0.5 and y > 0.1 and y < 0.5:
                    reward = -3.
        return reward, done
