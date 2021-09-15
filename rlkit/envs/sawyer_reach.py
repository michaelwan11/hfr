from rlkit.envs.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from . import register_env


@register_env("sawyer-reach")
class SawyerReachEnv(object):
    def __init__(
        self,
        random_init=False,
        n_tasks=60,
        randomize_tasks=True,
        sparse=False,
        task_types=["pick_place", "reach", "push"],
        task_type="reach",
        obs_type="plain",
        goal_low=(-0.1, 0.8, 0.05),
        goal_high=(0.1, 0.9, 0.3),
        liftThresh=0.04,
        sampleMode="equal",
        rewMode="orig",
        rotMode="fixed",  #'fixed',
        **kwargs
    ):
        if sparse:
            goal_low = (-0.1, 0.7, 0.05)
            goal_high = (0.1, 0.8, 0.3)
        self.env = SawyerReachPushPickPlaceEnv(
            random_init,
            sparse,
            task_types,
            task_type,
            obs_type,
            goal_low,
            goal_high,
            liftThresh,
            sampleMode,
            rewMode,
            rotMode,
            **kwargs
        )
        self.sparse = sparse
        print("env sparse:", self.env.sparse)
        self.n_train_goals = 50
        self.n_test_goals = 10
        self.goals = self.sample_tasks(self.n_train_goals + self.n_test_goals)
        self._goal = self.goals[0]
        print("goals:", self.goals)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if self.sparse:
            done = (reward == 0)
        else:
            done = False
        return ob, reward, done, info

    def reset(self):
        return self.env.reset()

    def sample_tasks(self, num_tasks):
        goals = self.env.sample_goals_(num_tasks)
        return goals

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.env.set_goal_(self._goal)
        self.reset()
        print('goal:', self.env.goal)

    def get_train_goals(self, n_train_tasks):
        return self.goals[:n_train_tasks]

    def get_all_task_idx(self):
        return range(self.n_train_goals + self.n_test_goals)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def render(self):
        self.env.render()

    def reward(self, info, goal):
        return self.env.reward(info, goal, task_type="reach")
