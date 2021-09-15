import random


class ML1Env(object):
    def __init__(self, train_env, test_env):
        self.train_env = train_env
        self.test_env = test_env
        self.train = True
        self.num_train_tasks = 50
        self.num_test_tasks = 10
        self.train_tasks = self.get_train_tasks()
        self._goal = self.train_tasks[0].get("goal", 0.0)
        self.test_tasks = self.get_test_tasks()
        self.observation_space = train_env.observation_space
        self.action_space = train_env.action_space

    def set_train(self, train=True):
        self.train = train

    def step(self, action):
        observation, reward, done, infos = (
            self.train_env.step(action) if self.train else self.test_env.step(action)
        )
        return (observation, reward, done, infos)

    def get_train_tasks(self):
        return [{"task": 0, "goal": g} for g in range(self.num_train_tasks)]

    def get_test_tasks(self):
        return [{"task": 0, "goal": g} for g in range(self.num_test_tasks)]

    def get_all_task_idx(self):
        return range(len(self.train_tasks + self.test_tasks))

    def reset_task(self, idx):
        if self.train:
            self.train_env.set_task(self.train_tasks[idx])
            self._goal = self.train_tasks[idx]["goal"]
            self.train_env.reset()
        else:
            print(self.test_tasks, len(self.test_tasks))
            print(idx - self.num_train_tasks)
            self.test_env.set_task(self.test_tasks[idx-self.num_train_tasks])
            self._goal = self.test_tasks[idx-self.num_train_tasks]["goal"]
            self.test_env.reset()

    def reset(self):
        if self.train:
            return self.train_env.reset()
        else:
            return self.test_env.reset()


class ML10Env(object):
    def __init__(self, train_env, test_env):
        pass

