import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        **kwargs
    ):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def add_start_obs(self, observation):
        """
        Add a start observation
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for (
            i,
            (obs, action, reward, next_obs, terminal, agent_info, env_info),
        ) in enumerate(
            zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
            )
        ):
            self.add_sample(
                obs, action, reward, terminal, next_obs, env_info, agent_info=agent_info
            )
            if i == 0:
                self.add_start_obs(obs)
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    @abc.abstractmethod
    def random_start_obs(self, task, batch_size):
        """
        Return a batch of start observations
        """
        pass
