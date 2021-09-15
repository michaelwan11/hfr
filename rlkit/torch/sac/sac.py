from collections import OrderedDict
import numpy as np

from scipy.special import logsumexp, softmax

import torch
import torch.optim as optim
from torch import nn as nn
import copy

import time
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
        self,
        env,
        train_tasks,
        eval_tasks,
        train_goals,
        latent_dim,
        nets,
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        rf_lr=1e-3,
        context_lr=1e-3,
        kl_lambda=1.0,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.0,
        optimizer_class=optim.Adam,
        recurrent=False,
        use_information_bottleneck=True,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        use_learned_reward=False,
        use_softmax=True,
        soft_target_tau=1e-2,
        plotter=None,
        render_eval_paths=False,
        **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            train_goals=train_goals,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.rf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.multitask_q = False

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.use_learned_reward = use_learned_reward
        self.use_softmax = use_softmax

        self.qf1, self.qf2, self.vf, self.rf, self.multitask_qf1, self.multitask_qf2, self.multitask_vf = nets[
            1:
        ]
        self.target_vf = self.vf.copy()
        self.multitask_target_vf = self.multitask_vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(), lr=policy_lr
        )
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.multitask_qf1_optimizer = optimizer_class(
            self.multitask_qf1.parameters(), lr=qf_lr
        )
        self.multitask_qf2_optimizer = optimizer_class(
            self.multitask_qf2.parameters(), lr=qf_lr
        )
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=vf_lr)
        self.rf_optimizer = optimizer_class(self.rf.parameters(), lr=rf_lr)
        self.multitask_vf_optimizer = optimizer_class(
            self.multitask_vf.parameters(), lr=vf_lr
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(), lr=context_lr
        )
        self.log_Z = np.zeros((self.n_train_goals,))

    ###### Torch stuff #####
    @property
    def networks(self):
        return (
            self.agent.networks
            + [self.agent]
            + [
                self.qf1,
                self.qf2,
                self.vf,
                self.target_vf,
                self.rf,
                self.multitask_qf1,
                self.multitask_qf2,
                self.multitask_vf,
                self.multitask_target_vf,
            ]
        )

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        """ unpack a batch and return individual elements """
        o = batch["observations"][None, ...]
        a = batch["actions"][None, ...]
        if sparse_reward:
            r = batch["sparse_rewards"][None, ...]
        else:
            r = batch["rewards"][None, ...]
        no = batch["next_observations"][None, ...]
        t = batch["terminals"][None, ...]
        return [o, a, r, no, t]

    def unpack_start_obs(self, batch):
        o = batch["start_obs"][None, ...]
        return [o]

    def sample_sac(self, indices, batch_size=None):
        """ sample batch of training data from a list of tasks for training the actor-critic """
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batch_size = self.batch_size if batch_size is None else batch_size
        batches = [
            ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(idx, batch_size=batch_size)
            )
            for idx in indices
        ]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_start_obs(self, indices, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        batches = [
            ptu.np_to_pytorch_batch(
                self.replay_buffer.random_start_obs(idx, batch_size=batch_size)
            )
            for idx in indices
        ]
        unpacked = [self.unpack_start_obs(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked[0]

    def sample_context(self, indices, use_sampler_buffer=False, batch_size=None):
        """ sample batch of context from a list of tasks from the replay buffer """
        # make method work given a single task index
        if not hasattr(indices, "__iter__"):
            indices = [indices]
        buffer = (
            self.sampler_enc_replay_buffer
            if use_sampler_buffer
            else self.enc_replay_buffer
        )
        batch_size = self.embedding_batch_size if batch_size is None else batch_size
        random_batches = [
            buffer.random_batch(idx, batch_size=batch_size, sequence=self.recurrent)
            for idx in indices
        ]
        batches = [ptu.np_to_pytorch_batch(batch) for batch in random_batches]
        context = [
            self.unpack_batch(batch, sparse_reward=self.sparse_rewards)
            for batch in batches
        ]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context, indices

    def relabel_hfr_q(
        self,
        paths,
        add_to_enc_buffer=True,
        add_original=True,
        random_relabel=False,
    ):
        # always add the orignal to sampler_enc_replay_buffer
        self.sampler_enc_replay_buffer.add_paths(self.task_idx, copy.deepcopy(paths))
        if add_original:
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                if self.should_clear[self.task_idx]:
                    self.enc_replay_buffer.task_buffers[self.task_idx].clear()
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
                self.should_clear[self.task_idx] = 0.0

        elif random_relabel:
            for path in paths:
                task_idx = np.random.choice(self.n_train_goals)
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)
                if add_to_enc_buffer:
                    if self.should_clear[task_idx]:
                        self.enc_replay_buffer.task_buffers[task_idx].clear()
                    self.enc_replay_buffer.add_path(task_idx, path)
                    # if this task gets chosen again on the same iteration we shouldn't
                    # remove what we just added
                    self.should_clear[task_idx] = 0.0
                self.replay_buffer.add_path(task_idx, path)

        else:
            # relabel using our method
            for path in paths:
                observations = path["observations"]
                actions = path["actions"]
                rewards = path["rewards"]
                path_obs = ptu.from_numpy(observations)
                path_actions = ptu.from_numpy(actions)
                # task dimension will be one
                context = np.expand_dims(
                    np.hstack([observations, actions, rewards]), axis=0
                )
                context = ptu.from_numpy(context)
                q_vals = np.zeros((self.n_train_goals))
                log_pis = np.zeros((self.n_train_goals))
                for idx, goal in enumerate(self.train_goals):
                    if self.use_learned_reward:
                        task_vectors = np.array([goal for _ in range(path_obs.shape[0])])
                        task_vectors = ptu.from_numpy(task_vectors)
                        task_vectors = task_vectors.view(path_obs.shape[0], -1)
                        rewards = self.rf(path_obs, path_actions, task_vectors)
                        rewards = rewards.squeeze()
                        context[0, :, -1] = rewards
                    else:
                        if self.sparse_rewards:
                            rewards = np.array(
                                [self.env.reward(info, goal)[1] for info in path["env_infos"]]
                            )
                        else:
                            rewards = np.array(
                                [self.env.reward(info, goal)[0] for info in path["env_infos"]]
                            )
                        context[0, :, -1] = ptu.from_numpy(rewards)
                    start_obs = self.sample_start_obs([idx],
                            batch_size=self.utility_batch_size)
                    policy_outputs, task_z = self.agent(start_obs, context)
                    actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
                    t, b, _ = start_obs.size()
                    start_obs = start_obs.view(t * b, -1)
                    actions = actions.view(t * b, -1)
                    min_qs = self._min_q(start_obs, actions, task_z)
                    q_vals[idx] = ptu.get_numpy(torch.mean(min_qs))
                    log_pis[idx] = ptu.get_numpy(torch.mean(log_pi))
                if self.relabel_statistics is None:
                    self.relabel_statistics = OrderedDict()
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation Q Vals", q_vals)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation log pi", log_pis)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation partition", self.log_Z)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation logits", q_vals - self.log_Z)
                    )
                dist = softmax(q_vals - self.log_Z)
                if self.use_softmax:
                    task_idx = np.random.choice(self.n_train_goals, p=dist)
                else:
                    task_idx = np.argmax(dist)
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    if self.use_learned_reward:
                        task_vectors = np.array([self.train_goals[task_idx] for _ in range(path_obs.shape[0])])
                        task_vectors = ptu.from_numpy(task_vectors)
                        task_vectors = task_vectors.view(path_obs.shape[0], -1)
                        rewards = self.rf(path_obs, path_actions, task_vectors)
                        rewards = ptu.get_numpy(rewards)
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)

                if add_to_enc_buffer:
                    if self.should_clear[task_idx]:
                        self.enc_replay_buffer.task_buffers[task_idx].clear()
                    self.enc_replay_buffer.add_path(task_idx, path)
                    # if this task gets chosen again on the same iteration we shouldn't
                    # remove what we just added
                    self.should_clear[task_idx] = 0.0
                self.replay_buffer.add_path(task_idx, path)
         
    def relabel_hfr_bellman(
        self,
        paths,
        add_to_enc_buffer=True,
        add_original=True,
        random_relabel=False,
    ):
        # always add the orignal to sampler_enc_replay_buffer
        self.sampler_enc_replay_buffer.add_paths(self.task_idx, copy.deepcopy(paths))
        if add_original:
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                if self.should_clear[self.task_idx]:
                    self.enc_replay_buffer.task_buffers[self.task_idx].clear()
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
                self.should_clear[self.task_idx] = 0.0

        elif random_relabel:
            for path in paths:
                task_idx = np.random.choice(self.n_train_goals)
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)
                if add_to_enc_buffer:
                    if self.should_clear[task_idx]:
                        self.enc_replay_buffer.task_buffers[task_idx].clear()
                    self.enc_replay_buffer.add_path(task_idx, path)
                    # if this task gets chosen again on the same iteration we shouldn't
                    # remove what we just added
                    self.should_clear[task_idx] = 0.0
                self.replay_buffer.add_path(task_idx, path)

        else:
            # relabel using our method
            for path in paths:
                observations = path["observations"]
                actions = path["actions"]
                rewards = path["rewards"]
                # task dimension will be one
                context = np.expand_dims(
                    np.hstack([observations, actions, rewards]), axis=0
                )
                context = ptu.from_numpy(context)
                negative_bellman = np.zeros((self.n_train_goals))
                for idx, goal in enumerate(self.train_goals):
                    if self.sparse_rewards:
                        rewards = np.array(
                            [self.env.reward(info, goal)[1] for info in path["env_infos"]]
                        )
                    else:
                        rewards = np.array(
                            [self.env.reward(info, goal)[0] for info in path["env_infos"]]
                        )
                    context[0, :, -1] = ptu.from_numpy(rewards)
                    sac_obs, sac_actions, sac_rewards, sac_next_obs, sac_terms = \
                            self.sample_sac([idx], batch_size=self.utility_batch_size)
                    policy_outputs, task_z = self.agent(sac_obs, context)
                    actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
                    t, b, _ = sac_obs.size()
                    sac_obs = sac_obs.view(t * b, -1)
                    sac_next_obs = sac_next_obs.view(t * b, -1)
                    sac_actions = sac_actions.view(t * b, -1)
                    sac_terms = sac_terms.view(t * b, -1)
                    sac_rewards = sac_rewards.view(t * b, -1)
                    sac_rewards = sac_rewards * self.reward_scale
                    min_qs = self._min_q(sac_obs, sac_actions, task_z)
                    with torch.no_grad():
                        target_v_values = self.target_vf(sac_next_obs, task_z)
                        q_target = sac_rewards + (1.0 - sac_terms) * self.discount * target_v_values
                        q1_pred = self.qf1(sac_obs, sac_actions, task_z)
                        q2_pred = self.qf2(sac_obs, sac_actions, task_z)
                        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean(
                            (q2_pred - q_target) ** 2
                        )
                        negative_bellman[idx] = ptu.get_numpy(-qf_loss)
                        # negative_bellman[idx] = ptu.get_numpy(qf_loss)
                if self.relabel_statistics is None:
                    self.relabel_statistics = OrderedDict()
                    zeros = np.zeros((self.n_train_goals))
                    # Adaptation q vals and adapatation log pi are meaningless in this
                    # case
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation Q Vals", zeros)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation log pi", zeros)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation partition", self.log_Z) 
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation logits", negative_bellman - self.log_Z)
                    )
                dist = softmax(negative_bellman - self.log_Z)
                task_idx = np.random.choice(self.n_train_goals, p=dist)
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)

                if add_to_enc_buffer:
                    if self.should_clear[task_idx]:
                        self.enc_replay_buffer.task_buffers[task_idx].clear()
                    self.enc_replay_buffer.add_path(task_idx, path)
                    # if this task gets chosen again on the same iteration we shouldn't
                    # remove what we just added
                    self.should_clear[task_idx] = 0.0
                self.replay_buffer.add_path(task_idx, path)

    def relabel_hipi(
        self,
        paths,
        add_to_enc_buffer=True,
        add_original=True,
        random_relabel=False
    ):
        # always add the orignal to sampler_enc_replay_buffer
        self.sampler_enc_replay_buffer.add_paths(self.task_idx, copy.deepcopy(paths))
        if add_original:
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                if self.should_clear[self.task_idx]:
                    self.enc_replay_buffer.task_buffers[self.task_idx].clear()
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
                self.should_clear[self.task_idx] = 0.0
        elif random_relabel:
            for path in paths:
                task_idx = np.random.choice(self.n_train_goals)
                if self.should_clear[task_idx]:
                    self.enc_replay_buffer.task_buffers[task_idx].clear()
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)
                self.enc_replay_buffer.add_path(task_idx, path)
                self.replay_buffer.add_path(task_idx, path)
                # if this task gets chosen again on the same iteration we shouldn't
                # remove what we just added
                self.should_clear[task_idx] = 0.0
        else:
            for path in paths:
                observations = path["observations"]
                actions = path["actions"]
                next_observations = path["next_observations"]
                terminals = path["terminals"]
                agent_infos = path["agent_infos"]
                total_rewards = np.zeros((self.n_train_goals,))
                env_infos = path["env_infos"]
                # train_goals x timesteps
                rewards = np.array(
                    [
                        np.array([self.env.reward(info, goal)[0] for info in env_infos])
                        for goal in self.train_goals
                    ]
                )
                dones = np.array(
                    [
                        np.array([self.env.reward(info, goal)[1] for info in env_infos])
                        for goal in self.train_goals
                    ]
                )
                # sample context for all tasks
                context, _ = self.sample_context(range(self.n_train_goals))
                # Create n_train_goals x path_length x obs_dim array
                obs = np.array([observations for _ in range(self.n_train_goals)])
                acts = np.array([actions for _ in range(self.n_train_goals)])
                obs, acts = ptu.from_numpy(obs), ptu.from_numpy(acts)
                policy_outputs, task_z = self.agent(obs, context)
                t, b, _ = obs.size()
                obs = obs.view(t * b, -1)
                acts = acts.view(t * b, -1)
                if self.multitask_q:
                    task_vectors = np.array([self.train_goals[i] for i in
                        range(self.n_train_goals)])
                    task_vectors = ptu.from_numpy(task_vectors)
                    task_vectors = [v.repeat(b, 1) for v in task_vectors]
                    task_vectors = torch.cat(task_vectors, dim=0)
                    min_qs = self._min_q(obs, acts, task_z, task_vectors)
                else:
                    min_qs = self._min_q(obs, acts, task_z)
                min_qs = min_qs.view(t, b)
                min_qs = ptu.get_numpy(min_qs)
                log_Z = logsumexp(min_qs, axis=1, b=1.0/b)
                dist = softmax(min_qs - np.expand_dims(log_Z, axis=1), axis=0)
                for i in range(b):
                    task_idx = np.random.choice(self.n_train_goals, p=dist[:, i])
                    if self.sparse_rewards:
                        reward, sparse_reward, done = self.env.reward(env_infos[i], self.train_goals[task_idx])
                        env_infos[i].update({"sparse_reward": sparse_reward})
                    else:
                        reward, done = self.env.reward(env_infos[i], self.train_goals[task_idx])
                    if add_to_enc_buffer:
                        if self.should_clear[task_idx]:
                            self.enc_replay_buffer.task_buffers[task_idx].clear()
                        self.enc_replay_buffer.add_sample(
                            task_idx,
                            observations[i],
                            actions[i],
                            reward,
                            done,
                            next_observations[i],
                            env_infos[i],
                            agent_info=agent_infos[i],
                        )
                        if self.relabel_statistics is None:
                            self.relabel_statistics = OrderedDict()
                            zeros = np.zeros((self.n_train_goals))
                            self.relabel_statistics.update(
                                create_stats_ordered_dict("Adaptation Q Vals", zeros)
                            )
                            self.relabel_statistics.update(
                                create_stats_ordered_dict("Adaptation log pi", zeros)
                            )
                            self.relabel_statistics.update(
                                create_stats_ordered_dict("Adaptation partition", zeros)
                            )
                            self.relabel_statistics.update(
                                create_stats_ordered_dict("Adaptation logits", zeros)
                            )
                        self.should_clear[task_idx] = 0.0
                    self.replay_buffer.add_sample(
                        task_idx,
                        observations[i],
                        actions[i],
                        reward,
                        done,
                        next_observations[i],
                        env_infos[i],
                        agent_info=agent_infos[i]
                    )

    def relabel_both_hindsight_traj(
        self,
        paths,
        add_to_enc_buffer=True,
        add_original=True,
        random_relabel=False
    ):
        # always add the orignal to sampler_enc_replay_buffer
        self.sampler_enc_replay_buffer.add_paths(self.task_idx, copy.deepcopy(paths))
        if add_original:
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                if self.should_clear[self.task_idx]:
                    self.enc_replay_buffer.task_buffers[self.task_idx].clear()
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
                self.should_clear[self.task_idx] = 0.0
        elif random_relabel:
            for path in paths:
                task_idx = np.random.choice(self.n_train_goals)
                if self.should_clear[task_idx]:
                    self.enc_replay_buffer.task_buffers[task_idx].clear()
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                if self.sparse_rewards:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    sparse_rewards = [rd[1] for rd in rewards_and_done]
                    terminals = np.array([rd[2] for rd in rewards_and_done])
                    for info, sparse_reward in zip(path["env_infos"], sparse_rewards):
                        info.update({"sparse_reward": sparse_reward})
                else:
                    rewards = np.array([rd[0] for rd in rewards_and_done])
                    terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)
                self.enc_replay_buffer.add_path(task_idx, path)
                self.replay_buffer.add_path(task_idx, path)
                # if this task gets chosen again on the same iteration we shouldn't
                # remove what we just added
                self.should_clear[task_idx] = 0.0
        else:
            num_paths = len(paths)
            partition_array = np.zeros((self.n_train_goals, num_paths))
            for i, path in enumerate(paths):
                env_infos = path["env_infos"]
                # train_goals x timesteps
                rewards = np.array(
                    [
                        np.array([self.env.reward(info, goal)[0] for info in env_infos])
                        for goal in self.train_goals
                    ]
                )
                cumulative_rewards = np.sum(rewards, axis=1)
                partition_array[:, i] = cumulative_rewards
            log_Z = logsumexp(partition_array, axis=1, b=1.0 / num_paths)

            for path in paths:
                observations = path["observations"]
                actions = path["actions"]
                next_observations = path["next_observations"]
                terminals = path["terminals"]
                agent_infos = path["agent_infos"]
                total_rewards = np.zeros((self.n_train_goals,))
                env_infos = path["env_infos"]
                # train_goals x timesteps
                rewards = np.array(
                    [
                        np.array([self.env.reward(info, goal)[0] for info in env_infos])
                        for goal in self.train_goals
                    ]
                )
                cumulative_rewards = np.sum(rewards, axis=1)
                dist = softmax(cumulative_rewards - log_Z)
                task_idx = np.random.choice(self.n_train_goals, p=dist)
                rewards_and_done = [
                    self.env.reward(info, self.train_goals[task_idx])
                    for info in path["env_infos"]
                ]
                rewards = np.array([rd[0] for rd in rewards_and_done])
                terminals = np.array([rd[1] for rd in rewards_and_done])
                path["rewards"] = rewards.reshape(-1, 1)
                path["terminals"] = terminals.reshape(-1, 1)
                if add_to_enc_buffer:
                    if self.should_clear[task_idx]:
                        self.enc_replay_buffer.task_buffers[task_idx].clear()
                    self.enc_replay_buffer.add_path(task_idx, path)
                    # if this task gets chosen again on the same iteration we shouldn't
                    # remove what we just added
                    self.should_clear[task_idx] = 0.0
                self.replay_buffer.add_path(task_idx, path)
                if self.relabel_statistics is None:
                    self.relabel_statistics = OrderedDict()
                    zeros = np.zeros((self.n_train_goals))
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation Q Vals", zeros)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation log pi", zeros)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation partition", zeros)
                    )
                    self.relabel_statistics.update(
                        create_stats_ordered_dict("Adaptation logits", zeros)
                    )

    def compute_log_Z(self):
        """Compute log of the partition function for meta relabeling"""
        indices = self.train_tasks
        # 20 seems to work reasonably well
        q_vals = np.zeros((self.n_train_goals, 20))
        for i in range(20):
            context, _ = self.sample_context(
                indices, use_sampler_buffer=True, batch_size=self.max_path_length
            )
            start_obs = self.sample_start_obs(indices,
                    batch_size=self.utility_batch_size)
            policy_outputs, task_z = self.agent(start_obs, context)
            actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            t, b, _ = start_obs.size()
            start_obs = start_obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            """
            task_vectors = ptu.one_hot(
                ptu.from_numpy(np.array(indices)).long(), self.n_train_goals
            )
            task_vectors = [v.repeat(b, 1) for v in task_vectors]
            task_vectors = torch.cat(task_vectors, dim=0)
            """
            # min_qs = self._min_q(start_obs, actions, task_z, task_vectors)
            min_qs = self._min_q(start_obs, actions, task_z)
            # soft_qs = min_qs - log_pi
            soft_qs = min_qs
            soft_qs = soft_qs.view(t, b, -1)
            soft_qs = torch.mean(soft_qs, axis=1)
            q_vals[:, i] = ptu.get_numpy(soft_qs[:, 0])
        log_Z = logsumexp(q_vals, axis=1, b=1.0 / 20)
        self.log_Z[:] = log_Z

    def compute_log_Z_bellman(self):
        indices = self.train_tasks
        negative_bellman = np.zeros((self.n_train_goals, 20))
        for i in range(20):
            context, _ = self.sample_context(
                indices, use_sampler_buffer=True, batch_size=self.max_path_length
            )
            sac_obs, sac_actions, sac_rewards, sac_next_obs, sac_terms = \
                    self.sample_sac(indices, batch_size=self.utility_batch_size)
            policy_outputs, task_z = self.agent(sac_obs, context)
            t, b, _ = sac_obs.size()
            sac_obs = sac_obs.view(t * b, -1)
            sac_next_obs = sac_next_obs.view(t * b, -1)
            sac_actions = sac_actions.view(t * b, -1)
            sac_rewards = sac_rewards.view(t * b, -1)
            sac_rewards = sac_rewards * self.reward_scale
            sac_terms = sac_terms.view(t * b, -1)
            min_qs = self._min_q(sac_obs, sac_actions, task_z)
            with torch.no_grad():
                target_v_values = self.target_vf(sac_next_obs, task_z)
                q_target = sac_rewards + (1.0 - sac_terms) * self.discount * target_v_values
                q1_pred = self.qf1(sac_obs, sac_actions, task_z)
                q2_pred = self.qf2(sac_obs, sac_actions, task_z)
                bellman = ((q1_pred - q_target) ** 2) + ((q2_pred - q_target) ** 2)
                bellman = bellman.view(t, b, -1)
                bellman = torch.mean(bellman, axis=1)
                negative_bellman[:, i] = ptu.get_numpy(-bellman[:, 0])
                # negative_bellman[:, i] = ptu.get_numpy(bellman[:, 0])

        log_Z = logsumexp(negative_bellman, axis=1, b=1.0/20)
        self.log_Z[:] = log_Z

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch, indices = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size : i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z, task_idx=None):
        if task_idx is None:
            q1 = self.qf1(obs, actions, task_z.detach())
            q2 = self.qf2(obs, actions, task_z.detach())
        else:
            q1 = self.multitask_qf1(obs, actions, task_idx)
            q2 = self.multitask_qf2(obs, actions, task_idx)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        if self.multitask_q:
            ptu.soft_update_from_to(
                self.multitask_vf, self.multitask_target_vf, self.soft_target_tau
            )

    def _take_step(self, indices, context):
        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        # task_vectors = ptu.one_hot(ptu.from_numpy(indices).long(), self.n_train_goals)
        if self.multitask_q:
            task_vectors = np.array([self.train_goals[i] for i in indices])
            task_vectors = ptu.from_numpy(task_vectors)
            task_vectors = [v.repeat(b, 1) for v in task_vectors]
            task_vectors = torch.cat(task_vectors, dim=0)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        if self.multitask_q:
            multitask_q1_pred = self.multitask_qf1(obs, actions, task_vectors)
            multitask_q2_pred = self.multitask_qf2(obs, actions, task_vectors)
            multitask_v_pred = self.multitask_vf(obs, task_vectors)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)
            if self.multitask_q:
                multitask_target_v_values = self.multitask_target_vf(next_obs, task_vectors)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        if self.use_learned_reward:
            self.rf_optimizer.zero_grad()
        if self.multitask_q:
            self.multitask_qf1_optimizer.zero_grad()
            self.multitask_qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1.0 - terms_flat) * self.discount * target_v_values
        reward_target = rewards_flat / self.reward_scale
        if self.use_learned_reward:
            task_vectors = np.array([self.train_goals[i] for i in indices])
            task_vectors = ptu.from_numpy(task_vectors)
            task_vectors = [v.repeat(b, 1) for v in task_vectors]
            task_vectors = torch.cat(task_vectors, dim=0)
            reward_pred = self.rf(obs, actions, task_vectors)
            rf_loss = self.rf_criterion(reward_pred, reward_target)
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean(
            (q2_pred - q_target) ** 2
        )
        if self.multitask_q:
            multitask_q_target = (
                rewards_flat
                + (1.0 - terms_flat) * self.discount * multitask_target_v_values
            )
            multitask_qf_loss = torch.mean(
                (multitask_q1_pred - multitask_q_target) ** 2
            ) + torch.mean((multitask_q2_pred - multitask_q_target) ** 2)
            multitask_qf_loss.backward()
        qf_loss.backward()
        if self.use_learned_reward:
            rf_loss.backward()
            self.rf_optimizer.step()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        if self.multitask_q:
            self.multitask_qf1_optimizer.step()
            self.multitask_qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)
        if self.multitask_q:
            multitask_min_q_new_actions = self._min_q(
                obs, new_actions, task_z, task_idx=task_vectors
            )

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        if self.multitask_q:
            multitask_v_target = multitask_min_q_new_actions - log_pi
            multitask_vf_loss = self.vf_criterion(
                multitask_v_pred, multitask_v_target.detach()
            )
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        if self.multitask_q:
            self.multitask_vf_optimizer.zero_grad()
            multitask_vf_loss.backward()
            self.multitask_vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics["Z mean train"] = z_mean
                self.eval_statistics["Z variance train"] = z_sig
                self.eval_statistics["KL Divergence"] = ptu.get_numpy(kl_div)
                self.eval_statistics["KL Loss"] = ptu.get_numpy(kl_loss)

            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            if self.multitask_q:
                self.eval_statistics["Multitask QF Loss"] = np.mean(
                    ptu.get_numpy(multitask_qf_loss)
                )
                self.eval_statistics["Multitask VF Loss"] = np.mean(
                    ptu.get_numpy(multitask_vf_loss)
                )
            self.eval_statistics["VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
            if self.use_learned_reward:
                self.eval_statistics["RF Loss"] = np.mean(ptu.get_numpy(rf_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict("Q Predictions", ptu.get_numpy(q1_pred))
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("V Predictions", ptu.get_numpy(v_pred))
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Log Pis", ptu.get_numpy(log_pi))
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Policy mu", ptu.get_numpy(policy_mean))
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std", ptu.get_numpy(policy_log_std)
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Reward", ptu.get_numpy(rewards_flat / self.reward_scale)
                )
            )

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot
