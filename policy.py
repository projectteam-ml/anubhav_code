import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union

from torch import tensor
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from UAV import Environment
from helpers import (
    Losses
)
from memory import Memory

env = Environment()

class DiffusionOPT():

    def __init__(
        self,
        state_dim,
        actor,
        actor_optim,
        action_dim,
        critic1,
        critic_optim1,
        critic2,
        critic_optim2,
        device,
        tau: float = 0.005,
        gamma: float = 0.9,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        lr_decay: bool = False,
        lr_maxt: int = 1000,
        expert_coef: bool = False,
        history_len: int = 10,
        **kwargs
    ) -> None:
        super().__init__()
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        self.history_len = history_len

        # Actor networks
        if actor is not None and actor_optim is not None:
            self._actor = actor
            self._target_actor = copy.deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim = actor_optim
            self._action_dim = action_dim

        # Critic networks
        if critic1 is not None and critic_optim1 is not None:
            self._critic1 = critic1
            self._target_critic1 = copy.deepcopy(critic1)
            self._critic_optim1 = critic_optim1
            self._target_critic1.eval()
        if critic2 is not None and critic_optim2 is not None:
            self._critic2 = critic2
            self._target_critic2 = copy.deepcopy(critic2)
            self._critic_optim2 = critic_optim2
            self._target_critic2.eval()

        # LR schedulers
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler1 = CosineAnnealingLR(self._critic_optim1, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler2 = CosineAnnealingLR(self._critic_optim2, T_max=lr_maxt, eta_min=0.)

        # History-aware memory
        self.memory = MemoryV2(
            limit=10**6,
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            history_len=self.history_len
        )

        self._steps = 0
        self.batch_size = 128
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._expert_coef = expert_coef
        self._device = device
        self.clip_grad = 10

    def select_action(self, state, action_history):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)
        hist = torch.from_numpy(action_history).unsqueeze(0).to(self._device)
        logits = self._actor(state, hist).detach().cpu().numpy()[0]
        return np.clip(logits, -1, 1)

    def update_critic(self, states, actions, rewards, next_states, terminals, next_hist):
        next_actions = self._target_actor(next_states, next_hist)
        target_Q1 = self._target_critic1(next_states, next_actions)
        target_Q2 = self._target_critic2(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - terminals) * self._gamma * target_Q.detach()

        current_Q1 = self._critic1(states, actions)
        loss_Q1 = nn.functional.mse_loss(current_Q1, target_Q)
        self._critic_optim1.zero_grad()
        loss_Q1.backward()
        self._critic_optim1.step()

        current_Q2 = self._critic2(states, actions)
        loss_Q2 = nn.functional.mse_loss(current_Q2, target_Q)
        self._critic_optim2.zero_grad()
        loss_Q2.backward()
        self._critic_optim2.step()

        return loss_Q1

    def update_policy(self, states, action_hist):
        actions = self._actor(states, action_hist)
        actions.requires_grad_(True)
        Q = self._critic1(states, actions)
        pg_loss = -Q.mean()

        self._actor.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.clip_grad)
        self._actor_optim.step()

        return pg_loss

    def add_samples(self,state, action, reward, next_state, terminal):
        self._steps += 1
        self.memory.append(state, action, reward, next_state, terminal=terminal)    

    def soft_update(self, tgt, src, tau):
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic1, self._critic1, self._tau)
        self.soft_update(self._target_critic2, self._critic2, self._tau)

    def learn(self, actor_update_freq):
        data = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, terminals, hist, next_hist = data

        # to device
        states     = torch.from_numpy(states).to(self._device)
        actions    = torch.from_numpy(actions).to(self._device)
        rewards    = torch.from_numpy(rewards).to(self._device)
        next_states= torch.from_numpy(next_states).to(self._device)
        terminals  = torch.from_numpy(terminals).to(self._device)
        hist       = torch.from_numpy(hist).to(self._device)
        next_hist  = torch.from_numpy(next_hist).to(self._device)

        critic_loss = self.update_critic(states, actions, rewards, next_states, terminals, next_hist)
        if actor_update_freq % 5 == 0:
            pg_loss = self.update_policy(states, hist)
        else:
            pg_loss = tensor(0.0)

        self.update_targets()
        return {"loss/critic": critic_loss.item(), "loss/actor": pg_loss.item()}
