from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from helpers import SinusoidalPosEmb
# from tianshou.data import Batch, ReplayBuffer, to_torch

class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Tanh if activation == 'mish' else nn.ReLU
        # _act = nn.Tanh if activation == 'mish' else nn.Tanh
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # print("state:",state)
        state = state.float()
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        # print(x.shape,time.shape,processed_state.shape)
        x = torch.cat([x, t, processed_state], dim=1)
        # print(x.shape)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        # return torch.tanh(x)
        return x

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([torch.tensor(v).flatten() for v in
                                   self.parameters()]))

class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        history_len: int = 10,
    ):
        super().__init__()
        self.history_len = history_len
        self.hidden_dim  = hidden_dim

        # --- Embeddings ---
        self.state_embed  = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        # <-- swap in SinusoidalPosEmb here:
        self.time_embed   = SinusoidalPosEmb(hidden_dim)

        # Positional embeddings for [state, time, action_0…action_{H-1}]
        self.pos_embedding = nn.Parameter(
            torch.zeros(history_len + 2, hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # --- Transformer Encoder (PreNorm + GELU + Dropout) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = hidden_dim,
            nhead           = n_heads,
            dim_feedforward = hidden_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers
        )

        # Final norm + deeper MLP head
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        state: torch.Tensor,           # [B, state_dim]
        t:     torch.Tensor,           # [B] or [B,1]
        action_history: torch.Tensor,  # [B, H, action_dim]
    ) -> torch.Tensor:
        B, H, A = action_history.shape
        assert H <= self.history_len, \
            f"History length {H} exceeds max {self.history_len}"

        # --- Embed tokens ---
        s_e = self.state_embed(state)                              # [B, D]
        a_e = self.action_embed(action_history)                    # [B, H, D]

        # Flatten t to shape [B]
        if t.ndim > 1:
            t_flat = t.squeeze(-1)
        else:
            t_flat = t
        t_e = self.time_embed(t_flat)                              # [B, D]

        # --- Assemble [state, time, actions...] → [B, H+2, D] ---
        seq = torch.cat([
            s_e.unsqueeze(1),  # [B,1,D]
            t_e.unsqueeze(1),  # [B,1,D]
            a_e                 # [B,H,D]
        ], dim=1)              # [B, H+2, D]

        # --- Add positional embeddings ---
        pos = self.pos_embedding[: H + 2].unsqueeze(0)             # [1, H+2, D]
        seq = seq + pos                                            # [B, H+2, D]

        # --- Transformer (batch_first=True) → [B, H+2, D] ---
        y = self.transformer(seq)

        # --- Pool & normalize ---
        pooled = y.mean(dim=1)    # mean over sequence dim → [B, D]
        pooled = self.norm(pooled)

        # --- Output head → [B, action_dim] ---
        return self.output_head(pooled)



class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        _act = nn.Tanh if activation == 'mish' else nn.ReLU
        # _act = nn.Tanh if activation == 'mish' else nn.Tanh
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
    # def forward(self, obs):
    #     return self.q1_net(obs), self.q2_net(obs)
    #
    # def q_min(self, obs):
    #     return torch.min(*self.forward(obs))
    def forward(self, state, action):
        # state = to_torch(state, device='cpu', dtype=torch.float32)
        # action = to_torch(action, device='cpu', dtype=torch.float32)
        state = state.float()
        action = action.float()
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x)

    def q_min(self, obs, action):
        # obs = to_torch(obs, device='cuda:0', dtype=torch.float32)
        # action = to_torch(action, device='cuda:0', dtype=torch.float32)
        return torch.min(*self.forward(obs, action))
