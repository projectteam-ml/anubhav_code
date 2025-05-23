# Import necessary libraries
import inspect
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import additional helper functions and utils
from helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from util1 import to_numpy
from utils import Progress, Silent

# Define the main Diffusion class that inherits from PyTorch's nn.Module
class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        model,
        max_action,
        beta_schedule='vp',
        n_timesteps=5,
        loss_type='l1',
        clip_denoised=False,
        explore_solution=False
    ):
        super(Diffusion, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        # beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.explore_solution = explore_solution

        # register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def predict_start_from_noise(self, x_t, t, noise):
        if self.explore_solution:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def get_params(self):
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in self.parameters()]))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Updated to accept action history along with state and timestep
    def p_mean_variance(self, x, t, state, action_history):
        # model now expects (state, t, action_history)
        noise_pred = self.model(state, t, action_history)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if self.clip_denoised:
            x_recon = x_recon.clamp(-self.max_action, self.max_action)
        else:
            raise RuntimeError("clip_denoised=False is not supported")

        model_mean, var, log_var = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, var, log_var

    def p_sample(self, x, t, state, action_history):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_var = self.p_mean_variance(x, t, state, action_history)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise

    def p_sample_loop(self, state, action_history, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        if return_diffusion:
            diffusion = [x]
        prog = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, action_history)
            prog.update({'t': i})
            if return_diffusion:
                diffusion.append(x)
        prog.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    # Updated to require action history
    def sample(self, state, action_history, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, action_history, shape, *args, **kwargs)
        return action.clamp(-self.max_action, self.max_action)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # Updated to pass action history into loss computation
    def p_losses(self, x_start, state, t, action_history, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # model predicts noise given state and history
        noise_pred = self.model(state, t, action_history)
        if self.explore_solution:
            return self.loss_fn(noise_pred, noise, weights)
        else:
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=noise_pred)
            return self.loss_fn(x_recon, x_start, weights)

    # Updated to accept history
    def loss(self, x, state, action_history, weights=1.0):
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, action_history, weights)

    # Updated forward signature
    def forward(self, state, action_history, *args, **kwargs):
        return self.sample(state, action_history, *args, **kwargs)
