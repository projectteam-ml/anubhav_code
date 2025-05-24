import argparse
import math
import random
from typing import Type
import csv
import os


import numpy as np
import pandas as pd
import torch

from tensorboardX import SummaryWriter

from diffusion import Diffusion
from model import TransformerDenoiser, DoubleCritic
from policy import DiffusionOPT
import seaborn as sns

from UAV import Environment


# writer = SummaryWriter("Loss")
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-len', type=int, default=10,
                        help="Number of past actions to feed into the Transformer")
    parser.add_argument("--exploration-noise", type=float, default=0.01) # default=0.01
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=5e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=5)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # whether the expert action is availiable
    parser.add_argument('--expert-coef', default=True)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)#0.6
    parser.add_argument('--prior-beta', type=float, default=0.4)#0.4

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args

def main(args=get_args()):
    # create environments
    env = Environment()
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.

    args.exploration_noise = args.exploration_noise * args.max_action

    # create actor
    actor_net = TransformerDenoiser(
    state_dim   = args.state_shape,
    action_dim  = args.action_shape,
    hidden_dim  = 128,
    n_heads     = 4,
    n_layers    = 3,
    dropout     = 0.1,
    history_len = args.history_len,
     ).to(args.device)

    # wrap actor in diffusion process
    actor = Diffusion(
        state_dim     = args.state_shape,
        action_dim    = args.action_shape,
        model         = actor_net,
        max_action    = args.max_action,
        beta_schedule = args.beta_schedule,
        n_timesteps   = args.n_timesteps,
        clip_denoised  = True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(),
                                   lr=args.actor_lr,
                                   weight_decay=args.wd)

    # Create critic
    critic1 = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim1 = torch.optim.Adam(
        critic1.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    # Create critic
    critic2 = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim2 = torch.optim.Adam(
        critic2.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

     policy with history support
    policy = DiffusionOPT(
        state_dim      = args.state_shape,
        actor          = actor,
        actor_optim    = actor_optim,
        action_dim     = args.action_shape,
        critic1        = critic1,
        critic_optim1  = critic_optim1,
        critic2        = critic2,
        critic_optim2  = critic_optim2,
        device         = args.device,
        tau            = args.tau,
        gamma          = args.gamma,
        estimation_step= args.n_step,
        lr_decay       = args.lr_decay,
        lr_maxt        = args.epoch,
        expert_coef    = args.expert_coef,
        history_len    = args.history_len,     # <-- pass history_len here too
    )

    total_steps = 0
    start_epsilon = 1
    end_epsilon = 0
    epsilon_steps = 5
    # writer = SummaryWriter("GDMTD3")
    max_episode_steps = 150
    
    log_file = "training_log.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
        "Total Episode",
        "Average Reward",
        "Average Secure AoI",
        "Average UAV Energy Consumption",
        "Actor Loss",
        "Critic Loss",
        "Bmin [bits]",
        "Energy Buffer Capacity [mJ]"
    ])

    for i_episode in range(10000):
        # print("-----------------------------------------------------------------------------")
        state = env.reset()
        state = np.asarray(state, dtype=np.float32)
        epsilon = end_epsilon + (start_epsilon - end_epsilon) * \
                       math.exp(-1. * i_episode / 30)

        total_critic_loss = 0
        total_actor_loss = 0
        time = 0

        Reward = 0
        Rate = 0
        Power = 0
        done = False
        t = 0

        while not done:
            if random.random() < epsilon:
                action = np.zeros(args.action_shape)
                for n in range(args.action_shape):
                    action[n] = random.uniform(-1,1)
            else:
                action = policy.select_action(state,i_episode)
            next_state, reward,done, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            Reward += reward



            t += 1
            # adding in memory
            policy.add_samples(state, action, reward, next_state, terminal=done)
            state = next_state

            # train the DDPG agent if needed
            if total_steps > policy.batch_size*5:
                # print("Update")
                # print(actor_net.get_params())
                loss = policy.learn(t)
                # print(actor_net.get_params())
                critic_loss = loss.get("loss/critic")
                actor_loss = loss.get("overall_loss")
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                # print(loss.get("loss/critic"),loss.get("overall_loss"))


            total_steps += 1
            time += 1

            # writer.add_scalar("Reward",Reward/100,i_episode*100+t)
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Convert energy list to raw float if needed
            avg_energy = env.avg_E
            if isinstance(avg_energy, list) or isinstance(avg_energy, np.ndarray):
                avg_energy = avg_energy[0]  # extract the value if it's in a list
            writer.writerow([
                i_episode,                       
                Reward/t,                   
                env.avg_A,                    
                env.avg_E[0],
                total_actor_loss / t if t > 0 else 0,
                total_critic_loss / t if t > 0 else 0,
                # Bmin,                         
                # energy_buffer_capacity        
            ])
        # print("Episode:",i_episode,"Reward:",Reward/100, "Rate:",Rate/100, "Power",Power/100)
        print("Episode:", i_episode, "Reward:", Reward / t)
        print("------------------------------------------------------------------------------------------------")
        



main()







