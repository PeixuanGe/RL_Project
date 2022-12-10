import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env
from atari_wrapper import wrap_deepmind
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import DQNPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
import cv2
import gym
#import envpool
from collections import deque

frames = deque([], maxlen=4)

def get_frames(obs, device):
    obs = np.vstack([frames[0], frames[1], frames[2], frames[3]])
    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)
    return obs


def preprocessing(img):
      img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
      # img = cv2.resize(img, (84,110), interpolation=cv2.INTER_AREA)[18:102,:]
      img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
      img = img.reshape((1, 84, 84))
      return img

def play(env, policy, video_path, device, task):

    #env = gym.wrappers.RecordVideo(env, video_path)
    obs = env.reset()
    #print(obs)
    #obs = obs[0]
    #for _ in range(4):
        #frames.append(preprocessing(obs))

    info = None

    writer = cv2.VideoWriter(task+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (160, 210))
    counter = 0
    total_rew = 0
    while True:
        counter += 1
        img = env.render(mode='rgb_array')

        #frames.append(preprocessing(obs))
        #obs = get_frames(obs, device)
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)
        obs = Batch(obs=obs, info=info)

        action = policy(obs)['act'][0]

        obs, rew, done, info = env.step(action)

        total_rew += rew
        writer.write(img)

        if done:
            print(total_rew)
            env.close()
            writer.release()
            break
            #obs = env.reset()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Zaxxon-v5")
    parser.add_argument("--double", type=int, default=0) #1 if true
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100) #the number of episodes for one policy evaluation
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward"
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    video_path = './'
    #env = gym.make(args.task, render_mode='rgb_array')
    env = wrap_deepmind("AlienNoFrameskip-v4",
                    episode_life=False,
                    clip_rewards=False,
                    frame_stack=4,
                    scale=False,
                    warp_frame=True
                        )

    action_shape = env.action_space.shape or env.action_space.n
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = DQN(4, 84, 84, action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
        is_double=1
    )
    #policy.load_state_dict(torch.load(args.load_path, map_location=args.device))
    policy.load_state_dict(torch.load('policy.pth', map_location=args.device))
    policy.eval()

    play(env, policy, video_path, args.device, 'Alien')