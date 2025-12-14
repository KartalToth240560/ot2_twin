from stable_baselines3 import PPO
import gym
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

env = gym.make('Pendulum-v1',g=9.81)


import os

os.environ['WANDB_API_KEY'] = '8e799aa123be181d9ce1b5bfd3e52bbc6da7a4b5'

import wandb
from wandb.integration.sb3 import WandbCallback

# initialize wandb project
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

time_steps = 100000

for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")