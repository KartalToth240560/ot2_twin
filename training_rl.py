import argparse
import os

# --- Suppress Tensorflow Warning --- 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wandb
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

# Import your custom environment class
from ot2_env_wrapper import OT2Env

from clearml import Task

task = Task.init(project_name='Mentor Group - Uther/Group 1', # NB: Replace YourName with your own name
                    task_name='Experiment1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=1000)


    
# PROTECT THE ENTRY POINT (Required for Windows Multiprocessing)
if __name__ == "__main__":
    args = parser.parse_args()

    # --- ENVIRONMENT SETUP ---
    # make_vec_env automatically adds the 'Monitor' wrapper for you!
    # SubprocVecEnv puts each simulation in a separate process to prevent the crash.
    
    # 1. Training Environment
    env = make_vec_env(OT2Env, n_envs=1, vec_env_cls=SubprocVecEnv)

    # 2. Evaluation Environment
    eval_env = make_vec_env(OT2Env, n_envs=1, vec_env_cls=SubprocVecEnv)

    # --- WANDB SETUP ---
    os.environ['WANDB_API_KEY'] = '8e799aa123be181d9ce1b5bfd3e52bbc6da7a4b5'
    
    
    run = wandb.init(
        project="ot2_twin", 
        sync_tensorboard=True,
        monitor_gym=True 
    )

    # --- MODEL SETUP ---
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        n_steps=args.n_steps, 
        n_epochs=args.n_epochs, 
        tensorboard_log=f"runs/{run.id}"
    )

    # --- CALLBACK SETUP ---
    save_freq = int(args.total_timesteps / 10)

    # 1. Stop Training on Reward Threshold
    stop_train_callback = StopTrainingOnRewardThreshold(
        reward_threshold=-0.001, 
        verbose=1
    )

    # 2. Eval Callback
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=stop_train_callback, 
        eval_freq=2048, 
        best_model_save_path=f"models/{run.id}/best_model", 
        log_path=f"runs/{run.id}", 
        deterministic=True, 
        render=False
    )

    # 3. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"models/{run.id}/checkpoints",
        name_prefix="rl_model"
    )

    # 4. Wandb Callback
    wandb_callback = WandbCallback(
        model_save_freq=save_freq,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    callbacks_list = [eval_callback, checkpoint_callback, wandb_callback]

    # --- START TRAINING ---
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callbacks_list, 
            progress_bar=True, 
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}"
        )
        
        # Save final model
        model.save(f"models/{run.id}/final_model")
        task.upload_artifact("model", artifact_object=f"{run.id}.zip")
        print("Training finished successfully.")
        
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        run.finish()